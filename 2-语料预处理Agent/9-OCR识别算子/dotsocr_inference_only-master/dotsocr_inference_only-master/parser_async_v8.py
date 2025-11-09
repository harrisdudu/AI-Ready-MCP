import os
import json
import time
import math
import re
import uuid
import base64
import random
import asyncio
import argparse
from pathlib import Path
from io import BytesIO
from collections import defaultdict
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

from tqdm.asyncio import tqdm
from tqdm import tqdm as TQDM_BASE
from PIL import Image
Image.MAX_IMAGE_PIXELS = 600000000
import fitz
import functools
import shutil
from datetime import datetime

from dots_ocr.model.inference_async import get_async_client, warmup_gpu, close_all_clients
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.doc_utils import fitz_doc_to_image, is_blank_page
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, pre_process_bboxes, draw_layout_on_image
from dots_ocr.utils.format_transformer_v2 import layoutjson2md, layoutjson2md_full_robust, layoutjson2md_simple_extract

#  断点续传功能
def is_file_already_processed(input_path, output_dir, filename):
    """
    检查文件是否已经被处理过（存在对应的MD文件）
    """
    try:
        # 构建预期的输出目录和MD文件路径
        save_dir = os.path.join(output_dir, filename)
        md_file_path = os.path.join(save_dir, f"{filename}.md")
        
        # 检查MD文件是否存在
        if os.path.exists(md_file_path):
            # 检查MD文件是否为空（只有大于0字节才算有效）
            if os.path.getsize(md_file_path) > 0:
                return True, md_file_path
            else:
                TQDM_BASE.write(f"Found empty MD file: {md_file_path}, will reprocess")
                return False, md_file_path
        return False, None
    except Exception as e:
        TQDM_BASE.write(f"Error checking if file is processed: {e}")
        return False, None

def get_processed_files(input_dir, output_dir):
    """
    获取输入目录中已处理的文件列表
    """
    processed_files = set()
    try:
        input_root = Path(input_dir).resolve()
        output_root = Path(output_dir).resolve()
        
        # 遍历输出目录，找出所有MD文件
        for md_file in output_root.rglob("*.md"):
            try:
                # 获取MD文件相对于输出目录的路径
                relative_path = md_file.relative_to(output_root)
                # MD文件名（不含扩展名）就是输入文件名
                filename = relative_path.stem
                # 构建对应的输入文件路径
                # 注意：MD文件可能在 output_dir/filename/filename.md 或 output_dir/filename.md
                if relative_path.parent == Path("."):
                    # MD文件直接在输出目录下：output_dir/filename.md
                    possible_input_paths = [
                        input_root / f"{filename}.pdf",
                        input_root / f"{filename}.png",
                        input_root / f"{filename}.jpg",
                        input_root / f"{filename}.jpeg",
                        input_root / f"{filename}.tiff",
                        input_root / f"{filename}.bmp",
                    ]
                else:
                    # MD文件在子目录下：output_dir/filename/filename.md
                    possible_input_paths = [
                        input_root / f"{filename}.pdf",
                        input_root / f"{filename}.png",
                        input_root / f"{filename}.jpg",
                        input_root / f"{filename}.jpeg",
                        input_root / f"{filename}.tiff",
                        input_root / f"{filename}.bmp",
                    ]
                
                for input_path in possible_input_paths:
                    if input_path.exists():
                        processed_files.add(str(input_path))
                        break
            except Exception as e:
                TQDM_BASE.write(f"Error processing MD file {md_file}: {e}")
                continue
    except Exception as e:
        TQDM_BASE.write(f"Error getting processed files: {e}")
    
    return processed_files

# ==============================================================================
#  图片-标题匹配及命名逻辑
# ==============================================================================
def calculate_center_distance(bbox1, bbox2):
    x1_center = (bbox1[0] + bbox1[2]) / 2; y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2; y2_center = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
def calculate_reading_order_score(bbox1, bbox2, index1, index2):
    return abs(index1 - index2) * 85 + (0 if bbox2[1] > bbox1[3] else 30)
def is_spatially_close(bbox1, bbox2, max_distance_ratio=1.8):
     # 计算两个bbox的中心点距离
    center_distance = calculate_center_distance(bbox1, bbox2)

    # 计算参考尺寸
    width1 = bbox1[2] - bbox1[0]; height1 = bbox1[3] - bbox1[1]
    width2 = bbox2[2] - bbox2[0]; height2 = bbox2[3] - bbox2[1]
    avg_dimension = (width1 + width2 + height1 + height2) / 4

    # 基于中心点距离
    if center_distance <= avg_dimension * max_distance_ratio: return True

    # --- 处理垂直对齐（上下）的情况 ---
    x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    overlap_ratio = x_overlap / min(width1, width2) if min(width1, width2) > 0 else 0
    if overlap_ratio > 0.3: # 至少30%的水平重叠
        vertical_distance = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
        if vertical_distance < max(height1, height2) * 1.0: return True

    # --- 处理水平对齐（左右）的情况 ---
    y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    y_overlap_ratio = y_overlap / min(height1, height2) if min(height1, height2) > 0 else 0
    if y_overlap_ratio > 0.3: # 至少30%的垂直重叠（保证在同一水平线上）
        horizontal_distance = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
        if horizontal_distance < max(width1, width2) * 0.5: return True # 水平间距小于最大宽度的一半
    return False
def get_relative_position_score(bbox1, bbox2):
    """计算相对位置得分（caption通常在图片下方或上方）"""
    y1_center = (bbox1[1] + bbox1[3]) / 2; y2_center = (bbox2[1] + bbox2[3]) / 2
    vertical_distance = abs(y1_center - y2_center)
    x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    return vertical_distance * 0.5 if x_overlap > 0 else vertical_distance * 1.5
def match_elements_with_captions(cells, element_category):
    """通用匹配函数：匹配指定类型的元素和标题"""
    elements = [{'index': i, 'cell': cell} for i, cell in enumerate(cells) if cell['category'] == element_category]
    captions = [{'index': i, 'cell': cell} for i, cell in enumerate(cells) if cell['category'] == 'Caption']
    matches = []; used_captions = set()
    for element in elements:
        candidate_captions = []
        for caption in captions:
            if caption['index'] in used_captions or not is_spatially_close(element['cell']['bbox'], caption['cell']['bbox']): continue
            reading_score = calculate_reading_order_score(element['cell']['bbox'], caption['cell']['bbox'], element['index'], caption['index'])
            center_distance = calculate_center_distance(element['cell']['bbox'], caption['cell']['bbox'])
            position_score = get_relative_position_score(element['cell']['bbox'], caption['cell']['bbox'])
            score = reading_score + position_score + center_distance * 0.1
            candidate_captions.append({'caption': caption, 'score': score})
        if candidate_captions:
            best_match = min(candidate_captions, key=lambda x: x['score'])
            matches.append({'element': element, 'caption_text': best_match['caption']['cell'].get('text', '').strip()})
            used_captions.add(best_match['caption']['index'])
        else:
            matches.append({'element': element, 'caption_text': None})
    return matches
def generate_image_name(picture_cell, picture_index, page_num):
    """Generates an image filename based on its pre-matched caption."""
    if picture_cell and picture_cell.get('caption_text'):
        caption_text = picture_cell['caption_text']
        caption_text = re.sub(r'(\d+)\.(\d+)', r'\1-\2', caption_text)
        safe_caption = re.sub(r'[^\w\s-]', '', caption_text).strip()
        safe_caption = re.sub(r'[-\s]+', '_', safe_caption)
        if safe_caption:
            return f"page_{page_num}_{safe_caption[:50]}"
    return f"page_{page_num}_img_{picture_index + 1}"

# ==============================================================================
#  表格截图功能
# ==============================================================================
def extract_table_screenshot(image, table_cell, page_num, table_index):
    """
    Extracts a table screenshot from the original image and names it
    based on its pre-matched caption.
    """
    try:
        table_bbox = table_cell['bbox']
        img_width, img_height = image.size
        x1, y1, x2, y2 = max(0, int(table_bbox[0])), max(0, int(table_bbox[1])), min(img_width, int(table_bbox[2])), min(img_height, int(table_bbox[3]))
        
        table_image = image.crop((x1, y1, x2, y2))
        
        filename_base = f"page_{page_num}_table_{table_index + 1}" # Default name
        if table_cell.get('caption_text'):
            caption_text = table_cell['caption_text']
            caption_text = re.sub(r'(\d+)\.(\d+)', r'\1-\2', caption_text)
            safe_caption = re.sub(r'[^\w\s-]', '', caption_text).strip()
            safe_caption = re.sub(r'[-\s]+', '_', safe_caption)
            if safe_caption:
                filename_base = f"page_{page_num}_table_{safe_caption[:50]}"
        
        return table_image, filename_base
    except Exception as e:
        TQDM_BASE.write(f"Error extracting table screenshot: {e}")
        return None, None

# ==============================================================================
#  辅助类
# ==============================================================================
class PerformanceMonitor:
    def __init__(self):
        self.stats = defaultdict(list); self.start_time = time.time()
    def record_inference_time(self, time_taken): self.stats['inference_times'].append(time_taken)
    def record_error(self, error_type): self.stats['errors'].append(error_type)
    def record_retry(self, attempt_count): self.stats['retry_attempts'].append(attempt_count)
    def get_summary(self):
        total_time = time.time() - self.start_time; inference_times = self.stats['inference_times']
        retry_attempts = self.stats.get('retry_attempts', []); summary = {'total_time': total_time, 'total_requests': len(inference_times), 'avg_inference_time': sum(inference_times) / len(inference_times) if inference_times else 0, 'total_errors': len(self.stats['errors']), 'total_retries': sum(retry_attempts) if retry_attempts else 0, 'error_types': defaultdict(int)}
        for error in self.stats['errors']: summary['error_types'][error] += 1
        return summary
    
# ==============================================================================
#  独立的PDF页面处理工作函数（用于进程池）
# ==============================================================================
def _process_pdf_page_worker(task_args):
    """
    这个函数在单独的进程中运行。
    现在它接收文件路径和页码，自己完成渲染。
    """
    # 参数解包
    input_path, page_idx, dpi, skip_blank_pages, white_threshold, noise_threshold, min_pixels, max_pixels = task_args
    
    try:
        # 在工作进程中打开文档，避免跨进程传递复杂对象
        with fitz.open(input_path) as doc:
            if page_idx >= len(doc):
                return {'status': 'error', 'error': f'Page index {page_idx} is out of bounds.'}
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            origin_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # 后续逻辑不变
        if skip_blank_pages and is_blank_page(origin_image, white_threshold, noise_threshold):
            return {'status': 'skipped_blank'}
        
        processed_image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        
        return {
            'status': 'success',
            'origin_image': origin_image,
            'processed_image': processed_image
        }
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': f'Error in worker for page {page_idx} of {os.path.basename(input_path)}: {str(e)}\n{traceback.format_exc()}'
        }
    
# ==============================================================================
#  主解析器类
# ==============================================================================
class DotsOCRParserOptimized:

    # 统一定义所有被认为是“成功”处理的状态
    SUCCESS_STATUSES = {
        'success', 
        'skipped_blank', 
        'success_fallback_text', 
        'success_fallback_image'
    }

    # 将正则表达式编译为类属性
    # 对数字列表的匹配增加了先行断言，要求数字后必须跟分隔符（点、顿号、空格等）
    # 这样可以避免将正文中以数字开头的句子（如 "2024年..."）误判为新段落。
    NEW_PARAGRAPH_PATTERN = re.compile(
        r'^\s*('
        # 规则1: Markdown标题、缩进、特殊项目符号
        r'#+|\u3000\u3000|[■●◆➢]|'
        # 规则2: 各种形式的列表项开头 (核心改进)
        r'[([（【].*?[)）】]|'  # 匹配任何被括号包围的内容，如 [第 1565 号]
        r'\d+[.、)\s]|'          # 匹配 "1."、"2)" 等
        r'[a-zA-Z][.)\s]|'        # 匹配 "A."、"B)" 等
        r'[一二三四五六七八九十百千]+[、.)\s]|' # 匹配 "一、"、"二." 等
        r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]' # 匹配带圈数字
        r')'
    )

    english_letter_pattern = re.compile(r'^[a-zA-Z]')

    def __init__(self, **kwargs):
        self.ip = kwargs.get('ip', 'localhost')
        self.port = kwargs.get('port', 8000)
        self.model_name = kwargs.get('model_name', 'model')
        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_completion_tokens = kwargs.get('max_completion_tokens', 16384)
        self.dpi = kwargs.get('dpi', 200)
        self.output_dir = kwargs.get('output_dir', './output')
        self.min_pixels = kwargs.get('min_pixels', None)
        self.max_pixels = kwargs.get('max_pixels', None)
        self.timeout = kwargs.get('timeout', 900.0)
        self.use_hf = kwargs.get('use_hf', False)
        self.enable_warmup = kwargs.get('enable_warmup', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.debug_matching = kwargs.get('debug_matching', False)
        self.blank_white_threshold = kwargs.get('blank_white_threshold', 0.98)
        self.blank_noise_threshold = kwargs.get('blank_noise_threshold', 0.002)
        self.page_concurrency = kwargs.get('page_concurrency', 24)
        self.num_cpu_workers = kwargs.get('num_cpu_workers', 16)
        self.queue_size = kwargs.get('queue_size', 500)
        self.api_key = kwargs.get('api_key', os.environ.get("API_KEY", "0"))
        self.enable_resume = kwargs.get('enable_resume', True)  # 启用断点续传
        self.force_reprocess = kwargs.get('force_reprocess', False)  # 强制重新处理
        self.concurrent_retries = kwargs.get('concurrent_retries', 4) # 并发重试次数
        self.enable_table_screenshot = kwargs.get('enable_table_screenshot', False) # 控制是否保存表格截图
        self.skip_uncaptioned_images = kwargs.get('skip_uncaptioned_images', False) # 如果图片没有标题则跳过
        self.badcase_collection_dir = kwargs.get('badcase_collection_dir', None) # badcase收集目录
        self.keep_page_header = kwargs.get('keep_page_header', False) # 保留页眉开关，默认不保留
        self.keep_page_footer = kwargs.get('keep_page_footer', False) # 保留页脚开关，默认不保留
        self.save_page_json = kwargs.get('save_page_json', False) # 控制是否保存单页JSON
        self.save_page_layout = kwargs.get('save_page_layout', False) # 控制是否保存布局图
        self.add_page_tag = kwargs.get('add_page_tag', False) # 控制是否页与页之间加入特殊标识符
        self.client = None
        self.monitor = PerformanceMonitor()
        
        ## 创建一个全局信号量来精确控制API总并发数。
        self.page_semaphore = asyncio.Semaphore(self.page_concurrency)
        
        num_workers = self.num_cpu_workers if self.num_cpu_workers > 0 else max(os.cpu_count() // 2, 16)
        print(f"Initializing ProcessPoolExecutor with {num_workers} workers.")
        self.process_pool = ProcessPoolExecutor(max_workers=num_workers)
        
        print(f"Using vLLM model with optimized producer-consumer model.")
        print(f"Global Page API Concurrency Limit: {self.page_concurrency}, CPU Workers: {num_workers}, Queue Size: {self.queue_size}")
        print(f"Feature: Save page JSON -> {'ENABLED' if self.save_page_json else 'DISABLED'}")
        print(f"Feature: Save page layout image -> {'ENABLED' if self.save_page_layout else 'DISABLED'}")
        
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    async def initialize(self):
        if not self.use_hf:
            print("Initializing async client...")
            max_connections = int(self.page_concurrency * 1.2) + 10
            self.client = await get_async_client(self.ip, self.port, self.timeout, max_connections, self.api_key)
            if self.enable_warmup:
                await warmup_gpu(self.ip, self.port, self.model_name, api_key=self.api_key)
        return self

    async def shutdown(self):
        print("Shutting down resources...")
        await close_all_clients()
        self.process_pool.shutdown(wait=True)
        print("Shutdown complete.")

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None; bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt
    
    # (在 DotsOCRParserOptimized 类中)
    async def _save_intermediate_outputs_async(self, save_dir, save_name_base, response, cells, origin_image):
        """
        根据开关异步保存单页的原始JSON响应和/或布局图。
        """
        if not self.save_page_json and not self.save_page_layout:
            return None, None

        loop = asyncio.get_running_loop()
        json_path, layout_path = None, None

        def _save_sync():
            nonlocal json_path, layout_path
            
            # 1. 保存原始JSON响应
            if self.save_page_json:
                json_data_to_save = None
                if isinstance(response, str):
                    try:
                        json_data_to_save = json.loads(response)
                    except json.JSONDecodeError:
                        json_data_to_save = {"raw_response": response}
                else:
                    json_data_to_save = response

                json_path = os.path.join(save_dir, f"{save_name_base}.json")
                with open(json_path, 'w', encoding="utf-8") as f:
                    json.dump(json_data_to_save, f, ensure_ascii=False, indent=2)

            # 2. 保存布局图
            if self.save_page_layout and origin_image and cells:
                layout_path = os.path.join(save_dir, f"{save_name_base}_layout.jpg")
                try:
                    layout_image = draw_layout_on_image(origin_image, cells)
                    if layout_image.mode != 'RGB':
                        layout_image = layout_image.convert('RGB')
                    layout_image.save(layout_path, "JPEG", quality=95)
                except Exception as e:
                    TQDM_BASE.write(f"Warning: Failed to draw layout on image for {save_name_base}. Saving original image instead. Error: {e}")
                    try:
                        fallback_image = origin_image.convert('RGB')
                        fallback_image.save(layout_path, "JPEG", quality=95)
                    except Exception as fallback_e:
                        TQDM_BASE.write(f"CRITICAL: Could not save fallback original image for {save_name_base}. Error: {fallback_e}")
                        layout_path = None # 保存失败

        await loop.run_in_executor(None, _save_sync)
        return json_path, layout_path

    # (在 DotsOCRParserOptimized 类中)
    async def _inference_with_vllm(self, image, prompt):
        start_time = time.time()
        # 移除内部重试循环，将所有异常向上抛出，由调用者处理
        try:
            from dots_ocr.model.inference_async import inference_with_vllm
            result = await inference_with_vllm(
                image, prompt, model_name=self.model_name, ip=self.ip, port=self.port, 
                temperature=self.temperature, top_p=self.top_p, max_completion_tokens=self.max_completion_tokens,
                timeout=self.timeout, client=self.client, max_retries=2, retry_delay=self.retry_delay
            )
            if result is not None:
                self.monitor.record_inference_time(time.time() - start_time)
                return result
            # 如果 result is None，也视为一种失败，并抛出异常
            raise ValueError("Inference call to vLLM returned None.")
        except Exception as e:
            # 记录错误，然后重新抛出异常，让上层重试逻辑接管
            error_type = type(e).__name__
            self.monitor.record_error(error_type)
            # TQDM_BASE.write(f"Inference failed on a single attempt: {error_type}: {e}") # 这一行可以省略，因为上层会打印更详细的重试信息
            raise e
    # (在 DotsOCRParserOptimized 类中，可以放在 _inference_with_vllm 之后)
    async def _race_inference_attempts(self, image, prompt, num_attempts: int):
        """
        同时发起多个推理请求，并返回第一个成功的结果。
        """
        if num_attempts <= 0:
            return None

        # 创建 num_attempts 个并发的推理任务
        tasks = [
            asyncio.create_task(self._inference_with_vllm(image, prompt))
            for _ in range(num_attempts)
        ]
        
        done, pending = set(), set(tasks)
        successful_result = None

        try:
            while pending and successful_result is None:
                # 等待任何一个任务完成
                done_part, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                done.update(done_part)

                # 检查已完成的任务中是否有成功的
                for task in done_part:
                    try:
                        result = task.result() # 获取任务结果
                        if result is not None:
                            successful_result = result
                            # 找到一个成功的结果，跳出循环
                            break 
                    except Exception:
                        # 如果任务失败（抛出异常），则忽略，继续等待其他任务
                        continue
        finally:
            # 无论成功与否，都要取消所有仍在挂起的任务以释放资源
            if pending:
                # TQDM_BASE.write(f"  Racing succeeded. Cancelling {len(pending)} redundant retry task(s).")
                for task in pending:
                    task.cancel()
                # 等待取消操作完成
                await asyncio.gather(*pending, return_exceptions=True)

        if successful_result:
            return successful_result
        else:
            # 如果所有并发尝试都失败了，则抛出最后一个遇到的异常，以便上层捕获
            last_exception = None
            for task in done:
                if task.exception():
                    last_exception = task.exception()
            if last_exception:
                raise last_exception
            raise Exception("All concurrent retry attempts failed without a specific exception.")

    # (在 DotsOCRParserOptimized 类中)
    def _process_base64_images_with_custom_naming(self, md_content, images_dir, page_num, cells, origin_image=None):
        """
        如果 self.skip_uncaptioned_images 为 True，则不保存没有标题的图片，并从MD内容中移除它们。
        """
        os.makedirs(images_dir, exist_ok=True)
        base64_pattern = r'!\[[^\]]*\]\s*\(\s*data:image/[^;]+;base64,([^)]+)\)'
        
        pictures = [cell for cell in cells if cell['category'] == 'Picture']
        picture_counter = 0
        
        def replace_base64(match):
            nonlocal picture_counter
            try:
                # 获取与当前base64图像匹配的 'Picture' 单元格信息
                current_pic_cell = pictures[picture_counter] if picture_counter < len(pictures) else None
                
                # 检查开关是否开启，以及当前图片是否有标题
                if self.skip_uncaptioned_images and (not current_pic_cell or not current_pic_cell.get('caption_text')):
                    picture_counter += 1  # 必须增加计数器以匹配下一个图片
                    # TQDM_BASE.write(f"Skipping image on page {page_num} because it has no caption.")
                    return ""  # 返回空字符串，从而从Markdown中删除此图片引用

                base64_data = match.group(1)
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data))

                image_filename_base = generate_image_name(current_pic_cell, picture_counter, page_num)
                
                counter = 0
                image_filename = image_filename_base
                while os.path.exists(os.path.join(images_dir, f"{image_filename}.jpg")):
                    counter += 1
                    image_filename = f"{image_filename_base}_{counter}"
                
                image_filename_with_ext = f"{image_filename}.jpg"
                image_path = os.path.join(images_dir, image_filename_with_ext)
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                image.save(image_path, 'JPEG', quality=95)
                
                picture_counter += 1 # 成功处理后，计数器增加
                return f"![{image_filename_with_ext}](images/{image_filename_with_ext})"
            
            except Exception as e:
                TQDM_BASE.write(f"Error processing base64 image: {e}")
                # 出现错误时，保持原始的base64标签，避免丢失内容
                return match.group(0)
        
        md_content = re.sub(base64_pattern, replace_base64, md_content)
        
        if self.enable_table_screenshot and origin_image and cells:
            tables = [cell for cell in cells if cell['category'] == 'Table']
            for table_index, table_cell in enumerate(tables):
                table_image, table_filename_base = extract_table_screenshot(
                    origin_image, table_cell, page_num, table_index
                )
                if table_image:
                    counter = 0
                    table_filename = table_filename_base
                    while os.path.exists(os.path.join(images_dir, f"{table_filename}.jpg")):
                        counter += 1
                        table_filename = f"{table_filename_base}_{counter}"
                    
                    table_filename_with_ext = f"{table_filename}.jpg"
                    table_path = os.path.join(images_dir, table_filename_with_ext)
                    if table_image.mode in ('RGBA', 'LA', 'P'):
                        table_image = table_image.convert('RGB')
                    table_image.save(table_path, 'JPEG', quality=95)

        return {"md_content": md_content, "page_num": page_num}
        
    # (在 DotsOCRParserOptimized 类中)
    def _perform_intra_page_matching(self, cells: List[dict]) -> List[dict]:
        """
        1. 使用中心点判断元素的【上下相对位置】，以正确处理内嵌标题。
        2. 使用边界框之间的【实际间隙】来测量【邻近距离】，以正确处理各种尺寸（尤其是非常高）的元素。
        """
        # ================= Step 0: 初始准备 =================
        picture_matches = match_elements_with_captions(cells, 'Picture')
        for match in picture_matches:
            pic_cell = match['element']['cell']
            pic_cell['is_matched'] = True
            if match['caption_text']:
                pic_cell['caption_text'] = match['caption_text']
                for cell in cells:
                    if cell.get('text', '').strip() == match['caption_text'] and cell['category'] == 'Caption':
                        cell['is_matched'] = True
                        cell['locked_for_picture'] = True
                        break

        tables = [cell for cell in cells if cell['category'] == 'Table' and not cell.get('is_matched')]
        captions = [cell for cell in cells if cell['category'] == 'Caption' and not cell.get('is_matched')]
        
        # ================= Pass 1: 高置信度匹配 (标题 -> 下方的表格) =================
        captions.sort(key=lambda c: c['bbox'][1])
        
        for cap_cell in captions:
            if cap_cell.get('is_matched'):
                continue

            best_table_candidate = None
            min_distance = float('inf')

            cap_bbox = cap_cell['bbox']
            cap_width = cap_bbox[2] - cap_bbox[0]
            y_center_cap = (cap_bbox[1] + cap_bbox[3]) / 2

            for tbl_cell in tables:
                if tbl_cell.get('is_matched'):
                    continue

                tbl_bbox = tbl_cell['bbox']
                y_center_tbl = (tbl_bbox[1] + tbl_bbox[3]) / 2
                
                # 1. 使用中心点判断【相对位置】 (解决内嵌问题)
                if y_center_tbl < y_center_cap:
                    continue
                
                # 2. 使用边界框间隙计算【邻近距离】 (解决高个子表格问题)
                # 这个距离是标题底部到表格顶部的实际空白距离。
                # 对于内嵌标题，这个值可能是负数，但只要不大，就没问题。
                vertical_distance = tbl_bbox[1] - cap_bbox[3]
                
                # 3. 基于实际间隙设置阈值
                if vertical_distance > 300: # 使用一个对“间隙”合理的阈值
                    continue

                # 规则: 必须有显著的水平重叠
                tbl_width = tbl_bbox[2] - tbl_bbox[0]
                x_overlap = max(0, min(tbl_bbox[2], cap_bbox[2]) - max(tbl_bbox[0], cap_bbox[0]))
                overlap_ratio = x_overlap / min(tbl_width, cap_width) if min(tbl_width, cap_width) > 0 else 0
                if overlap_ratio < 0.2:
                    continue

                # 评分标准依然是距离越近越好
                score = vertical_distance 
                if score < min_distance:
                    min_distance = score
                    best_table_candidate = tbl_cell

            if best_table_candidate:
                best_table_candidate['is_matched'] = True
                best_table_candidate['caption_text'] = cap_cell.get('text', '').strip()
                cap_cell['is_matched'] = True

        # ================= Pass 2: 补充匹配 =================
        tables.sort(key=lambda t: t['bbox'][1])

        for tbl_cell in tables:
            if tbl_cell.get('is_matched'):
                continue
            best_caption_candidate = None
            min_distance = float('inf')
            tbl_bbox = tbl_cell['bbox']
            tbl_width = tbl_bbox[2] - tbl_bbox[0]

            for cap_cell in captions:
                if cap_cell.get('is_matched'):
                    continue
                cap_bbox = cap_cell['bbox']
                if cap_bbox[1] < tbl_bbox[3]:
                    continue
                vertical_distance = cap_bbox[1] - tbl_bbox[3]
                if vertical_distance > 300:
                    continue
                cap_width = cap_bbox[2] - cap_bbox[0]
                x_overlap = max(0, min(tbl_bbox[2], cap_bbox[2]) - max(tbl_bbox[0], cap_bbox[0]))
                overlap_ratio = x_overlap / min(tbl_width, cap_width) if min(tbl_width, cap_width) > 0 else 0
                if overlap_ratio < 0.2:
                    continue
                
                if vertical_distance < min_distance:
                    min_distance = vertical_distance
                    best_caption_candidate = cap_cell
            
            if best_caption_candidate:
                tbl_cell['is_matched'] = True
                tbl_cell['caption_text'] = best_caption_candidate.get('text', '').strip()
                best_caption_candidate['is_matched'] = True
                
        return cells

    # (在 DotsOCRParserOptimized 类中)
    def _is_path_clean_between(self, elem1_info, elem2_info, all_pages_data, allowed_categories):
        """
        检查两个元素之间的路径是否“干净”。
        它首先对元素按阅读顺序排序，确保逻辑的普适性。
        """
        # 保证 start_info 始终在 end_info 之前
        elements = sorted([elem1_info, elem2_info], key=lambda x: (x['page_num'], x['cell']['bbox'][1]))
        start_info, end_info = elements[0], elements[1]

        start_page_idx = start_info['page_num'] - 1
        end_page_idx = end_info['page_num'] - 1

        for p_idx in range(start_page_idx, end_page_idx + 1):
            page_cells = all_pages_data[p_idx].get('cells', [])
            if not page_cells:
                continue

            y_start_check = 0
            y_end_check = float('inf')

            if p_idx == start_page_idx:
                y_start_check = start_info['cell']['bbox'][3]
            if p_idx == end_page_idx:
                y_end_check = end_info['cell']['bbox'][1]

            for cell in page_cells:
                # 跳过我们正在检查的元素本身
                if cell is start_info['cell'] or cell is end_info['cell']:
                    continue

                cell_y_center = (cell['bbox'][1] + cell['bbox'][3]) / 2
                
                if y_start_check <= cell_y_center <= y_end_check:
                    if cell['category'] not in allowed_categories:
                        if self.debug_matching:
                            TQDM_BASE.write(
                                f"[DEBUG] Path blocked on page {p_idx + 1} between elements on pg {start_info['page_num']} "
                                f"and pg {end_info['page_num']} by a '{cell['category']}' element."
                            )
                        return False
        return True

    def _is_valid_continuation_page(self, page_data, allowed_categories):
        """辅助函数：检查一个页面是否是有效的表格连续页。"""
        if not page_data.get('cells'):
            return False # 空页面不是连续页
        for cell in page_data['cells']:
            if cell['category'] not in allowed_categories:
                return False
        return True

    # (在 DotsOCRParserOptimized 类中)
    def _is_contiguous(self, elem1_info: dict, elem2_info: dict, all_pages_data: List[dict]) -> bool:
        """
        检查两个元素（通常是表格）之间是否是物理连续的。
        连续意味着它们之间只允许有页眉和页脚。
        """
        # 确保 elem1 在 elem2 之前
        if (elem1_info['page_num'], elem1_info['cell']['bbox'][1]) > (elem2_info['page_num'], elem2_info['cell']['bbox'][1]):
            elem1_info, elem2_info = elem2_info, elem1_info

        # 如果两个表格在同一页，且中间有其他类型的元素，则不连续
        if elem1_info['page_num'] == elem2_info['page_num']:
            page_idx = elem1_info['page_num'] - 1
            y_start = elem1_info['cell']['bbox'][3]
            y_end = elem2_info['cell']['bbox'][1]
            for cell in all_pages_data[page_idx].get('cells', []):
                if cell is elem1_info['cell'] or cell is elem2_info['cell']:
                    continue
                cell_y_center = (cell['bbox'][1] + cell['bbox'][3]) / 2
                if y_start < cell_y_center < y_end:
                    # 只要中间有一个非表格元素，就不算绝对连续（页眉页脚通常在页边，不在此区域）
                    if cell['category'] != 'Table':
                        return False
        
        # 检查跨页的路径清洁度
        # 注意：这里的检查比 _is_path_clean_between 更严格，只允许页眉页脚
        return self._is_path_clean_between(
            elem1_info, elem2_info, all_pages_data, 
            allowed_categories={'Table', 'Page-header', 'Page-footer'}
        )
    # (在 DotsOCRParserOptimized 类中添加这个新函数)
    def _is_toc_entry(self, text: str, min_dots: int = 4) -> bool:
        """
        通过启发式方法判断一个文本块是否为目录（TOC）条目。
        主要检测是否存在大量的点状连接符（英文句号或中文省略号）。
        """
        if not text:
            return False
        
        # 同时计算英文句号 '.' 和中文省略号 '…' 的数量
        dot_count = text.count('.') + text.count('…')
        
        # 如果点状符号的总数超过阈值，则认为是目录条目
        return dot_count >= min_dots

    # (在 DotsOCRParserOptimized 类中)
    def _perform_cross_page_table_caption_matching(self, all_pages_data: List[dict]):
        """
        三步聚类匹配法，增加纠错与释放逻辑。
        此版本可以推翻页内匹配的错误，释放被错误占用的标题。
        """
        # ==============================================================================
        #  Phase 1: 聚类成组 (Group contiguous tables)
        # ==============================================================================
        all_tables = []
        for page_data in all_pages_data:
            for cell in page_data.get('cells', []):
                if cell['category'] == 'Table':
                    all_tables.append({'page_num': page_data['original_page_num'], 'cell': cell})
        
        all_tables.sort(key=lambda x: (x['page_num'], x['cell']['bbox'][1]))
        
        if not all_tables:
            return

        table_groups = []
        visited_table_ids = set()
        
        for i, tbl_info in enumerate(all_tables):
            if id(tbl_info['cell']) in visited_table_ids:
                continue
                
            current_group = [tbl_info]
            visited_table_ids.add(id(tbl_info['cell']))
            last_member_in_group = tbl_info
            
            for j in range(i + 1, len(all_tables)):
                next_tbl_info = all_tables[j]
                if id(next_tbl_info['cell']) in visited_table_ids:
                    continue
                
                if self._is_contiguous(last_member_in_group, next_tbl_info, all_pages_data):
                    current_group.append(next_tbl_info)
                    visited_table_ids.add(id(next_tbl_info['cell']))
                    last_member_in_group = next_tbl_info
            
            table_groups.append(current_group)

        # ==============================================================================
        #  Phase 2 & 3: 寻找归属、纠错、广播标题
        # ==============================================================================
        
        # 预先构建一个从标题文本到实际单元格对象的映射，以便快速查找和修改
        caption_text_to_cell_map = {}
        for page_data in all_pages_data:
            for cell in page_data.get('cells', []):
                if cell['category'] == 'Caption':
                    caption_text_to_cell_map[cell.get('text', '').strip()] = cell

        ALLOWED_CATEGORIES_IN_PATH = {'Table', 'Page-header', 'Page-footer'}
        SEARCH_WINDOW = 12
        
        for group in table_groups:
            group_caption_text = None
            
            # 2.1 确定组的权威标题：优先使用组内第一个元素的页内匹配结果
            first_table_in_group = group[0]
            if first_table_in_group['cell'].get('is_matched') and first_table_in_group['cell'].get('caption_text'):
                group_caption_text = first_table_in_group['cell']['caption_text']
                if self.debug_matching:
                    TQDM_BASE.write(f"[DEBUG] Group starting on page {group[0]['page_num']} inherits caption '{group_caption_text[:30]}...' from its first element.")
            
            # 2.2 【纠错与释放】检查组内其他成员是否有错误的页内匹配，并释放被它们错误占用的标题
            if group_caption_text:
                for tbl_info in group:
                    old_caption = tbl_info['cell'].get('caption_text')
                    # 如果一个表格有标题，但不是本组的权威标题，说明它错误地匹配了
                    if old_caption and old_caption != group_caption_text:
                        if self.debug_matching:
                            TQDM_BASE.write(f"[DEBUG] Correcting wrong match. Table on page {tbl_info['page_num']} had caption '{old_caption[:30]}...'.")
                        
                        # 在映射中找到被错误占用的标题单元格
                        caption_cell_to_release = caption_text_to_cell_map.get(old_caption)
                        if caption_cell_to_release:
                            # 释放它！
                            caption_cell_to_release['is_matched'] = False
                            if self.debug_matching:
                                TQDM_BASE.write(f"[DEBUG] Caption '{old_caption[:30]}...' has been RELEASED and is available again.")

            # 2.3 如果组是孤儿（即第一个元素也没有匹配到标题），则为它寻找一个孤儿标题
            if not group_caption_text:
                best_candidate = {'caption_info': None, 'score': float('inf')}
                group_start_info = group[0]
                group_end_info = group[-1]
                
                # 收集当前所有真正未使用的标题
                unmatched_captions = []
                for page_data in all_pages_data:
                    for cell in page_data.get('cells', []):
                        if (cell['category'] == 'Caption' and 
                            not cell.get('is_matched') and 
                            not cell.get('locked_for_picture')):
                            unmatched_captions.append({'page_num': page_data['original_page_num'], 'cell': cell})
                
                for cap_info in unmatched_captions:
                    page_diff_head = group_start_info['page_num'] - cap_info['page_num']
                    if 0 <= page_diff_head <= SEARCH_WINDOW:
                        if self._is_path_clean_between(cap_info, group_start_info, all_pages_data, ALLOWED_CATEGORIES_IN_PATH):
                            distance = calculate_center_distance(cap_info['cell']['bbox'], group_start_info['cell']['bbox'])
                            score = distance + page_diff_head * 500 # 跨页惩罚
                            if page_diff_head == 0 and group_start_info['cell']['bbox'][1] < cap_info['cell']['bbox'][3]: continue # 确保标题在上方
                            if score < best_candidate['score']:
                                best_candidate['score'] = score
                                best_candidate['caption_info'] = cap_info

                    page_diff_tail = cap_info['page_num'] - group_end_info['page_num']
                    if 0 <= page_diff_tail <= SEARCH_WINDOW:
                        if self._is_path_clean_between(group_end_info, cap_info, all_pages_data, ALLOWED_CATEGORIES_IN_PATH):
                            distance = calculate_center_distance(group_end_info['cell']['bbox'], cap_info['cell']['bbox'])
                            score = distance + page_diff_tail * 500 # 跨页惩罚
                            if page_diff_tail == 0 and cap_info['cell']['bbox'][1] < group_end_info['cell']['bbox'][3]: continue # 确保标题在下方
                            if score < best_candidate['score']:
                                best_candidate['score'] = score
                                best_candidate['caption_info'] = cap_info
                
                if best_candidate['caption_info']:
                    source_caption_info = best_candidate['caption_info']
                    group_caption_text = source_caption_info['cell'].get('text', '').strip()
                    # 标记新找到的标题为已使用
                    source_caption_info['cell']['is_matched'] = True
                    if self.debug_matching:
                         TQDM_BASE.write(f"[DEBUG] Orphan group matched with caption on pg {source_caption_info['page_num']}: '{group_caption_text[:30]}...'")

            # 3. 如果找到了权威标题（无论是继承的还是新找的），就广播给组内所有成员
            if group_caption_text:
                # 确保权威标题本身被标记为已匹配
                authoritative_caption_cell = caption_text_to_cell_map.get(group_caption_text)
                if authoritative_caption_cell:
                    authoritative_caption_cell['is_matched'] = True

                for tbl_info in group:
                    tbl_info['cell']['is_matched'] = True
                    tbl_info['cell']['caption_text'] = group_caption_text

    # (在 DotsOCRParserOptimized 类中)
    async def _process_single_page_optimized_streaming(self, page_data: dict):
        """
        处理单个页面，融合了老版本的健壮重试逻辑与新版本的并发重试和精细降级策略。
        """
        origin_image = page_data['origin_image']
        processed_image = page_data['processed_image']
        prompt = self.get_prompt(page_data['prompt_mode'], bbox=page_data.get('bbox'), origin_image=origin_image, image=processed_image)
        original_page_num = page_data['original_page_num']
        page_idx = page_data['page_idx']
        save_dir = page_data.get('save_dir')

        last_error = None
        last_raw_response_for_fallback = None

        # ==============================================================================
        #  STAGE 1: 尝试获取并处理一个有效的响应
        # ==============================================================================

        # --- 1.1 主尝试 ---
        try:
            response = await self._inference_with_vllm(processed_image, prompt)
            last_raw_response_for_fallback = response
            if not response:
                raise ValueError("Inference call returned an empty response.")
            
            loop = asyncio.get_running_loop()
            post_process_func = functools.partial(post_process_output, response, page_data['prompt_mode'], origin_image, processed_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            cells, _ = await loop.run_in_executor(None, post_process_func)

            if not (cells and isinstance(cells, list) and all(isinstance(c, dict) and 'bbox' in c for c in cells)):
                raise ValueError("Post-processing yielded invalid or empty cells data.")

            # 如果主尝试完全成功，直接生成MD并返回
            layout_func = functools.partial(layoutjson2md_full_robust, origin_image, cells, 'text', True)
            md_content_no_hf = await loop.run_in_executor(None, layout_func)
            
            save_name_base = f"{Path(page_data['filename']).stem}_page_{original_page_num}"
            json_path, layout_path = await self._save_intermediate_outputs_async(
                save_dir, save_name_base, response, cells, origin_image
            )
            
            return {
                'page_no': page_idx, 'original_page_num': original_page_num, 'status': 'success',
                'md_content': md_content_no_hf, 'cells': cells, 'origin_image': origin_image,
                'page_json_path': json_path, 'page_layout_path': layout_path
            }

        except Exception as e:
            last_error = e
            TQDM_BASE.write(f"⚠️ Page {original_page_num} initial attempt failed: {type(e).__name__}. Triggering retries...")

        # --- 1.2 并发竞速重试 (仅当主尝试失败时执行) ---
        if self.concurrent_retries > 0:
            try:
                self.monitor.record_retry(self.concurrent_retries) # 记录N次重试
                TQDM_BASE.write(f"  Racing {self.concurrent_retries} concurrent requests for page {original_page_num}...")
                
                # 调用并发竞速函数
                response = await self._race_inference_attempts(processed_image, prompt, num_attempts=self.concurrent_retries)
                
                last_raw_response_for_fallback = response
                if not response:
                    raise ValueError("All concurrent inference calls returned an empty response.")

                loop = asyncio.get_running_loop()
                post_process_func = functools.partial(post_process_output, response, page_data['prompt_mode'], origin_image, processed_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                cells, _ = await loop.run_in_executor(None, post_process_func)

                if not (cells and isinstance(cells, list) and all(isinstance(c, dict) and 'bbox' in c for c in cells)):
                    raise ValueError("Post-processing yielded invalid or empty cells data from raced response.")

                # 如果并发重试成功
                TQDM_BASE.write(f"  ✅ Concurrent retry succeeded for page {original_page_num}.")
                layout_func = functools.partial(layoutjson2md_full_robust, origin_image, cells, 'text', True)
                md_content_no_hf = await loop.run_in_executor(None, layout_func)
                
                save_name_base = f"{Path(page_data['filename']).stem}_page_{original_page_num}"
                json_path, layout_path = await self._save_intermediate_outputs_async(
                    save_dir, save_name_base, response, cells, origin_image
                )
                
                return {
                    'page_no': page_idx, 'original_page_num': original_page_num, 'status': 'success',
                    'md_content': md_content_no_hf, 'cells': cells, 'origin_image': origin_image,
                    'page_json_path': json_path, 'page_layout_path': layout_path
                }
            
            except Exception as e:
                last_error = e
                TQDM_BASE.write(f"  ❌ All {self.concurrent_retries} concurrent attempts failed for page {original_page_num}. Final error: {type(e).__name__}")


        # ==============================================================================
        #  STAGE 2: 降级策略 (如果所有尝试都失败了)
        # ==============================================================================
        TQDM_BASE.write(f"Page {original_page_num} failed all attempts. Last known error: {type(last_error).__name__}. Attempting fallbacks.")

        # --- 降级1: 简单文本提取 ---
        if last_raw_response_for_fallback:
            try:
                loop = asyncio.get_running_loop()
                post_process_func = functools.partial(post_process_output, last_raw_response_for_fallback, page_data['prompt_mode'], origin_image, processed_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                cells_for_text, _ = await loop.run_in_executor(None, post_process_func)
                if cells_for_text and isinstance(cells_for_text, list):
                    md_content_simple = layoutjson2md_simple_extract(cells_for_text, 'text', True)
                    if md_content_simple and md_content_simple.strip():
                        TQDM_BASE.write(f"  ✅ Text extraction fallback succeeded for page {original_page_num}.")
                        return {
                            'page_no': page_idx, 'original_page_num': original_page_num, 
                            'status': 'success_fallback_text', 'md_content': md_content_simple, 
                            'cells': [], 'origin_image': origin_image,
                            'page_json_path': None, 'page_layout_path': None
                        }
            except Exception as text_fallback_e:
                TQDM_BASE.write(f"  ⚠️ Text extraction fallback also failed for page {original_page_num}. Reason: {text_fallback_e}")

        # --- 降级2: 截图保底 ---
        TQDM_BASE.write(f"Page {original_page_num} failed text extraction. Saving error screenshot as final placeholder.")
        try:
            images_dir = os.path.join(save_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            bad_image_filename = f"page_{original_page_num}_bad.jpg"
            bad_image_path = os.path.join(images_dir, bad_image_filename)
            if origin_image.mode != 'RGB':
                origin_image = origin_image.convert('RGB')
            origin_image.save(bad_image_path, "JPEG", quality=95)
            
            if self.badcase_collection_dir:
                try:
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    date_specific_badcase_dir = os.path.join(self.badcase_collection_dir, today_str)
                    os.makedirs(date_specific_badcase_dir, exist_ok=True)
                    badcase_filename_base = Path(page_data.get('filename', 'unknown_file')).stem
                    badcase_file_name = f"{badcase_filename_base}_page_{original_page_num}.jpg"
                    badcase_target_path = os.path.join(date_specific_badcase_dir, badcase_file_name)
                    shutil.copyfile(bad_image_path, badcase_target_path)
                    TQDM_BASE.write(f"  Copied badcase screenshot for page {original_page_num} to {badcase_target_path}")
                except Exception as copy_e:
                    TQDM_BASE.write(f"  WARNING: Failed to copy badcase screenshot. Reason: {copy_e}")
            
            final_md = f"![{bad_image_filename}](images/{bad_image_filename})"
            TQDM_BASE.write(f"  Fallback succeeded. Screenshot for page {original_page_num} saved.")
            return {
                'page_no': page_idx, 'original_page_num': original_page_num, 
                'status': 'success_fallback_image', 'md_content': final_md, 
                'cells': [], 'origin_image': origin_image,
                'page_json_path': None, 'page_layout_path': None
            }
        except Exception as e:
            TQDM_BASE.write(f"  CRITICAL: Image fallback failed for page {original_page_num}. Reason: {e}")
            last_error = f"{str(last_error)}\nAdditionally, failed to save fallback image: {e}"
            return {
                'page_no': page_idx, 'original_page_num': original_page_num, 
                'status': 'error', 'error': str(last_error),
                'page_json_path': None, 'page_layout_path': None
            }
    # (在 DotsOCRParserOptimized 类中添加这三个新函数)
    def _is_sentence_end(self, text: str) -> bool:
        """检查文本是否以句子结束符结尾，会忽略末尾的空白和右引号/括号。"""
        if not text:
            return False
        
        # 定义句子结束符和需要忽略的结尾标点
        SENTENCE_END_CHARS = {'。', '？', '！', '.', '?', '!', ';', '；'}
        CLOSING_PUNCTUATION = {'"', "'", '”', '’', '』', ')', '）', ']'}

        stripped_text = text.rstrip()
        # 循环剔除末尾的右侧标点
        while stripped_text and stripped_text[-1] in CLOSING_PUNCTUATION:
            stripped_text = stripped_text[:-1].rstrip()
        
        if not stripped_text:
            return True # 如果只剩下标点，也认为是一个结束

        return stripped_text[-1] in SENTENCE_END_CHARS

    def _starts_new_paragraph(self, text: str) -> bool:
        """检查文本是否以新段落的典型特征开始（如序号、标题、缩进）。"""
        if not text:
            return False
        
        # 使用预编译的类属性
        return bool(self.NEW_PARAGRAPH_PATTERN.match(text))

    # (在 DotsOCRParserOptimized 类中)
    def _perform_cross_page_text_merging(self, all_pages_layout_data: List[dict]):
        """
        采用后置判断逻辑，精确区分装饰性与结构性 Section-header。
        """
        for i in range(len(all_pages_layout_data) - 1):
            current_page_data = all_pages_layout_data[i]
            next_page_data = all_pages_layout_data[i+1]
            
            if not current_page_data.get('cells') or not next_page_data.get('cells'):
                continue

            # 步骤 1: 寻找当前页的最后一个 'Text' 块
            last_text_block_curr_page = None
            for cell in reversed(current_page_data['cells']):
                if cell.get('category') == 'Text':
                    last_text_block_curr_page = cell
                    break
            if not last_text_block_curr_page:
                continue

            # 步骤 2: 寻找下一页的第一个 'Text' 块
            first_text_block_next_page = None
            first_text_block_index = -1
            for idx, cell in enumerate(next_page_data['cells']):
                if cell.get('category') == 'Text':
                    first_text_block_next_page = cell
                    first_text_block_index = idx
                    break
            if not first_text_block_next_page:
                continue

            # 步骤 3: 验证路径
            if self._is_path_clean_between_texts(last_text_block_curr_page, first_text_block_next_page, current_page_data, next_page_data):
                text1 = last_text_block_curr_page.get('text', '')
                text2 = first_text_block_next_page.get('text', '')
                
                # 步骤 4.1: 检查句子是否已经结束
                if self._is_sentence_end(text1) or self._is_toc_entry(text1):
                    continue

                # 步骤 4.2: 检查下一页的文本块是否由一个“结构性”标题引导
                # 我们检查紧邻第一个文本块之前的那个元素。
                if first_text_block_index > 0:
                    element_before = next_page_data['cells'][first_text_block_index - 1]
                    # 如果文本块前面是一个 Section-header
                    # （我们通过检查它是否以序号或特定模式开头来判断）
                    if element_before.get('category') == 'Section-header':
                        header_text = element_before.get('text', '').strip()
                        # 如果这个 Section-header 看起来像一个真正的标题 (如 "一、", "第2章")
                        # 我们就认为它开启了新段落，不应该合并。
                        # self.NEW_PARAGRAPH_PATTERN 已经包含了对 "一、", "1." 等的匹配
                        if self._starts_new_paragraph(header_text):
                            if self.debug_matching:
                                TQDM_BASE.write(f"[DEBUG] Merge blocked by a structural Section-header: '{header_text}'")
                            continue

                # 步骤 4.3: 检查文本块本身是否像新段落的开始
                if self._starts_new_paragraph(text2) or self._is_toc_entry(text2):
                    continue
                
                # 步骤 5: 执行合并
                merge_type = "Standard"
                stripped_text1 = text1.rstrip()
                if (stripped_text1.endswith('-') and len(stripped_text1) > 1 and self.english_letter_pattern.match(text2.lstrip())):
                    last_text_block_curr_page['text'] = stripped_text1[:-1] + text2.lstrip()
                    merge_type = "Hyphenated"
                else:
                    last_text_block_curr_page['text'] += text2
                first_text_block_next_page['to_be_deleted'] = True
                
                if self.debug_matching:
                    TQDM_BASE.write(f"[DEBUG] Merged text ({merge_type}) from page {next_page_data['original_page_num']} into page {current_page_data['original_page_num']}.")
        
        # 步骤 6: 清理
        for page_data in all_pages_layout_data:
            if page_data.get('cells'):
                page_data['cells'] = [cell for cell in page_data['cells'] if not cell.get('to_be_deleted')]

    # (在 DotsOCRParserOptimized 类中)
    def _is_path_clean_between_texts(self, last_text_cell_page1, first_text_cell_page2, page1_data, page2_data) -> bool:
        """
        允许 Section-header 通过路径检查，将判断逻辑后置。
        因为部分 Page-header 会被转换为 Section-header，不能在此阶段一刀切地阻断。
        """
        # 允许列表中临时加入 Section-header，真正的判断将在合并函数中进行。
        # Title 仍然是硬性障碍。
        ALLOWED_CATEGORIES_IN_PATH = {'Picture', 'Page-header', 'Page-footer', 'Section-header'}

        # 1. 检查第一页的剩余部分
        found_last_text = False
        for cell in page1_data.get('cells', []):
            if cell is last_text_cell_page1:
                found_last_text = True
                continue
            if found_last_text:
                if cell['category'] not in ALLOWED_CATEGORIES_IN_PATH:
                    if self.debug_matching:
                        TQDM_BASE.write(f"[DEBUG] Path blocked on page {page1_data['original_page_num']} by a hard obstacle: '{cell['category']}'.")
                    return False

        # 2. 检查第二页的起始部分
        for cell in page2_data.get('cells', []):
            if cell is first_text_cell_page2:
                break
            if cell['category'] not in ALLOWED_CATEGORIES_IN_PATH:
                if self.debug_matching:
                    TQDM_BASE.write(f"[DEBUG] Path blocked on page {page2_data['original_page_num']} by a hard obstacle: '{cell['category']}'.")
                return False

        return True

    # (在 DotsOCRParserOptimized 类中)
    async def parse_pdf(self, input_path, filename, prompt_mode, save_dir, page_progress_callback=None, bbox=None, skip_blank_pages=False):
        # ==================================================================
        #  PASS 1: Concurrent page processing to get raw layouts
        # ==================================================================
        page_queue = asyncio.Queue(maxsize=self.queue_size)
        results_storage = {}; total_pages_expected = 0
        
        # (在 DotsOCRParserOptimized.parse_pdf 内部)
        async def page_producer(num_consumers):
            nonlocal total_pages_expected
            loop = asyncio.get_running_loop()

            try:
                # 仅用于获取页数，然后立即关闭，或在循环外打开
                # 为了安全，我们只在这里获取页数
                try:
                    with fitz.open(input_path) as doc:
                        total_pages_expected = doc.page_count
                except Exception as doc_open_error:
                    TQDM_BASE.write(f"FATAL: Could not open or read PDF file {filename}: {doc_open_error}")
                    total_pages_expected = 0

                if total_pages_expected == 0:
                    TQDM_BASE.write(f"Warning: File {filename} has 0 pages or could not be opened.")
                    # 确保即使文件有问题，消费者也能正常退出
                    for _ in range(num_consumers):
                        await page_queue.put(None)
                    return

                # 主循环：快速分发任务，不做任何耗时操作
                for page_idx in range(total_pages_expected):
                    # 准备传递给工作进程的纯数据参数
                    task_args = (
                        input_path, # 传递文件路径
                        page_idx,   # 传递页码
                        self.dpi,   # 传递DPI
                        skip_blank_pages, 
                        self.blank_white_threshold, 
                        self.blank_noise_threshold, 
                        self.min_pixels, 
                        self.max_pixels
                    )

                    # 提交任务到进程池，得到一个 future
                    process_future = loop.run_in_executor(
                        self.process_pool, 
                        _process_pdf_page_worker, 
                        task_args
                    )

                    # 将 future 和其他元数据打包，立刻放入队列
                    # 消费者将负责 await 这个 future
                    page_task_info = {
                        'future': process_future, # 关键：传递future对象
                        'page_idx': page_idx, 
                        'original_page_num': page_idx + 1, 
                        'prompt_mode': prompt_mode, 
                        'bbox': bbox, 
                        'filename': filename, 
                        'save_dir': save_dir
                    }
                    # 这个 put 是异步的，但因为队列通常有空间，所以会立刻返回
                    await page_queue.put(page_task_info)

            except Exception as e:
                import traceback
                TQDM_BASE.write(f"FATAL error in page producer for file '{filename}': {e}\n{traceback.format_exc()}")
            finally:
                # 任务分发完毕，发送停止信号
                TQDM_BASE.write(f"Producer for '{filename}' finished dispatching all {total_pages_expected} page tasks.")
                for _ in range(num_consumers):
                    await page_queue.put(None) 
        
        # (在 DotsOCRParserOptimized.parse_pdf 内部)
        async def page_consumer(consumer_id):
            while True:
                # 从队列中获取任务包
                page_task_info = await page_queue.get()
                if page_task_info is None: 
                    break
                
                page_data = None  # 初始化为None
                try:
                    # 1. 等待后台进程完成图片预处理
                    # page_task_info['future'] 是 run_in_executor 返回的对象
                    pre_processing_result = await page_task_info['future']

                    # 2. 检查预处理结果
                    if pre_processing_result['status'] != 'success':
                        # 如果是空白页或错误，直接存入结果并更新进度条
                        status = pre_processing_result.get('status', 'processing_error')
                        error = pre_processing_result.get('error', 'Unknown pre-processing error')
                        page_num = page_task_info['original_page_num']
                        
                        results_storage[page_num] = {
                            'page_no': page_task_info['page_idx'], 
                            'original_page_num': page_num, 
                            'status': status,
                            'error': error if status != 'skipped_blank' else None
                        }
                        if page_progress_callback: 
                            page_progress_callback()
                        continue # 处理下一个任务

                    # 3. 准备调用API的数据
                    # 将预处理结果和元数据合并
                    page_data = {**page_task_info, **pre_processing_result}
                    
                    # 4. 调用API（此部分逻辑不变）
                    async with self.page_semaphore:
                        result = await self._process_single_page_optimized_streaming(page_data)
                    results_storage[page_data['original_page_num']] = result

                    if page_progress_callback: 
                        page_progress_callback()

                except Exception as e:
                    # page_data 可能在 await future 之后才被赋值，所以要用 page_task_info
                    page_num = page_task_info.get('original_page_num', 'N/A')
                    TQDM_BASE.write(f"Critical error in consumer {consumer_id} for page {page_num} of {filename}: {e}")
                    
                    # 记录失败结果
                    results_storage[page_num] = {
                        'page_no': page_task_info.get('page_idx', -1),
                        'original_page_num': page_num,
                        'status': 'error',
                        'error': str(e)
                    }
                    if page_progress_callback: 
                        page_progress_callback()

        num_consumers = self.page_concurrency
        producer_task = asyncio.create_task(page_producer(num_consumers))
        consumer_tasks = [asyncio.create_task(page_consumer(i)) for i in range(num_consumers)]
        await asyncio.gather(producer_task, *consumer_tasks)
        
        all_pages_layout_data = []
        for i in range(total_pages_expected):
            page_num = i + 1
            if page_num in results_storage:
                all_pages_layout_data.append(results_storage[page_num])
            else:
                all_pages_layout_data.append({'page_no': i, 'original_page_num': page_num, 'status': 'missing_result', 'error': 'Page data was lost or failed before storage.'})
                
        all_results_for_jsonl = []
        skipped_count = 0; hard_failed_pages_count = 0

        for result in all_pages_layout_data:
            status = result.get('status', 'unknown_error')
            page_num = result.get('original_page_num')
            
            if 'cells' in result and result.get('cells'):
                result['cells'] = self._perform_intra_page_matching(result['cells'])
            
            if status == 'skipped_blank': 
                skipped_count += 1
            # 使用 self.SUCCESS_STATUSES 进行检查，更准确地统计失败页面
            elif status not in self.SUCCESS_STATUSES: 
                hard_failed_pages_count += 1
            
            all_results_for_jsonl.append({'page_no': page_num, 'file_path': input_path, 'status': status, 'filename': filename, 'error': result.get('error') if status not in self.SUCCESS_STATUSES else None})
        
        self._perform_cross_page_table_caption_matching(all_pages_layout_data)
        self._perform_cross_page_text_merging(all_pages_layout_data)

        if hard_failed_pages_count > 0: TQDM_BASE.write(f"⚠️ File '{filename}' was processed with {hard_failed_pages_count} hard-failed page(s). The output MD will contain placeholders for them.")

        final_md_pages = []
        for result in all_pages_layout_data:
            status = result.get('status')
            
            # 使用 self.SUCCESS_STATUSES 进行检查，并区分不同成功状态的处理方式
            # 只有主成功状态 'success' 才需要从 cells 重新生成MD
            # 降级成功状态直接使用它们已经生成的MD内容
            if status == 'success':
                # --- 根据开关过滤 cells 并重新生成 Markdown ---
                original_cells = result.get('cells', [])
                cells_to_render = original_cells

                # 根据开关过滤页眉
                if not self.keep_page_header:
                    cells_to_render = [cell for cell in cells_to_render if cell.get('category') != 'Page-header']
                
                # 根据开关过滤页脚
                if not self.keep_page_footer:
                    cells_to_render = [cell for cell in cells_to_render if cell.get('category') != 'Page-footer']

                current_md = layoutjson2md_full_robust(
                    result['origin_image'], 
                    cells_to_render,          # 使用我们自己过滤后的列表
                    'text', 
                    no_page_hf=False          # 禁用函数内置的过滤功能
                )
                final_md_pages.append({
                'md': current_md, 
                'cells': original_cells, # 传递原始 cells 用于图片命名
                'original_page_num': result['original_page_num'],
                'origin_image': result.get('origin_image')
                })
            elif status in ['success_fallback_text', 'success_fallback_image']:
                final_md_pages.append({
                    'md': result.get('md_content', ''),
                    'cells': [],
                    'original_page_num': result['original_page_num'],
                    'origin_image': result.get('origin_image')
                })
            elif status not in ['skipped_blank']:
                page_num = result.get('original_page_num', 'N/A')
                error_md_placeholder = f"\n\n[Page {page_num} could not be rendered or processed]\n\n"
                final_md_pages.append({
                    'md': error_md_placeholder, 
                    'cells': [], 
                    'original_page_num': page_num,
                    'origin_image': None
                })

        if not final_md_pages:
            if skipped_count > 0 and skipped_count == total_pages_expected:
                TQDM_BASE.write(f"INFO: File '{filename}' consists entirely of {skipped_count} blank page(s). No .md file will be generated.")
                return all_results_for_jsonl
            else:
                TQDM_BASE.write(f"ERROR: No content was extracted from '{filename}'. Aborting file creation.")
                return [{"error": "No content could be generated for this file", "file_path": input_path}]

        images_dir = os.path.join(save_dir, "images")
        loop = asyncio.get_running_loop()
        tasks = []
        for page_data in final_md_pages:
            origin_image = page_data.get('origin_image')
            task = loop.run_in_executor(None, functools.partial(self._process_base64_images_with_custom_naming, page_data['md'], images_dir, page_data['original_page_num'], page_data['cells'], origin_image))
            tasks.append(task)

        final_md_contents = await asyncio.gather(*tasks)
        if self.add_page_tag:
            content_per_page_dict = {}
            for final_md_content in final_md_contents:
                num = final_md_content["page_num"] - 1
                if num not in content_per_page_dict:
                    content_per_page_dict[num] = final_md_content["md_content"]
                else:
                    content_per_page_dict[num] += "\n\n" + final_md_content["md_content"]
            sorted_content_per_page_dict = {k: content_per_page_dict[k] for k in sorted(content_per_page_dict.keys())}
            combined_md_content = ""
            for num, md_content in sorted_content_per_page_dict.items():
                combined_md_content += md_content + "\n" + f"<special_page_num_tag>{num}</special_page_num_tag>" + "\n"
        else:
            final_md_contents = [final_md_content["md_content"] for final_md_content in final_md_contents]
            combined_md_content = "\n\n".join(final_md_contents)
        clean_md_content = re.sub(r'(\n\s*){3,}', '\n\n', combined_md_content)

        # combined_md_path = os.path.join(save_dir, f"{filename}.md")
        combined_md_path = self._get_unique_md_path(save_dir, filename) # 新命名
        with open(combined_md_path, "w", encoding="utf-8") as md_file: md_file.write(clean_md_content)

        for res in all_results_for_jsonl:
            if res['status'] in self.SUCCESS_STATUSES: res['output_md_path'] = combined_md_path; break
        if all_results_for_jsonl: all_results_for_jsonl[0]['skipped_blank_pages'] = skipped_count
        return all_results_for_jsonl
    # (在 DotsOCRParserOptimized 类中)
    async def parse_file(self, input_path: str, output_dir: str = "", prompt_mode: str = "prompt_layout_all_en", page_progress_callback=None, bbox=None, skip_blank_pages=False, rename_to: Optional[str] = None):
        output_dir = output_dir or self.output_dir; output_dir = os.path.abspath(output_dir)

        # --- 根据 rename_to 参数决定文件名 ---
        if rename_to:
            # 如果提供了重命名参数，则使用它作为基础文件名
            # 移除了所有可能的文件扩展名，以确保一个干净的基础名称
            filename = os.path.splitext(os.path.basename(rename_to))[0]
        else:
            # 否则，使用原始输入文件的名称
            filename, _ = os.path.splitext(os.path.basename(input_path))

        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        # 断点续传检查 (这里的逻辑现在是正确的，因为它基于最终的文件名)
        if self.enable_resume and not self.force_reprocess:
            is_processed, md_path = is_file_already_processed(input_path, output_dir, filename)
            if is_processed:
                TQDM_BASE.write(f"Skipping already processed file: {input_path} (MD file exists: {md_path})")
                # 返回模拟的成功结果，保持与实际处理一致的结构
                return [{
                    'page_no': 0,
                    'original_page_num': 1,
                    'file_path': input_path,
                    'output_md_path': md_path,
                    'status': 'success',
                    'error': None,
                    'filename': filename,
                    'skipped': True  # 标记为跳过的文件
                }]
        
        try:
            # 从这里开始，函数后面的逻辑都使用我们新确定的 `filename`
            file_ext = os.path.splitext(input_path)[1]
            if file_ext.lower() == '.pdf':
                results = await self.parse_pdf(input_path, filename, prompt_mode, save_dir, page_progress_callback, bbox=bbox, skip_blank_pages=skip_blank_pages)
            elif file_ext.lower() in image_extensions:
                results = await self.parse_image_simple(input_path, filename, prompt_mode, save_dir, bbox=bbox)
            else: raise ValueError(f"File extension {file_ext} not supported.")
            return results
        except Exception as e:
            TQDM_BASE.write(f"Critical error parsing file {input_path}: {e}")
            return [{"error": str(e), "file_path": input_path}]

    # (在 DotsOCRParserOptimized 类中)
    async def parse_image_simple(self, input_path, filename, prompt_mode, save_dir,  bbox=None):
        """
        重构后的图片解析方法，复用PDF的单页处理逻辑以获得健壮性。
        """
        try:
            origin_image = fetch_image(input_path)
            processed_image = fetch_image(origin_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            
            page_data = {
                'page_idx': 0,
                'original_page_num': 1,
                'origin_image': origin_image, 
                'processed_image': processed_image, 
                'prompt_mode': prompt_mode,
                'bbox': bbox,
                'filename': filename,
                'save_dir': save_dir
            }

            result = await self._process_single_page_optimized_streaming(page_data)

            status = result.get('status')
            if status in self.SUCCESS_STATUSES:
                # --- 根据开关过滤 cells 并重新生成 Markdown ---
                original_cells = result.get('cells', [])
                if original_cells:
                    original_cells = self._perform_intra_page_matching(original_cells)

                cells_to_render = original_cells
                if not self.keep_page_header:
                    cells_to_render = [cell for cell in cells_to_render if cell.get('category') != 'Page-header']
                if not self.keep_page_footer:
                    cells_to_render = [cell for cell in cells_to_render if cell.get('category') != 'Page-footer']
                
                # 使用过滤后的 cells 重新生成 Markdown 内容
                # 仅在完全成功时才需要从cells重新生成MD
                if status == 'success':
                    md_content = layoutjson2md_full_robust(
                        origin_image,
                        cells_to_render,
                        'text',
                        no_page_hf=False
                    )
                
                images_dir = os.path.join(save_dir, "images")
                loop = asyncio.get_running_loop()
                # 使用新生成的md_content，但传递原始cells列表用于图片-标题匹配
                image_processing_func = functools.partial(self._process_base64_images_with_custom_naming, md_content, images_dir, 1, original_cells, origin_image)
                final_md_content = await loop.run_in_executor(None, image_processing_func)
                
                # md_file_path = os.path.join(save_dir, f"{filename}.md")
                md_file_path = self._get_unique_md_path(save_dir, filename) # 新命名方式
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(final_md_content["md_content"])

                result['file_path'] = input_path
                result['output_md_path'] = md_file_path
                return [result]
            else:
                result['file_path'] = input_path
                return [result]

        except Exception as e:
            TQDM_BASE.write(f"Critical error during image preparation for {input_path}: {e}")
            return [{ "error": str(e), "file_path": input_path, 'status': 'error', 'filename': filename, 'page_no': 0, 'original_page_num': 1 }]
        
    # (在 DotsOCRParserOptimized 类中)
    def _get_unique_md_path(self, save_dir: str, base_filename: str) -> str:
        """
        根据基础文件名生成一个唯一的MD文件路径，如果已存在则添加后缀_2, _3等。
        """
        # 移除可能存在的文件扩展名，以防传入带.md的名称
        base_filename = os.path.splitext(base_filename)[0]
        
        md_path = os.path.join(save_dir, f"{base_filename}.md")
        
        # 如果文件不存在，直接返回路径
        if not os.path.exists(md_path):
            return md_path
        
        # 如果文件已存在，则开始尝试添加后缀
        counter = 2
        while True:
            new_filename = f"{base_filename}_{counter}"
            new_md_path = os.path.join(save_dir, f"{new_filename}.md")
            if not os.path.exists(new_md_path):
                TQDM_BASE.write(f"Warning: Output file for '{base_filename}' already exists. Saving as '{os.path.basename(new_md_path)}' instead.")
                return new_md_path
            counter += 1
        
# ==============================================================================
#  主执行函数 
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Dots.OCR Parser (Performance Optimized & Reviewed Version) - V4 with Table Screenshots")
    
    # --- 功能性参数 ---
    # 使用互斥组确保 --input-dir 和 --input-file 只能使用一个
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=str, help="Input directory containing files to process.")
    input_group.add_argument("--input-file", type=str, help="Path to a single PDF or image file to process.")
    
    parser.add_argument("--output-dir", type=str, default="./output", help="Output root directory")
    parser.add_argument("--rename", type=str, default=None, help="Rename the output file to this value (without extension). If used with --input-dir, all files will be renamed based on this, which is usually not desired.")
    parser.add_argument("--prompt", choices=list(dict_promptmode_to_prompt.keys()), type=str, default="prompt_layout_all_en")
    parser.add_argument('--bbox', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'))
    parser.add_argument("--skip_blank_pages", action='store_true', help="Skip blank pages in PDF processing")
    parser.add_argument("--enable_table_screenshot", action='store_true', help="Enable saving screenshots of detected tables. Disabled by default.")
    parser.add_argument("--skip_uncaptioned_images", action='store_true', help="If enabled, images without a matched caption will not be saved or included in the final Markdown file.")
    parser.add_argument("--keep_page_header", action='store_true', help="Keep page headers in the final Markdown output. (Default: remove)")
    parser.add_argument("--keep_page_footer", action='store_true', help="Keep page footers in the final Markdown output. (Default: remove)")
    parser.add_argument("--concurrent_retries", type=int, default=4, help="Number of concurrent retry attempts for a failed page. Set to 0 to disable and use serial retries. Default: 4")
    parser.add_argument("--save_page_json", action='store_true', help="Save the raw model JSON response for each page.")
    parser.add_argument("--save_page_layout", action='store_true', help="Save an image of each page with layout boxes drawn on it.")
    # --- 连接与模型参数 ---
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=900.0, help="Timeout for API requests in seconds")
    # --- 性能与并发调优参数 ---
    parser.add_argument("--file_concurrency", type=int, default=24, help="Number of files to process concurrently.")
    parser.add_argument("--page_concurrency", type=int, default=16, help="Global limit for concurrent page API requests across all files.")
    parser.add_argument("--num_cpu_workers", type=int, default=0, help="Number of CPU worker processes for page rendering. (0 for auto-detect based on CPU cores)")
    parser.add_argument("--queue_size", type=int, default=800, help="Max size of the in-memory page queue between CPU processing and API calls.")
    # --- 图像处理与重试参数 ---
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=7, help="Maximum number of retry attempts for a failed API call")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Initial retry delay in seconds")
    parser.add_argument("--blank_white_threshold", type=float, default=0.98)
    parser.add_argument("--blank_noise_threshold", type=float, default=0.002)
    # --- 调试与其它 ---
    parser.add_argument("--no_warmup", action='store_true', help="Disable GPU warmup")
    parser.add_argument("--debug_matching", action='store_true', help="Enable detailed caption matching debug output")
    parser.add_argument("--api_key", type=str, help="API key for authentication")
    parser.add_argument("--badcase_collection_dir", type=str, default="/mnt/dotsocr_badcase", help="Directory to save failed page screenshots (badcases).")
    parser.add_argument("--add_page_tag", type=bool, help="Whether add <special_page_num_tag> for following QA pipeline")

    # --- 断点续传参数 ---
    parser.add_argument("--disable_resume", action='store_true', help="Disable resume functionality (process all files)")
    parser.add_argument("--force_reprocess", action='store_true', help="Force reprocessing of all files, ignoring existing MD files")
    parser.add_argument("--show_skipped", action='store_true', default=True, help="Show information about skipped files in summary (default: enabled)")
    
    args = parser.parse_args()
    parser_args = vars(args)
    parser_args['enable_warmup'] = not args.no_warmup
    # parser_args['enable_resume'] = not args.disable_resume
    parser_args['save_page_json'] = args.save_page_json
    parser_args['save_page_layout'] = args.save_page_layout

    # 如果使用了 --rename，则强制禁用断点续传功能并发出警告
    if args.rename and not args.disable_resume:
        print("Warning: --rename is used, which is incompatible with the resume feature. Disabling resume functionality for this run.")
        parser_args['enable_resume'] = False
        # 强制重新处理，以防之前有不完整的重命名文件
        parser_args['force_reprocess'] = True 
    else:
        parser_args['enable_resume'] = not args.disable_resume
    
    # 将 parser 和进度条变量在 try 块外部声明，以便 finally 块可以访问
    dots_ocr_parser = None
    file_progress = None
    page_progress = None
    
    # 统计变量
    successful_files = 0
    perfectly_successful_files = 0
    processed_pages = 0
    skipped_blank_pages = 0
    failed_pages = []
    skipped_files = 0
    all_files_to_process = []
    total_pages_all_files = 0

    try:
        dots_ocr_parser = DotsOCRParserOptimized(**parser_args)

        # --- Badcase 目录创建逻辑 ---
        if args.badcase_collection_dir:
            badcase_dir_path = Path(args.badcase_collection_dir).resolve()
            badcase_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Badcase collection directory set to: {badcase_dir_path}")
        
        # --- 根据输入参数构建文件列表 ---
        output_root = Path(args.output_dir).resolve()
        
        if args.input_file:
            # 处理单个文件输入
            input_path = Path(args.input_file).resolve()
            if not input_path.is_file():
                print(f"Error: Provided input file does not exist or is not a file: {args.input_file}")
                return
            all_files_to_process.append(input_path)
            # 对于单文件，其父目录是输入根目录，用于计算相对路径
            input_root = input_path.parent
        else:
            # 处理目录输入
            input_root = Path(args.input_dir).resolve()
            for ext in ['*.pdf'] + [f'*.{e.lstrip(".")}' for e in image_extensions]:
                all_files_to_process.extend(input_root.rglob(ext))

        if not all_files_to_process:
            if args.input_dir:
                 print(f"No supported files found in {input_root}")
            return
        
        # 断点续传：过滤掉已处理的文件
        files_to_skip = set()
        if dots_ocr_parser.enable_resume and not args.force_reprocess:
            processed_files = get_processed_files(str(input_root), args.output_dir)
            files_to_skip = processed_files
            skipped_files = len(files_to_skip)
            
            if processed_files:
                print(f"Found {len(processed_files)} already processed files, will skip them.")
                if args.show_skipped:
                    print("Skipped files:")
                    for i, file_path in enumerate(sorted(processed_files)[:10]):  # 只显示前10个
                        print(f"  {i+1}. {file_path}")
                    if len(processed_files) > 10:
                        print(f"  ... and {len(processed_files) - 10} more")
        
        if not args.force_reprocess:
            all_files_to_process = [f for f in all_files_to_process if str(f) not in files_to_skip]
        
        if not all_files_to_process:
            print("All files have been processed already. Use --force_reprocess to reprocess all files.")
            return

        for file_path in all_files_to_process:
            if file_path.suffix.lower() == '.pdf':
                try:
                    with fitz.open(str(file_path)) as doc: total_pages_all_files += doc.page_count
                except Exception: total_pages_all_files += 1
            else: total_pages_all_files += 1
        print(f"Found {len(all_files_to_process)} files to process with {total_pages_all_files} total pages.")
        num_workers_to_display = dots_ocr_parser.process_pool._max_workers
        print(f"Performance settings: file_concurrency={args.file_concurrency}, page_concurrency={args.page_concurrency}, num_cpu_workers={num_workers_to_display}")

        if args.enable_table_screenshot: print("Feature enabled: Table screenshots will be saved.")
        else: print("Feature disabled: Table screenshots will NOT be saved. (Use --enable_table_screenshot to turn on)")

        if args.skip_uncaptioned_images: print("Feature enabled: Images without captions will be SKIPPED.")
        else: print("Feature disabled: All images will be included. (Use --skip_uncaptioned_images to turn on)")
        
        await dots_ocr_parser.initialize()
        semaphore = asyncio.Semaphore(args.file_concurrency)
        
        page_progress = tqdm(total=total_pages_all_files, desc="Processing pages", unit="pages", position=1, smoothing=0.4)
        
        async def process_file_with_semaphore(file_path):
            nonlocal successful_files, perfectly_successful_files, processed_pages, skipped_blank_pages, failed_pages
            async with semaphore:
                relative_path = file_path.relative_to(input_root); target_output_dir = output_root / relative_path.parent
                if file_path.suffix.lower() == '.pdf': page_progress.set_description(f"Processing pages (File: {file_path.name})")
                def page_callback(): nonlocal processed_pages; page_progress.update(1); processed_pages += 1
                
                # 将 rename 参数传递给 parse_file 函数
                results = await dots_ocr_parser.parse_file(
                    str(file_path), 
                    output_dir=str(target_output_dir), 
                    prompt_mode=args.prompt, 
                    bbox=args.bbox, 
                    page_progress_callback=page_callback, 
                    skip_blank_pages=args.skip_blank_pages,
                    rename_to=args.rename  # 传递重命名参数
                )
                
                if results and 'skipped_blank_pages' in results[0]:
                    skipped_blank_pages += results[0].get('skipped_blank_pages', 0)
                    
                is_file_successful = results and all(r.get('status') in dots_ocr_parser.SUCCESS_STATUSES or r.get('skipped') for r in results)
                    
                if is_file_successful:
                    successful_files += 1
                    is_file_perfect = all(r.get('status') in ['success', 'skipped_blank'] or r.get('skipped') for r in results)
                    if is_file_perfect:
                        perfectly_successful_files += 1
                    if file_path.suffix.lower() != '.pdf': 
                        # 对于非PDF文件，确保进度条也被更新
                        if not any(r.get('skipped') for r in results):
                            page_progress.update(1); processed_pages += 1
                else:
                    TQDM_BASE.write(f"Failed to process: {file_path}")
                    if results:
                        for result in results:
                            if result.get('status') not in dots_ocr_parser.SUCCESS_STATUSES and not result.get('skipped'):
                                failed_pages.append({'file_path': str(file_path), 'filename': file_path.name, 'page_no': result.get('page_no', 'N/A'), 'original_page_num': result.get('original_page_num', 'N/A'), 'error': result.get('error', 'Unknown error'), 'status': result.get('status', 'unknown')})

        file_progress = tqdm(all_files_to_process, desc="Processing files", unit="files", position=0, smoothing=0.2)
        tasks = [process_file_with_semaphore(fp) for fp in all_files_to_process]
        for task in asyncio.as_completed(tasks):
            try:
                await task
            except Exception as e:
                TQDM_BASE.write(f"A file processing task failed unexpectedly: {e}")
            finally:
                file_progress.update(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down gracefully...")

    finally:
        # 无论程序是正常完成还是被中断，这个块都会被执行
        print("\nProcess finished or interrupted. Shutting down resources...")
        if file_progress:
            file_progress.close()
        if page_progress:
            page_progress.close()
            
        if dots_ocr_parser:
            await dots_ocr_parser.shutdown()

    # --- 摘要打印 ---
    # 这部分代码现在在 finally 块之后，确保即使中断也能打印出当前状态
    print("\n" + "="*25 + "\n      Processing Summary\n" + "="*25)
    
    total_attempted_files = len(all_files_to_process)
    total_original_files = total_attempted_files + skipped_files
    
    print(f"Original files found: {total_original_files}")
    if skipped_files > 0:
        print(f"Already processed files skipped: {skipped_files}")
        print(f"New files processed / attempted: {total_attempted_files}")
    
    print(f"Successfully processed: {successful_files}\nFailed to process: {total_attempted_files - successful_files}")

    if successful_files > 0:
        print(f"  - Flawlessly processed: {perfectly_successful_files}")
        print(f"  - Processed with fallbacks: {successful_files - perfectly_successful_files}")
    
    if total_attempted_files > 0: print(f"Success rate (new files): {successful_files/total_attempted_files*100:.1f}%")
    if total_original_files > 0: print(f"Overall completion: {successful_files + skipped_files}/{total_original_files} files ({(successful_files + skipped_files)/total_original_files*100:.1f}%)")
    
    print(f"\nOriginal total pages found: {total_pages_all_files}")
    print(f"Total pages processed (incl. blank): {processed_pages}")
    
    if args.skip_blank_pages:
        non_blank_processed = processed_pages - skipped_blank_pages
        print(f"  - Blank pages skipped: {skipped_blank_pages}")
        print(f"  - Non-blank pages processed: {non_blank_processed}")
        if total_pages_all_files > 0:
            print(f"Blank page ratio: {skipped_blank_pages / total_pages_all_files * 100:.1f}%")
            print(f"Processing completion: {processed_pages}/{total_pages_all_files} pages ({processed_pages / total_pages_all_files * 100:.1f}%)")
    elif total_pages_all_files > 0:
        print(f"Page completion rate: {processed_pages / total_pages_all_files * 100:.1f}%")
    
    if failed_pages:
        print(f"\n❌ Failed Pages Details ({len(failed_pages)} pages):" + "\n" + "-" * 50)
        for i, page in enumerate(failed_pages[:10]): print(f"  {i+1}. File: {page['filename']}\n     Page: {page['original_page_num']}\n     Error: {page['error']}\n     Status: {page['status']}\n")
        if len(failed_pages) > 10: print(f"  ... and {len(failed_pages) - 10} more failed pages")
        print("-" * 50)
    
    if dots_ocr_parser:
        summary = dots_ocr_parser.monitor.get_summary()
        print("\n" + "="*25 + "\n     Performance Summary\n" + "="*25)
        print(f"Total processing time: {summary['total_time']:.2f}s\nTotal inference requests: {summary['total_requests']}\nAverage inference time: {summary['avg_inference_time']:.2f}s")
        if summary['total_requests'] > 0 and summary['total_time'] > 0:
            print(f"Overall throughput: {summary['total_requests'] / summary['total_time']:.2f} pages/sec")
        print(f"Total errors during inference: {summary['total_errors']}")
        if summary['total_retries'] > 0: print(f"Total retries performed: {summary['total_retries']}")
        if summary['error_types']:
            print("Error types breakdown:"); [print(f"  - {error_type}: {count}") for error_type, count in summary['error_types'].items()]
    
    print("="*25 + "\n\nBatch processing finished!")

if __name__ == "__main__":
    asyncio.run(main())