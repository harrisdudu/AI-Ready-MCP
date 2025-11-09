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
from typing import List, Tuple, Optional, Dict, Set
from concurrent.futures import ProcessPoolExecutor

from tqdm.asyncio import tqdm
from tqdm import tqdm as TQDM_BASE
from PIL import Image
Image.MAX_IMAGE_PIXELS = 600000000
import fitz
import functools
import shutil
from datetime import datetime
import cv2
from pyzbar import pyzbar
import imagehash
from collections import defaultdict
import numpy as np
import traceback
import contextlib  # 用于重定向 stderr
import io          # 用于捕获 stderr
from pyzbar.pyzbar import ZBarSymbol # 用于指定 pyzbar 检测类型
import aiofiles

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

# 1. 纯计算逻辑：哈希聚类找出重复组
def find_duplicate_filenames(name_hash_map: Dict[str, imagehash.ImageHash], 
                             group_threshold: int, 
                             similarity_pct: float) -> Set[str]:
    if len(name_hash_map) < group_threshold:
        return set()
    sample_hash = next(iter(name_hash_map.values()))
    # phash 默认是 64 位, hash_size = 64
    hash_size = len(str(sample_hash)) * 4
    dist_threshold = int(hash_size * (1 - similarity_pct / 100.0))
    
    filenames = list(name_hash_map.keys())
    visited = set()
    files_to_delete = set()
    
    for i in range(len(filenames)):
        f = filenames[i]
        if f in visited: continue
        
        current_group = {f}
        visited.add(f)
        hash_f = name_hash_map[f]
        
        for j in range(i + 1, len(filenames)):
            g = filenames[j]
            if g in visited: continue
            
            if (hash_f - name_hash_map[g]) <= dist_threshold:
                current_group.add(g)
                visited.add(g)
                
        if len(current_group) >= group_threshold:
            files_to_delete.update(current_group)
            
    return files_to_delete

# 2. 文本处理逻辑：清理 Markdown 引用
def clean_markdown_content(md_content: str, files_to_remove: Set[str]) -> str:
    if not files_to_remove:
        return md_content
    # 构建更安全的正则表达式以匹配文件名
    pattern_str = r'!\[[^\]]*\]\s*\(\s*images/(?:' + '|'.join(re.escape(fn) for fn in files_to_remove) + r')\s*\)\n?'
    cleaned_content = re.sub(pattern_str, '', md_content)
    # 清理连续的空行
    cleaned_content = re.sub(r'(\n\s*){3,}', '\n\n', cleaned_content).strip()
    return cleaned_content

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
    渲染 PDF 指定页并落盘到 tmp_dir，仅跨进程返回路径，避免把大图像对象回传到主进程。
    - 自动按 max_pixels 计算有效 DPI，源头控像素。
    - 如启用空白页跳过，子进程直接检测并返回 'skipped_blank'。
    """
    import os, math, tempfile, traceback
    import fitz
    from PIL import Image
    from dots_ocr.utils.image_utils import fetch_image
    from dots_ocr.utils.doc_utils import is_blank_page as _is_blank

    (input_path, page_idx, dpi, skip_blank_pages,
     white_threshold, noise_threshold, min_pixels, max_pixels, tmp_dir) = task_args

    try:
        with fitz.open(input_path) as doc:
            if page_idx >= len(doc):
                return {'status': 'error', 'error': f'Page index {page_idx} out of bounds.', 'page_idx': page_idx}

            page = doc[page_idx]

            # 依据 max_pixels 预估并调整 DPI（源头控像素）
            if max_pixels:
                w_pt, h_pt = page.rect.width, page.rect.height
                scale = dpi / 72.0
                est_pixels = (w_pt * scale) * (h_pt * scale)
                if est_pixels > max_pixels:
                    shrink = math.sqrt(max_pixels / max(1.0, est_pixels))
                    dpi_eff = max(36, int(dpi * shrink))
                else:
                    dpi_eff = dpi
            else:
                dpi_eff = dpi

            pix = page.get_pixmap(dpi=dpi_eff)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # 空白页快速跳过
        if skip_blank_pages and _is_blank(img, white_threshold, noise_threshold):
            return {'status': 'skipped_blank', 'page_idx': page_idx}

        # 二次像素护栏（与你原本的 fetch_image 保持一致）
        processed = fetch_image(img, min_pixels=min_pixels, max_pixels=max_pixels)

        os.makedirs(tmp_dir, exist_ok=True)
        orig_path = os.path.join(tmp_dir, f"p{page_idx:06d}_orig.jpg")
        proc_path = os.path.join(tmp_dir, f"p{page_idx:06d}_proc.jpg")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
        img.save(orig_path, "JPEG", quality=92)
        processed.save(proc_path, "JPEG", quality=92)

        # 减少子进程堆驻留
        del img, processed

        return {
            'status': 'success',
            'page_idx': page_idx,
            'origin_path': orig_path,
            'processed_path': proc_path,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': f'{e}\n{traceback.format_exc()}',
            'page_idx': page_idx
        }
    
# ==============================================================================
#  独立的跨页后处理工作函数（用于进程池）
# ==============================================================================
def _run_all_post_processing_worker(parser_init_kwargs: dict, layout_data: list) -> list:
    """
    这个函数在单独的进程中运行，用于执行计算密集型的跨页后处理。
    它接收解析器的初始化参数，在子进程中创建一个干净的实例来执行任务。
    """
    # 1. 在子进程中创建一个临时的、干净的解析器实例
    # 这个实例没有网络连接、信号量或进程池等不可序列化的属性
    parser = DotsOCRParserOptimized(**parser_init_kwargs)
    
    # 2. 执行原有的后处理逻辑
    # 2.1 页内匹配
    for result in layout_data:
        if 'cells' in result and result.get('cells'):
            result['cells'] = parser._perform_intra_page_matching(result['cells'])
    
    # 2.2 跨页表格匹配
    parser._perform_cross_page_table_caption_matching(layout_data)
    
    # 2.3 跨页文本合并
    parser._perform_cross_page_text_merging(layout_data)
    
    # 3. 返回处理后的数据
    return layout_data
    
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
        self.num_cpu_workers = kwargs.get('num_cpu_workers', 32)

        md_gen_concurrency = kwargs.get('md_gen_concurrency', 0)
        if md_gen_concurrency <= 0:
            # 如果是0或负数，自动设置为CPU worker的数量，这是一个合理的默认值
            # 因为MD生成也依赖CPU进行渲染
            num_workers_for_md = self.num_cpu_workers if self.num_cpu_workers > 0 else max(os.cpu_count() // 2, 32)
            self.md_gen_concurrency = num_workers_for_md
        else:
            self.md_gen_concurrency = md_gen_concurrency
        
        # 创建一个用于MD生成的信号量
        self.md_gen_semaphore = asyncio.Semaphore(self.md_gen_concurrency)
        TQDM_BASE.write(f"MD Generation Concurrency set to: {self.md_gen_concurrency}")

        self.queue_size = kwargs.get('queue_size', 300)
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
        self.filter_qr_barcodes = kwargs.get('filter_qr_barcodes', True) # 默认开启二维码/条形码过滤
        self.filter_duplicates = kwargs.get('filter_duplicates', True) # 默认开启重复图片过滤
        self.client = None
        self.monitor = PerformanceMonitor()

        # === 保存可序列化的初始化参数 ===
        self.init_kwargs = kwargs.copy()

        # --- 检查二维码/条形码依赖库 ---
        self.qr_scanner_available = False
        if self.filter_qr_barcodes:
            try:
                import cv2
                from pyzbar import pyzbar
                import numpy as np
                self.qr_scanner_available = True
                TQDM_BASE.write("QR/barcode scanning dependencies (pyzbar, opencv) are available.")
            except ImportError:
                TQDM_BASE.write("WARNING: QR/barcode filtering is enabled, but 'pyzbar' or its dependency 'libzbar' is not installed. This feature will be disabled.")
        
        ## 创建一个全局信号量来精确控制API总并发数。
        self.page_semaphore = asyncio.Semaphore(self.page_concurrency)
        
        num_workers = self.num_cpu_workers if self.num_cpu_workers > 0 else max(os.cpu_count() // 2, 32)
        TQDM_BASE.write(f"Initializing ProcessPoolExecutor with {num_workers} workers.")
        self.process_pool = ProcessPoolExecutor(max_workers=num_workers)
        
        TQDM_BASE.write(f"Using vLLM model with optimized producer-consumer model.")
        TQDM_BASE.write(f"Global Page Concurrency: {self.page_concurrency} | CPU Workers: {num_workers} | Queue Size: {self.queue_size}")
        TQDM_BASE.write(f"Feature: Save page JSON -> {'ENABLED' if self.save_page_json else 'DISABLED'}")
        TQDM_BASE.write(f"Feature: Save page layout image -> {'ENABLED' if self.save_page_layout else 'DISABLED'}")
        
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    async def initialize(self):
        if not self.use_hf:
            TQDM_BASE.write("Initializing async client...")
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
    
    async def _generate_md_for_one_page(self, page_idx, all_pages_layout_data, images_dir, time_warn_s: float = 5.0):
        """
        根据 PASS-1/2 的结果生成单页 MD：
        - 如 status == success，按 origin_image_path 打开原图，调用 layoutjson2md_full_robust；
        - 其它降级分支直接使用已有 md_content；
        - base64 落地与命名仍在线程池中完成；
        """
        loop = asyncio.get_running_loop()
        result = all_pages_layout_data[page_idx]
        page_num = page_idx + 1
        status = result.get('status')
        md_content = ""
        t0 = time.time()

        async with self.md_gen_semaphore:
            try:
                if status == 'success':
                    origin_path = result.get('origin_image_path')
                    original_cells = result.get('cells', []) or []

                    # 过滤页眉/页脚
                    cells_to_render = [
                        c for c in original_cells
                        if (self.keep_page_header or c.get('category') != 'Page-header')
                        and (self.keep_page_footer or c.get('category') != 'Page-footer')
                    ]

                    def _mk_md_sync():
                        from PIL import Image
                        with Image.open(origin_path) as _o:
                            orig = _o.convert('RGB')
                            return layoutjson2md_full_robust(orig, cells_to_render, 'text', False)

                    md_content = await loop.run_in_executor(None, _mk_md_sync)

                elif status in ('success_fallback_text', 'success_fallback_image'):
                    md_content = result.get('md_content', '')

                elif status != 'skipped_blank':
                    md_content = f"\n\n[Page {page_num} could not be rendered or processed]\n\n"

                # base64 -> 文件，命名、过滤二维码/跳过无标题图
                if md_content:
                    # 需要 origin_image 才能截表格，读取一次
                    origin_path = result.get('origin_image_path')
                    original_cells = result.get('cells', []) or []

                    def _img_proc_sync():
                        from PIL import Image
                        origin_image = None
                        try:
                            if origin_path and os.path.exists(origin_path):
                                with Image.open(origin_path) as _o:
                                    origin_image = _o.convert('RGB').copy()
                            return self._process_base64_images_with_custom_naming(
                                md_content, images_dir, page_num, original_cells, origin_image
                            )
                        finally:
                            origin_image = None

                    page_md_obj = await loop.run_in_executor(None, _img_proc_sync)
                    return page_md_obj

                return None
            finally:
                t = time.time() - t0
                if t > time_warn_s:
                    TQDM_BASE.write(f"[MD] Page {page_num} MD-gen took {t:.1f}s (windowed).")
    
    # (在 DotsOCRParserOptimized 类中)
    def _is_qr_or_barcode(self, image: Image.Image) -> bool:
        """
        使用新的两阶段健壮检测逻辑，判断 PIL.Image 对象是否包含二维码或条形码。
        1. 快速 OpenCV QR 检测。
        2. 全面的 PyZBar 检测（带 stderr 捕获以增强稳定性）。
        """
        if not self.qr_scanner_available:
            return False
        
        is_qr = False
        try:
            # 将 PIL Image 转换为 OpenCV 格式
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h, w = cv2_image.shape[:2]
            
            # 忽略过小的图片，避免误判
            if min(h, w) <= 40:
                return False

            # 1. 先尝试OpenCV的QR码检测（更快）
            try:
                qr = cv2.QRCodeDetector()
                ok, _, points, _ = qr.detectAndDecodeMulti(cv2_image)
                if ok and points is not None and len(points) > 0:
                    is_qr = True
            except Exception:
                # 某些 OpenCV 版本可能没有 Multi，回退到单码检测
                try:
                    _, pts, _ = qr.detectAndDecode(cv2_image)
                    if pts is not None and len(pts) > 0:
                        is_qr = True
                except Exception:
                    pass # OpenCV 检测失败，继续

            # 2. OpenCV未检测到，再用PyZBar检测条形码/QR码（更全面但可能不稳定）
            if not is_qr:
                try:
                    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
                    # 自动二值化预处理，提高检测率
                    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    stderr_capture = io.StringIO()
                    decoded_objects = []
                    
                    # 重定向stderr以捕获ZBar内部的C断言错误，防止程序崩溃
                    with contextlib.redirect_stderr(stderr_capture):
                        decoded_objects = pyzbar.decode(
                            binary_image,
                            symbols=[  # 明确指定检测类型以提高效率
                                ZBarSymbol.QRCODE, ZBarSymbol.CODE128, ZBarSymbol.EAN13,
                                ZBarSymbol.EAN8, ZBarSymbol.UPCA, ZBarSymbol.UPCE,
                                ZBarSymbol.CODE39, ZBarSymbol.CODE93
                            ]
                        )
                    
                    stderr_output = stderr_capture.getvalue()
                    
                    # 仅当无断言错误且有解码结果时，视为有效
                    if decoded_objects and "Assertion" not in stderr_output:
                        is_qr = True
                        
                except Exception:
                    pass # PyZBar 也失败了

        except Exception as e:
            TQDM_BASE.write(f"Warning: QR/barcode detection failed with an unexpected error: {e}")
            return False
            
        return is_qr

    async def _filter_duplicate_images(self, md_content: str, images_dir: str, group_threshold: int = 3, similarity_threshold_pct: float = 75.0) -> str:
        """
        使用新的聚类算法对 Markdown 中引用的图片进行近似去重。
        - 计算感知哈希 (phash)。
        - 使用 find_duplicate_filenames 函数进行聚类。
        - 如果一个“相似簇”的图片数量达到阈值，则删除整个簇的文件和引用。
        """
        
        # 定义同步阻塞逻辑的内部函数，以便在线程池中执行
        def _sync_filter_logic():
            if not os.path.isdir(images_dir):
                return md_content

            try:
                # 1. 提取 Markdown 中引用的所有图片文件名
                ref_pattern = re.compile(r'!\[[^\]]*\]\s*\(\s*images/([^)]+)\s*\)')
                referenced_filenames = set(ref_pattern.findall(md_content))
                
                if len(referenced_filenames) < group_threshold:
                    return md_content

                # 2. 计算所有引用文件的感知哈希
                name_hash_map = {}
                for fn in referenced_filenames:
                    path = os.path.join(images_dir, fn)
                    if not os.path.exists(path):
                        continue
                    try:
                        with Image.open(path) as img:
                            name_hash_map[fn] = imagehash.phash(img)
                    except Exception as e:
                        TQDM_BASE.write(f"Warning: Could not calculate hash for image {fn}: {e}")

                if len(name_hash_map) < group_threshold:
                    return md_content

                # 3. 调用新的聚类函数找到要删除的重复文件组
                files_to_delete = find_duplicate_filenames(
                    name_hash_map, group_threshold, similarity_threshold_pct
                )

                if not files_to_delete:
                    TQDM_BASE.write("No significant near-duplicate image groups found.")
                    return md_content

                # TQDM_BASE.write(f"Found {len(files_to_delete)} near-duplicate images to remove.")

                # 4. 使用新的清理函数从 Markdown 中移除引用
                cleaned_md_content = clean_markdown_content(md_content, files_to_delete)

                # 5. 从磁盘删除物理文件
                for fn in files_to_delete:
                    path = os.path.join(images_dir, fn)
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError as e:
                        TQDM_BASE.write(f"Error removing duplicate image file {path}: {e}")

                return cleaned_md_content

            except Exception as e:
                TQDM_BASE.write(f"An unexpected error occurred during duplicate image filtering: {e}")
                TQDM_BASE.write(traceback.format_exc())
                return md_content

        # 在默认的线程池中异步执行同步逻辑
        loop = asyncio.get_running_loop()
        result_md = await loop.run_in_executor(None, _sync_filter_logic)
        return result_md
    
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
        内存友好的并发竞速重试：
        - 对同一张大图只进行一次编码（JPEG bytes），所有并发分支复用同一份 payload；
        - 任一分支成功即立刻取消其余分支，最小化显存/内存与带宽占用；
        - 与已适配的 inference_async.inference_with_vllm（支持 bytes/data-URL）配合良好。
        """
        if num_attempts <= 0:
            return None

        # ★ 仅编码一次：把图像统一成 JPEG bytes 作为共享载荷
        payload = image
        try:
            if isinstance(image, (bytes, bytearray, str)):
                # bytes / data-URL / 本地路径字符串 直接复用
                payload = image
            else:
                bio = BytesIO()
                (image.convert("RGB") if getattr(image, "mode", None) != "RGB" else image).save(bio, "JPEG", quality=92)
                payload = bio.getvalue()
                # 明确释放临时 buffer 引用
                del bio
        except Exception as e:
            # 编码失败则退回原始对象（下层会再做一次兜底适配）
            TQDM_BASE.write(f"[race] Failed to pre-encode image once, fallback to original object. Reason: {e}")

        # 构建并发任务（每路都复用同一个 payload，不重复编码）
        tasks = [asyncio.create_task(self._inference_with_vllm(payload, prompt)) for _ in range(num_attempts)]
        last_exception = None

        try:
            pending = set(tasks)

            # 先等第一批完成者；若没有成功结果，再继续等待其余 pending
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    try:
                        res = t.result()
                        if res is not None:
                            # 成功即取消剩余所有任务
                            for p in pending:
                                p.cancel()
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                            return res
                    except Exception as exc:
                        last_exception = exc
                        # 继续等待其他任务的结果
                        continue

            # 全部任务完成但没有成功结果
            if last_exception:
                raise last_exception
            raise Exception("All concurrent retry attempts failed without a specific exception.")
        finally:
            # 双重保险：确保所有未完成任务被取消并清理
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    # (在 DotsOCRParserOptimized 类中)
    def _process_base64_images_with_custom_naming(self, md_content, images_dir, page_num, cells, origin_image=None):
        """
        - 根据开关过滤二维码和条形码。
        - 如果 self.skip_uncaptioned_images 为 True，则不保存没有标题的图片，并从MD内容中移除它们。
        """
        os.makedirs(images_dir, exist_ok=True)
        base64_pattern = r'!\[[^\]]*\]\s*\(\s*data:image/[^;]+;base64,([^)]+)\)'
        
        pictures = [cell for cell in cells if cell['category'] == 'Picture']
        picture_counter = 0
        
        def replace_base64(match):
            nonlocal picture_counter
            try:
                base64_data = match.group(1)
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data))
                
                # --- 新增：二维码/条形码过滤逻辑 ---
                if self.filter_qr_barcodes and self._is_qr_or_barcode(image):
                    picture_counter += 1
                    return "" # 返回空字符串，删除图片引用

                # 获取与当前base64图像匹配的 'Picture' 单元格信息
                current_pic_cell = pictures[picture_counter] if picture_counter < len(pictures) else None
                
                # 检查开关是否开启，以及当前图片是否有标题
                if self.skip_uncaptioned_images and (not current_pic_cell or not current_pic_cell.get('caption_text')):
                    picture_counter += 1
                    return ""  # 返回空字符串，从而从Markdown中删除此图片引用

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
        
        # 表格截图逻辑
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
        处理单页：主尝试 + 并发竞速重试 + 降级。
        - 支持 page_data 提供 'origin_image_path' / 'processed_image_path'（首选），
        仅在需要时打开 PIL.Image，并在同一函数内尽快释放。
        - 推理阶段直接使用 processed_image_path（字符串）作为 payload，避免重复编码。
        """
        origin_path    = page_data.get('origin_image_path')
        processed_path = page_data.get('processed_image_path')
        origin_image_obj   = page_data.get('origin_image')     # 向后兼容
        processed_image_obj= page_data.get('processed_image')  # 向后兼容

        original_page_num = page_data['original_page_num']
        page_idx = page_data['page_idx']
        save_dir = page_data.get('save_dir')
        prompt_mode = page_data['prompt_mode']
        bbox = page_data.get('bbox')

        last_error = None
        last_raw_response_for_fallback = None

        # 准备 prompt：如需 bbox，需要已知尺寸；简单打开一次获取 size 后立刻关闭
        def _open_for_size(p):
            from PIL import Image
            with Image.open(p) as im:
                return im.size

        if bbox is not None and prompt_mode == 'prompt_grounding_ocr':
            if processed_image_obj is not None:
                w, h = processed_image_obj.size
            elif processed_path:
                w, h = await asyncio.get_running_loop().run_in_executor(None, _open_for_size, processed_path)
            else:
                w, h = (0, 0)
            # 这里 origin_image 仅用于 pre_process_bboxes 的校准；不开整图也行
            prompt = self.get_prompt(prompt_mode, bbox=bbox, origin_image=None,
                                    image=type('Tmp', (), {'width': w, 'height': h})())
        else:
            prompt = self.get_prompt(prompt_mode)

        # === 主尝试 ===
        try:
            # 推理：优先用路径，避免把大图编码到内存
            payload = processed_path if processed_path else (processed_image_obj or origin_image_obj)
            response = await self._inference_with_vllm(payload, prompt)
            last_raw_response_for_fallback = response
            if not response:
                raise ValueError("Inference call returned an empty response.")

            # post_process 需要 PIL.Image；按需打开，两张图都开（有些规则需要两张）
            def _post_sync():
                from PIL import Image
                if processed_image_obj is not None:
                    proc = processed_image_obj
                else:
                    with Image.open(processed_path) as _p:
                        proc = _p.convert('RGB').copy()
                if origin_image_obj is not None:
                    orig = origin_image_obj
                else:
                    with Image.open(origin_path) as _o:
                        orig = _o.convert('RGB').copy()

                cells, _ = post_process_output(
                    response, prompt_mode, orig, proc,
                    min_pixels=self.min_pixels, max_pixels=self.max_pixels
                )
                return cells, orig  # 把 orig 返回，后面可能需要画布局/表格截图

            loop = asyncio.get_running_loop()
            cells, origin_image_for_saving = await loop.run_in_executor(None, _post_sync)

            if not (cells and isinstance(cells, list) and all(isinstance(c, dict) and 'bbox' in c for c in cells)):
                raise ValueError("Post-processing yielded invalid or empty cells data.")

            # （可选）保存单页 JSON / 布局图
            save_name_base = f"{Path(page_data['filename']).stem}_page_{original_page_num}"
            await self._save_intermediate_outputs_async(save_dir, save_name_base, response, cells, origin_image_for_saving)

            # 及时释放内存驻留
            del origin_image_for_saving

            return {
                'page_no': page_idx, 'original_page_num': original_page_num,
                'status': 'success',
                'cells': cells,
                'page_json_path': None, 'page_layout_path': None
            }

        except Exception as e:
            last_error = e
            TQDM_BASE.write(f"⚠️ Page {original_page_num} initial attempt failed: {type(e).__name__}. Triggering retries...")

        # === 并发竞速重试 ===
        if self.concurrent_retries > 0:
            try:
                self.monitor.record_retry(self.concurrent_retries)
                TQDM_BASE.write(f"  Racing {self.concurrent_retries} concurrent requests for page {original_page_num}...")
                payload = processed_path if processed_path else (processed_image_obj or origin_image_obj)
                response = await self._race_inference_attempts(payload, prompt, num_attempts=self.concurrent_retries)
                last_raw_response_for_fallback = response
                if not response:
                    raise ValueError("All concurrent inference calls returned an empty response.")

                def _post_sync2():
                    from PIL import Image
                    if processed_image_obj is not None:
                        proc = processed_image_obj
                    else:
                        with Image.open(processed_path) as _p:
                            proc = _p.convert('RGB').copy()
                    if origin_image_obj is not None:
                        orig = origin_image_obj
                    else:
                        with Image.open(origin_path) as _o:
                            orig = _o.convert('RGB').copy()
                    cells, _ = post_process_output(
                        response, prompt_mode, orig, proc,
                        min_pixels=self.min_pixels, max_pixels=self.max_pixels
                    )
                    return cells, orig

                loop = asyncio.get_running_loop()
                cells, origin_image_for_saving = await loop.run_in_executor(None, _post_sync2)

                if not (cells and isinstance(cells, list) and all(isinstance(c, dict) and 'bbox' in c for c in cells)):
                    raise ValueError("Post-processing yielded invalid or empty cells data from raced response.")

                save_name_base = f"{Path(page_data['filename']).stem}_page_{original_page_num}"
                await self._save_intermediate_outputs_async(save_dir, save_name_base, response, cells, origin_image_for_saving)
                del origin_image_for_saving

                TQDM_BASE.write(f"  ✅ Concurrent retry succeeded for page {original_page_num}.")
                return {
                    'page_no': page_idx, 'original_page_num': original_page_num,
                    'status': 'success',
                    'cells': cells,
                    'page_json_path': None, 'page_layout_path': None
                }

            except Exception as e:
                last_error = e
                TQDM_BASE.write(f"  ❌ All {self.concurrent_retries} concurrent attempts failed for page {original_page_num}. Final error: {type(e).__name__}")

        # === 降级：文本抽取 ===
        TQDM_BASE.write(f"Page {original_page_num} failed all attempts. Last known error: {type(last_error).__name__}. Attempting fallbacks.")
        if last_raw_response_for_fallback:
            try:
                def _text_fallback_sync():
                    from PIL import Image
                    if processed_image_obj is not None:
                        proc = processed_image_obj
                    else:
                        with Image.open(processed_path) as _p:
                            proc = _p.convert('RGB').copy()
                    if origin_image_obj is not None:
                        orig = origin_image_obj
                    else:
                        with Image.open(origin_path) as _o:
                            orig = _o.convert('RGB').copy()
                    cells_for_text, _ = post_process_output(
                        last_raw_response_for_fallback, prompt_mode, orig, proc,
                        min_pixels=self.min_pixels, max_pixels=self.max_pixels
                    )
                    return cells_for_text

                cells_for_text = await asyncio.get_running_loop().run_in_executor(None, _text_fallback_sync)
                if cells_for_text and isinstance(cells_for_text, list):
                    md_content_simple = layoutjson2md_simple_extract(cells_for_text, 'text', True)
                    if md_content_simple and md_content_simple.strip():
                        TQDM_BASE.write(f"  ✅ Text extraction fallback succeeded for page {original_page_num}.")
                        return {
                            'page_no': page_idx, 'original_page_num': original_page_num,
                            'status': 'success_fallback_text',
                            'md_content': md_content_simple,
                            'cells': [],
                            'page_json_path': None, 'page_layout_path': None
                        }
            except Exception as text_fallback_e:
                TQDM_BASE.write(f"  ⚠️ Text extraction fallback also failed for page {original_page_num}. Reason: {text_fallback_e}")

        # === 降级：截图保底 ===
        TQDM_BASE.write(f"Page {original_page_num} failed text extraction. Saving error screenshot as final placeholder.")
        try:
            images_dir = os.path.join(save_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            bad_image_filename = f"page_{original_page_num}_bad.jpg"
            bad_image_path = os.path.join(images_dir, bad_image_filename)

            def _save_bad_sync():
                from PIL import Image
                if origin_image_obj is not None:
                    orig = origin_image_obj
                    if orig.mode != 'RGB': orig = orig.convert('RGB')
                    orig.save(bad_image_path, "JPEG", quality=95)
                else:
                    with Image.open(origin_path) as _o:
                        orig = _o.convert('RGB')
                        orig.save(bad_image_path, "JPEG", quality=95)

            await asyncio.get_running_loop().run_in_executor(None, _save_bad_sync)

            if self.badcase_collection_dir:
                try:
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    date_dir = os.path.join(self.badcase_collection_dir, today_str)
                    os.makedirs(date_dir, exist_ok=True)
                    badcase_filename_base = Path(page_data.get('filename', 'unknown_file')).stem
                    badcase_file_name = f"{badcase_filename_base}_page_{original_page_num}.jpg"
                    shutil.copyfile(bad_image_path, os.path.join(date_dir, badcase_file_name))
                except Exception as copy_e:
                    TQDM_BASE.write(f"  WARNING: Failed to copy badcase screenshot. Reason: {copy_e}")

            final_md = f"![{bad_image_filename}](images/{bad_image_filename})"
            TQDM_BASE.write(f"  Fallback succeeded. Screenshot for page {original_page_num} saved.")
            return {
                'page_no': page_idx, 'original_page_num': original_page_num,
                'status': 'success_fallback_image', 'md_content': final_md,
                'cells': [],
                'page_json_path': None, 'page_layout_path': None
            }
        except Exception as e:
            TQDM_BASE.write(f"  CRITICAL: Image fallback failed for page {original_page_num}. Reason: {e}")
            return {
                'page_no': page_idx, 'original_page_num': original_page_num,
                'status': 'error', 'error': f"{type(last_error).__name__}: {last_error} | plus: {e}",
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
    async def parse_pdf(self, input_path, filename, prompt_mode, save_dir,
                    page_progress_callback=None, bbox=None, skip_blank_pages=False):
        """
        内存友好版：
        - 子进程只返回图片路径；
        - 生产者仅投 page_idx，消费者再去进程池渲染；
        - PASS-3 流式写出，限制在途任务数量并按页序写入。
        """
        import tempfile

        loop = asyncio.get_running_loop()
        TQDM_BASE.write(f"[{filename}] PASS 1/3: Concurrently processing pages to extract layout data...")

        # --- 轻量队列：限制为 ~2x page_concurrency，避免大量滞留 ---
        qsize = min(self.queue_size, max(2 * self.page_concurrency, 64))
        page_queue = asyncio.Queue(maxsize=qsize)

        results_storage = {}
        total_pages_expected = 0

        # --- 专属临时目录（随文件清理） ---
        tmp_dir = tempfile.mkdtemp(prefix=f"{filename}_tmp_", dir=save_dir)

        # === Producer：只投 page_idx ===
        async def page_producer(num_consumers):
            nonlocal total_pages_expected
            try:
                with fitz.open(input_path) as doc:
                    total_pages_expected = doc.page_count or 0

                if total_pages_expected == 0:
                    TQDM_BASE.write(f"Warning: File {filename} has 0 pages or could not be opened.")
                    for _ in range(num_consumers):
                        await page_queue.put(None)
                    return

                for page_idx in range(total_pages_expected):
                    await page_queue.put(page_idx)
            except Exception as e:
                import traceback
                TQDM_BASE.write(f"FATAL error in page producer for file '{filename}': {e}\n{traceback.format_exc()}")
            finally:
                for _ in range(num_consumers):
                    await page_queue.put(None)

        # === Consumer：进程池渲染 -> 读取图片 -> 推理 -> 结果入存储 ===
        async def page_consumer(consumer_id):
            while True:
                page_idx = await page_queue.get()
                if page_idx is None:
                    break
                page_num = page_idx + 1
                try:
                    task_args = (
                        input_path, page_idx, self.dpi, skip_blank_pages,
                        self.blank_white_threshold, self.blank_noise_threshold,
                        self.min_pixels, self.max_pixels, tmp_dir
                    )
                    pre = await loop.run_in_executor(self.process_pool, _process_pdf_page_worker, task_args)

                    if pre.get('status') != 'success':
                        status = pre.get('status', 'processing_error')
                        error  = pre.get('error', 'Unknown pre-processing error')
                        results_storage[page_num] = {
                            'page_no': page_idx, 'original_page_num': page_num,
                            'status': status, 'error': error if status != 'skipped_blank' else None
                        }
                        if page_progress_callback: page_progress_callback()
                        continue

                    # 直接把路径传递到推理阶段；不要复制 PIL.Image，不要删除临时文件
                    page_data = {
                        'page_idx': page_idx,
                        'original_page_num': page_num,
                        'prompt_mode': prompt_mode,
                        'bbox': bbox,
                        'filename': filename,
                        'save_dir': save_dir,
                        'origin_image_path': pre['origin_path'],
                        'processed_image_path': pre['processed_path'],
                    }

                    async with self.page_semaphore:
                        result = await self._process_single_page_optimized_streaming(page_data)

                    # PASS-1 存入结果；保留 origin 路径供 PASS-3 使用
                    if 'origin_image' in result:       # 兼容旧结构，确保不驻留
                        del result['origin_image']
                    result['origin_image_path']   = pre['origin_path']
                    result['processed_image_path'] = pre['processed_path']  # 如需节省磁盘，后面 PASS-3 完成后再统一清理 tmp_dir
                    results_storage[page_num] = result

                    if page_progress_callback: page_progress_callback()

                except Exception as e:
                    import traceback
                    TQDM_BASE.write(f"Critical error in consumer {consumer_id} for page {page_num} of {filename}: {e}\n{traceback.format_exc()}")
                    results_storage[page_num] = {
                        'page_no': page_idx, 'original_page_num': page_num,
                        'status': 'error', 'error': str(e)
                    }
                    if page_progress_callback: page_progress_callback()

        # 启动生产/消费
        num_consumers = self.page_concurrency
        producer_task = asyncio.create_task(page_producer(num_consumers))
        consumer_tasks = [asyncio.create_task(page_consumer(i)) for i in range(num_consumers)]
        await asyncio.gather(producer_task, *consumer_tasks)

        # 汇总 PASS-1 结果（轻量）
        all_pages_layout_data = [
            results_storage.get(i + 1, {
                'page_no': i, 'original_page_num': i + 1,
                'status': 'missing_result', 'error': 'Page data was lost.'
            }) for i in range(total_pages_expected)
        ]
        all_results_for_jsonl = []
        skipped_count = sum(1 for r in all_pages_layout_data if r.get('status') == 'skipped_blank')
        hard_failed_pages_count = sum(1 for r in all_pages_layout_data if r.get('status') not in self.SUCCESS_STATUSES)

        # === PASS-2：跨页后处理（线程池） ===
        TQDM_BASE.write(f"[{filename}] PASS 2/3: Performing cross-page analysis...")
        all_pages_layout_data = await loop.run_in_executor(
            self.process_pool, _run_all_post_processing_worker, self.init_kwargs, all_pages_layout_data
        )

        if hard_failed_pages_count > 0:
            TQDM_BASE.write(f"⚠️ File '{filename}' was processed with {hard_failed_pages_count} hard-failed page(s). Placeholders will be generated.")

        if not any(r.get('status') in self.SUCCESS_STATUSES for r in all_pages_layout_data):
            if skipped_count > 0 and skipped_count == total_pages_expected:
                TQDM_BASE.write(f"INFO: File '{filename}' consists entirely of {skipped_count} blank page(s). No .md file will be generated.")
            else:
                TQDM_BASE.write(f"ERROR: No content was extracted from '{filename}'. Aborting file creation.")
            # 清理 tmp_dir
            try: shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception: pass
            return [{"error": "No content could be generated", "file_path": input_path}]

        # === PASS-3：窗口化并发 + 顺序流式写出 ===
        TQDM_BASE.write(f"[{filename}] PASS 3/3: Generating final Markdown (streaming)...")
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        combined_md_path = self._get_unique_md_path(save_dir, filename)

        # 窗口适度放大，降低头阻塞概率（仍有限制，避免内存放飞）
        window = max(8, min(self.md_gen_concurrency // 2 if self.md_gen_concurrency > 0 else 16, 64))
        next_to_schedule = 0
        next_to_write = 0
        pending = {}

        async with aiofiles.open(combined_md_path, "w", encoding="utf-8") as md_file:
            # 初始填充窗口
            while next_to_schedule < total_pages_expected and len(pending) < window:
                pending[next_to_schedule] = asyncio.create_task(
                    self._generate_md_for_one_page(next_to_schedule, all_pages_layout_data, images_dir)
                )
                next_to_schedule += 1

            # 顺序写出（允许后面的任务先完成，但必须等位点页号才写）
            while next_to_write < total_pages_expected:
                if next_to_write not in pending:
                    while next_to_schedule < total_pages_expected and len(pending) < window:
                        pending[next_to_schedule] = asyncio.create_task(
                            self._generate_md_for_one_page(next_to_schedule, all_pages_layout_data, images_dir)
                        )
                        next_to_schedule += 1

                res = await pending.pop(next_to_write)
                page_text = (res or {}).get("md_content", "")

                if page_text:
                    if self.add_page_tag:
                        await md_file.write(f"{page_text}\n<special_page_num_tag>{next_to_write}</special_page_num_tag>\n")
                    else:
                        await md_file.write(page_text + "\n\n")

                # 写完立即释放 cells，避免驻留
                if next_to_write < len(all_pages_layout_data) and "cells" in all_pages_layout_data[next_to_write]:
                    all_pages_layout_data[next_to_write]["cells"] = None

                next_to_write += 1

        # （可选）图片近似去重：读回文本处理后再写回（文本相对小，影响可控）
        if self.filter_duplicates:
            try:
                async with aiofiles.open(combined_md_path, "r", encoding="utf-8") as f:
                    md_text = await f.read()
                md_text = re.sub(r'(\n\s*){3,}', '\n\n', md_text).strip()
                md_text = await self._filter_duplicate_images(md_text, images_dir)
                async with aiofiles.open(combined_md_path, "w", encoding="utf-8") as f:
                    await f.write(md_text)
            except Exception as e:
                TQDM_BASE.write(f"WARNING: duplicate image filtering failed for '{filename}': {e}")

        TQDM_BASE.write(f"[{filename}] Successfully saved final output to {combined_md_path}")

        # 组装 JSONL 结果
        for i in range(total_pages_expected):
            result = results_storage.get(i + 1, {})
            status = result.get('status', 'unknown_error')
            all_results_for_jsonl.append({
                'page_no': result.get('original_page_num', i + 1),
                'file_path': input_path,
                'status': status,
                'filename': filename,
                'error': result.get('error') if status not in self.SUCCESS_STATUSES else None
            })

        for res in all_results_for_jsonl:
            if res['status'] in self.SUCCESS_STATUSES:
                res['output_md_path'] = combined_md_path
                break

        # 清理 tmp_dir
        try: shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception: pass

        return all_results_for_jsonl

    # (在 DotsOCRParserOptimized 类中)
    async def parse_file(self, input_path: str, output_dir: str = "", prompt_mode: str = "prompt_layout_all_en", page_progress_callback=None, bbox=None, skip_blank_pages=False, rename_to: Optional[str] = None):
        output_dir = output_dir or self.output_dir; output_dir = os.path.abspath(output_dir)

        # --- 根据 rename_to 参数决定文件名 ---
        if rename_to:
            filename = os.path.splitext(os.path.basename(rename_to))[0]
        else:
            filename, _ = os.path.splitext(os.path.basename(input_path))

        save_dir = os.path.join(output_dir, filename)
        
        # 断点续传检查
        if self.enable_resume and not self.force_reprocess:
            is_processed, md_path = is_file_already_processed(input_path, output_dir, filename)
            if is_processed:
                TQDM_BASE.write(f"Skipping already processed file: {input_path} (MD file exists: {md_path})")
                return [{
                    'page_no': 0, 'original_page_num': 1, 'file_path': input_path,
                    'output_md_path': md_path, 'status': 'success', 'error': None,
                    'filename': filename, 'skipped': True
                }]
            else:
                # ==================================================================
                #  新增核心逻辑：清理不完整的旧输出
                # ==================================================================
                # 如果文件未被处理过（is_processed is False），但其输出目录
                # （可能包含旧的 images/ 目录）已存在，则将其彻底删除。
                if os.path.isdir(save_dir):
                    try:
                        TQDM_BASE.write(f"Found incomplete output for '{filename}'. Cleaning directory: {save_dir}")
                        shutil.rmtree(save_dir)
                    except OSError as e:
                        TQDM_BASE.write(f"Warning: Could not clean up directory {save_dir}. Error: {e}")
                # ==================================================================

        # 确保目录存在（因为可能刚刚被删除了，或者首次创建）
        os.makedirs(save_dir, exist_ok=True)
        
        try:
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
    async def parse_image_simple(self, input_path, filename, prompt_mode, save_dir, bbox=None):
        """
        重构后的图片解析方法，复用PDF的单页处理逻辑以获得健壮性。
        - 修复：_process_base64_images_with_custom_naming 的返回值类型处理
        - 新增：去重后空行清理，与 parse_pdf 行为对齐
        """
        try:
            # 1) 读取与预处理
            origin_image = fetch_image(input_path)
            processed_image = fetch_image(origin_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

            # 2) 构造与 PDF 对齐的 page_data，走统一推理与降级流程
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

            # 3) 成功路径（含降级成功）
            if status in self.SUCCESS_STATUSES:
                # 3.1 完全成功：从 cells 重新生成 MD（可过滤页眉/页脚 & 做图文匹配）
                if status == 'success':
                    original_cells = result.get('cells', []) or []
                    if original_cells:
                        # 页内匹配（图/表与标题）
                        original_cells = self._perform_intra_page_matching(original_cells)

                    # 过滤页眉/页脚（与 parse_pdf 保持一致）
                    cells_to_render = original_cells
                    if not self.keep_page_header:
                        cells_to_render = [c for c in cells_to_render if c.get('category') != 'Page-header']
                    if not self.keep_page_footer:
                        cells_to_render = [c for c in cells_to_render if c.get('category') != 'Page-footer']

                    # 生成 MD
                    md_content = layoutjson2md_full_robust(
                        origin_image,
                        cells_to_render,
                        'text',
                        no_page_hf=False
                    )
                else:
                    # 3.2 降级成功：直接使用降级阶段生成的 MD
                    md_content = result.get('md_content', '')
                    original_cells = []  # 无需再做匹配

                # 3.3 将 base64 图片落地、命名、可选过滤二维码/条码、可选跳过无标题图
                images_dir = os.path.join(save_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                loop = asyncio.get_running_loop()

                image_processing_func = functools.partial(
                    self._process_base64_images_with_custom_naming,
                    md_content,
                    images_dir,
                    1,                 # page_num 固定 1（单张图片）
                    original_cells,    # 用于命名与匹配
                    origin_image
                )
                final_md_result = await loop.run_in_executor(None, image_processing_func)
                final_md_text = final_md_result["md_content"]

                # 3.4 近似重复图片去重（仅针对 MD 中被引用的图片）
                if self.filter_duplicates:
                    final_md_text = await self._filter_duplicate_images(final_md_text, images_dir)

                # 3.5 清理冗余空行（与 parse_pdf 对齐）
                final_md_text = re.sub(r'(\n\s*){3,}', '\n\n', final_md_text)

                # 3.6 写出 MD
                md_file_path = self._get_unique_md_path(save_dir, filename)
                async with aiofiles.open(md_file_path, "w", encoding="utf-8") as md_file:
                    await md_file.write(final_md_text)

                # 回填结果并返回
                result['file_path'] = input_path
                result['output_md_path'] = md_file_path
                return [result]

            # 4) 失败路径：透传错误
            else:
                result['file_path'] = input_path
                return [result]

        except Exception as e:
            TQDM_BASE.write(f"Critical error during image preparation for {input_path}: {e}")
            return [{
                "error": str(e),
                "file_path": input_path,
                'status': 'error',
                'filename': filename,
                'page_no': 0,
                'original_page_num': 1
            }]
        
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
    parser.add_argument("--no_filter_qr_barcodes", dest='filter_qr_barcodes', action='store_false', help="Disable filtering of QR codes and barcodes. (Filtering is enabled by default)")
    parser.add_argument("--no_filter_duplicates", dest='filter_duplicates', action='store_false', help="Disable filtering of duplicate images (3 or more). (Filtering is enabled by default)")
    # --- 连接与模型参数 ---
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=900.0, help="Timeout for API requests in seconds")
    # --- 性能与并发调优参数 ---
    parser.add_argument("--file_concurrency", type=int, default=8, help="Number of files to process concurrently.")
    parser.add_argument("--page_concurrency", type=int, default=48, help="Global limit for concurrent page API requests across all files.")
    parser.add_argument("--num_cpu_workers", type=int, default=32, help="Number of CPU worker processes for page rendering. (0 for auto-detect based on CPU cores)")
    parser.add_argument("--md_gen_concurrency", type=int, default=32, help="Concurrency for the final MD generation stage. Set to 0 to auto-detect based on CPU cores. Default: 0")
    parser.add_argument("--queue_size", type=int, default=300, help="Max size of the in-memory page queue between CPU processing and API calls.")
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
    parser_args['save_page_json'] = args.save_page_json
    parser_args['save_page_layout'] = args.save_page_layout

    # 如果使用了 --rename，则强制禁用断点续传功能
    if args.rename and not args.disable_resume:
        TQDM_BASE.write("Warning: --rename is used, which is incompatible with the resume feature. Disabling resume functionality for this run.")
        parser_args['enable_resume'] = False
        parser_args['force_reprocess'] = True 
    else:
        parser_args['enable_resume'] = not args.disable_resume
    
    # 进度/统计变量
    dots_ocr_parser = None
    file_progress = None
    page_progress = None
    
    successful_files = 0
    perfectly_successful_files = 0
    processed_pages = 0
    skipped_blank_pages = 0
    failed_pages = []
    skipped_files = 0
    all_files_to_process = []
    total_pages_all_files = 0  # 注意：我们会把它改为“全量页数”

    try:
        dots_ocr_parser = DotsOCRParserOptimized(**parser_args)

        # --- Badcase 目录 ---
        if args.badcase_collection_dir:
            badcase_dir_path = Path(args.badcase_collection_dir).resolve()
            badcase_dir_path.mkdir(parents=True, exist_ok=True)
            TQDM_BASE.write(f"Badcase collection directory set to: {badcase_dir_path}")
        
        # --- 构建“全量输入文件列表” ---
        output_root = Path(args.output_dir).resolve()
        if args.input_file:
            input_path = Path(args.input_file).resolve()
            if not input_path.is_file():
                TQDM_BASE.write(f"Error: Provided input file does not exist or is not a file: {args.input_file}")
                return
            all_input_files = [input_path]
            input_root = input_path.parent
        else:
            input_root = Path(args.input_dir).resolve()
            all_input_files = []
            for ext in ['*.pdf'] + [f'*.{e.lstrip(".")}' for e in image_extensions]:
                all_input_files.extend(input_root.rglob(ext))
            all_input_files = sorted(all_input_files)

        if not all_input_files:
            if args.input_dir:
                TQDM_BASE.write(f"No supported files found in {input_root}")
            return

        processed_files = set()
        if parser_args['enable_resume'] and not args.force_reprocess:
            processed_files = get_processed_files(str(input_root), args.output_dir)
            skipped_files = len(processed_files)
            if processed_files and args.show_skipped:
                TQDM_BASE.write(f"Found {len(processed_files)} already processed files.")
        
        # --- 计算“全量页数/已完成页数”，以便 tqdm initial 起点 ---
        def _page_count_of(p: Path) -> int:
            if p.suffix.lower() == '.pdf':
                try:
                    with fitz.open(str(p)) as doc:
                        return int(doc.page_count) or 1
                except Exception:
                    return 1
            else:
                return 1

        # 全量文件/页数
        total_files_overall = len(all_input_files)
        total_pages_overall = 0
        for fp in all_input_files:
            total_pages_overall += _page_count_of(fp)

        # 已完成文件/页数
        files_done = 0
        pages_done = 0
        if processed_files:
            processed_files_norm = set(map(str, all_input_files)) & set(processed_files)
            files_done = len(processed_files_norm)
            for fp in all_input_files:
                if str(fp) in processed_files_norm:
                    pages_done += _page_count_of(fp)

        # --- 得到“剩余待处理列表” ---
        if not args.force_reprocess and parser_args['enable_resume']:
            remaining_files = [f for f in all_input_files if str(f) not in processed_files]
        else:
            remaining_files = all_input_files

        if not remaining_files:
            # 直接以 100% 展示
            file_progress = tqdm(total=total_files_overall, initial=files_done, desc="Processing files", unit="files", position=0, smoothing=0.7)
            page_progress = tqdm(total=total_pages_overall, initial=pages_done, desc="Processing pages", unit="pages", position=1, smoothing=0.7)
            TQDM_BASE.write(f"All files have been processed already ({files_done}/{total_files_overall}). Use --force_reprocess to reprocess all files.")
            return

        # --- 打印 resume 概览 ---
        TQDM_BASE.write(f"Resuming from: files {files_done}/{total_files_overall}, pages {pages_done}/{total_pages_overall}.")
        TQDM_BASE.write(f"New files to process: {len(remaining_files)} (overall {total_files_overall}).")

        await dots_ocr_parser.initialize()
        semaphore = asyncio.Semaphore(args.file_concurrency)

        # === tqdm 以“全量”为 total，以“已完成”为 initial ===
        file_progress = tqdm(
            total=total_files_overall,
            initial=files_done,
            desc="Processing files",
            unit="files",
            position=0,
            smoothing=0.7
        )
        page_progress = tqdm(
            total=total_pages_overall,
            initial=pages_done,
            desc="Processing pages",
            unit="pages",
            position=1,
            smoothing=0.7
        )

        # 为了让后续汇总更直观，这里把本轮“已处理页数计数器”起点设置为已完成页数
        processed_pages = pages_done
        total_pages_all_files = total_pages_overall

        # 性能提示
        num_workers_to_display = dots_ocr_parser.process_pool._max_workers
        TQDM_BASE.write(f"Performance settings: file_concurrency={args.file_concurrency}, page_concurrency={args.page_concurrency}, num_cpu_workers={num_workers_to_display}")
        if args.enable_table_screenshot: TQDM_BASE.write("Feature enabled: Table screenshots will be saved.")
        else: TQDM_BASE.write("Feature disabled: Table screenshots will NOT be saved. (Use --enable_table_screenshot to turn on)")
        if args.skip_uncaptioned_images: TQDM_BASE.write("Feature enabled: Images without captions will be SKIPPED.")
        else: TQDM_BASE.write("Feature disabled: All images will be included. (Use --skip_uncaptioned_images to turn on)")

        async def process_file_with_semaphore(file_path):
            nonlocal successful_files, perfectly_successful_files, processed_pages, skipped_blank_pages, failed_pages
            async with semaphore:
                relative_path = file_path.relative_to(input_root)
                target_output_dir = output_root / relative_path.parent

                # 给页级进度条一个友好的文件名提示
                if file_path.suffix.lower() == '.pdf':
                    page_progress.set_description(f"Processing pages (File: {file_path.name})")

                def page_callback():
                    nonlocal processed_pages
                    page_progress.update(1)
                    processed_pages += 1

                results = await dots_ocr_parser.parse_file(
                    str(file_path),
                    output_dir=str(target_output_dir),
                    prompt_mode=args.prompt,
                    bbox=args.bbox,
                    page_progress_callback=page_callback,
                    skip_blank_pages=args.skip_blank_pages,
                    rename_to=args.rename
                )

                if results and 'skipped_blank_pages' in results[0]:
                    skipped_blank_pages += results[0].get('skipped_blank_pages', 0)

                is_file_successful = results and all(
                    r.get('status') in dots_ocr_parser.SUCCESS_STATUSES or r.get('skipped') for r in results
                )
                if is_file_successful:
                    successful_files += 1
                    is_file_perfect = all(
                        r.get('status') in ['success', 'skipped_blank'] or r.get('skipped') for r in results
                    )
                    if is_file_perfect:
                        perfectly_successful_files += 1
                    # 非PDF文件没有页级回调，这里补 1 页（只对“新处理”的文件）
                    if file_path.suffix.lower() != '.pdf':
                        if not any(r.get('skipped') for r in results):
                            page_progress.update(1)
                            processed_pages += 1
                else:
                    TQDM_BASE.write(f"Failed to process: {file_path}")
                    if results:
                        for result in results:
                            if result.get('status') not in dots_ocr_parser.SUCCESS_STATUSES and not result.get('skipped'):
                                failed_pages.append({
                                    'file_path': str(file_path),
                                    'filename': file_path.name,
                                    'page_no': result.get('page_no', 'N/A'),
                                    'original_page_num': result.get('original_page_num', 'N/A'),
                                    'error': result.get('error', 'Unknown error'),
                                    'status': result.get('status', 'unknown')
                                })

        tasks = [process_file_with_semaphore(fp) for fp in remaining_files]
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
        print("\nProcess finished or interrupted. Shutting down resources...")
        if file_progress:
            file_progress.close()
        if page_progress:
            page_progress.close()
        if dots_ocr_parser:
            await dots_ocr_parser.shutdown()

    # --- 摘要打印（total_pages_all_files 现为全量） ---
    TQDM_BASE.write("\n" + "="*25 + "\n      Processing Summary\n" + "="*25)
    
    total_attempted_files = len(all_input_files) - (files_done if (parser_args['enable_resume'] and not args.force_reprocess) else 0)
    total_original_files = len(all_input_files)

    TQDM_BASE.write(f"Original files found: {total_original_files}")
    if parser_args['enable_resume'] and not args.force_reprocess:
        TQDM_BASE.write(f"Already processed files skipped: {files_done}")
        TQDM_BASE.write(f"New files processed / attempted: {total_attempted_files}")
    
    TQDM_BASE.write(f"Successfully processed: {successful_files}\nFailed to process: {total_attempted_files - successful_files}")

    if successful_files > 0:
        TQDM_BASE.write(f"  - Flawlessly processed: {perfectly_successful_files}")
        TQDM_BASE.write(f"  - Processed with fallbacks: {successful_files - perfectly_successful_files}")
    
    if total_attempted_files > 0:
        TQDM_BASE.write(f"Success rate (new files): {successful_files/total_attempted_files*100:.1f}%")
    if total_original_files > 0:
        TQDM_BASE.write(f"Overall completion: {successful_files + files_done}/{total_original_files} files ({(successful_files + files_done)/total_original_files*100:.1f}%)")
    
    TQDM_BASE.write(f"\nOriginal total pages found: {total_pages_all_files}")
    TQDM_BASE.write(f"Total pages processed (incl. blank): {processed_pages}")

    if args.skip_blank_pages:
        non_blank_processed = processed_pages - skipped_blank_pages
        TQDM_BASE.write(f"  - Blank pages skipped: {skipped_blank_pages}")
        TQDM_BASE.write(f"  - Non-blank pages processed: {non_blank_processed}")
        if total_pages_all_files > 0:
            TQDM_BASE.write(f"Blank page ratio: {skipped_blank_pages / total_pages_all_files * 100:.1f}%")
            TQDM_BASE.write(f"Processing completion: {processed_pages}/{total_pages_all_files} pages ({processed_pages / total_pages_all_files * 100:.1f}%)")
    elif total_pages_all_files > 0:
        TQDM_BASE.write(f"Page completion rate: {processed_pages / total_pages_all_files * 100:.1f}%")
    
    if failed_pages:
        TQDM_BASE.write(f"\n❌ Failed Pages Details ({len(failed_pages)} pages):" + "\n" + "-" * 50)
        for i, page in enumerate(failed_pages[:10]):
            TQDM_BASE.write(f"  {i+1}. File: {page['filename']}\n     Page: {page['original_page_num']}\n     Error: {page['error']}\n     Status: {page['status']}\n")
        if len(failed_pages) > 10:
            TQDM_BASE.write(f"  ... and {len(failed_pages) - 10} more failed pages")
        TQDM_BASE.write("-" * 50)
    
    if dots_ocr_parser:
        summary = dots_ocr_parser.monitor.get_summary()
        print("\n" + "="*25 + "\n     Performance Summary\n" + "="*25)
        print(f"Total processing time: {summary['total_time']:.2f}s\nTotal inference requests: {summary['total_requests']}\nAverage inference time: {summary['avg_inference_time']:.2f}s")
        if summary['total_requests'] > 0 and summary['total_time'] > 0:
            print(f"Overall throughput: {summary['total_requests'] / summary['total_time']:.2f} pages/sec")
        print(f"Total errors during inference: {summary['total_errors']}")
        if summary['total_retries'] > 0:
            print(f"Total retries performed: {summary['total_retries']}")
        if summary['error_types']:
            print("Error types breakdown:")
            [print(f"  - {error_type}: {count}") for error_type, count in summary['error_types'].items()]
    
    print("="*25 + "\n\nBatch processing finished!")

if __name__ == "__main__":
    asyncio.run(main())