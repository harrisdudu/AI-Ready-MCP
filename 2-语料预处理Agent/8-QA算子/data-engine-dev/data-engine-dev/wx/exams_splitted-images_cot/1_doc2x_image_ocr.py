# modified from songyi's code

import os
import os.path as osp
import requests
import shutil
import concurrent.futures

import re
import json


from PIL import Image, ImageOps
import io
from io import BytesIO
import base64
import zipfile

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


'''nohup python 1_doc2x_image_ocr.py /home/wangxi/workspace/huawei/source > doc2x_$(date +%m%d%H%M).log 2>&1 &'''
'''nohup python 1_doc2x_image_ocr.py /home/wangxi/workspace/huawei/source --resume > doc2x_$(date +%m%d%H%M).log 2>&1 &'''


MAX_WORKERS = 8
RATE_LIMIT = 1   # 每秒最多任务数
MAX_RETRIES = 10  # 最大重试次数
BASE_URL = "https://v2.doc2x.noedgeai.com/api/v2/parse/img/layout"
HEADERS = {'Authorization': 'Bearer sk-q0skdac8mabmobtkrlsuxdjrv4kkb9uf'}
CACHE_LOG = "temp_doc2x_cache.log"
RETRY_LOG = "retry_log.json"  # 重试日志文件


# 全局锁 + 最后一次请求时间
rate_limit_lock = threading.Lock()
last_request_time = 0.0


def load_retry_log():
    """加载重试日志"""
    if os.path.exists(RETRY_LOG):
        try:
            with open(RETRY_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_retry_log(retry_log):
    """保存重试日志"""
    with open(RETRY_LOG, "w", encoding="utf-8") as f:
        json.dump(retry_log, f, ensure_ascii=False, indent=2)


def check_existing_results(output_path):
    """
    检查已存在的处理结果，返回需要重新处理的图片列表
    
    Args:
        output_path: 输出目录路径
    
    Returns:
        list: 需要重新处理的图片信息列表
    """
    print("检查已存在的处理结果...")
    
    retry_log = load_retry_log()
    need_retry_images = []
    
    if not os.path.exists(output_path):
        print("输出目录不存在，将处理所有图片")
        return []
    
    # 遍历输出目录结构
    for competition_dir in os.listdir(output_path):
        competition_path = osp.join(output_path, competition_dir)
        if not osp.isdir(competition_path):
            continue
            
        for sub_dir in os.listdir(competition_path):
            sub_path = osp.join(competition_path, sub_dir)
            if not osp.isdir(sub_path):
                continue
                
            # 检查题干、答案、解析目录
            for content_dir in ["题干", "答案", "解析"]:
                content_path = osp.join(sub_path, content_dir)
                if not osp.exists(content_path):
                    continue
                    
                # 检查每个图片的输出文件夹
                for img_folder in os.listdir(content_path):
                    img_folder_path = osp.join(content_path, img_folder)
                    if not osp.isdir(img_folder_path):
                        continue
                    
                    # 检查是否存在md文件
                    md_file = osp.join(img_folder_path, f"{img_folder}.md")
                    if not os.path.exists(md_file):
                        # 检查重试次数
                        retry_key = f"{competition_dir}_{sub_dir}_{content_dir}_{img_folder}"
                        retry_count = retry_log.get(retry_key, 0)
                        
                        if retry_count < MAX_RETRIES:
                            # 需要重新处理
                            need_retry_images.append({
                                'competition_dir': competition_dir,
                                'sub_dir': sub_dir,
                                'content_dir': content_dir,
                                'img_folder': img_folder,
                                'retry_count': retry_count,
                                'img_path': osp.join(img_folder_path, "original_image")  # 这里需要从原始路径获取
                            })
                        else:
                            print(f"跳过 {retry_key}: 已达到最大重试次数 ({MAX_RETRIES})")
    
    print(f"发现 {len(need_retry_images)} 张图片需要重新处理")
    return need_retry_images


def find_original_image_path(input_path, competition_dir, sub_dir, content_dir, img_folder):
    """
    根据输出目录结构找到对应的原始图片路径
    
    Args:
        input_path: 输入目录路径
        competition_dir: 竞赛目录名
        sub_dir: 子目录名
        content_dir: 内容目录名（题干/答案/解析）
        img_folder: 图片文件夹名
    
    Returns:
        str: 原始图片路径，如果找不到返回None
    """
    # 尝试常见的图片扩展名
    img_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for ext in img_extensions:
        img_file = img_folder + ext
        img_path = osp.join(input_path, competition_dir, sub_dir, content_dir, img_file)
        if os.path.exists(img_path):
            return img_path
    
    return None


def update_retry_count(competition_dir, sub_dir, content_dir, img_folder, success=True):
    """
    更新重试次数
    
    Args:
        competition_dir: 竞赛目录名
        sub_dir: 子目录名
        content_dir: 内容目录名
        img_folder: 图片文件夹名
        success: 是否处理成功
    """
    retry_log = load_retry_log()
    retry_key = f"{competition_dir}_{sub_dir}_{content_dir}_{img_folder}"
    
    if success:
        # 处理成功，从重试日志中移除
        if retry_key in retry_log:
            del retry_log[retry_key]
    else:
        # 处理失败，增加重试次数
        retry_log[retry_key] = retry_log.get(retry_key, 0) + 1
    
    save_retry_log(retry_log)


def unzip_base64(zip_str, target_dir):
    if zip_str == "": return []
    # 1. Base64 解码成二进制
    zip_bytes = base64.b64decode(zip_str)
    # 2. 用 zipfile 打开
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        # 获取zip文件中的文件列表
        file_list = zf.namelist()
        
        print(f"  ZIP文件内容: {file_list}")
        print(f"  目标目录: {target_dir}")
        
        # 解压文件，但避免创建重复的目录结构
        extracted_files = []
        for file_name in file_list:
            # 读取文件内容
            with zf.open(file_name) as source_file:
                # 获取文件名（去掉可能的路径前缀）
                base_name = osp.basename(file_name)
                # 直接写入目标目录
                target_file_path = osp.join(target_dir, base_name)
                print(f"    解压: {file_name} -> {target_file_path}")
                with open(target_file_path, 'wb') as target_file:
                    shutil.copyfileobj(source_file, target_file)
                extracted_files.append(base_name)
        
        return extracted_files

def log_cache(cache):
    """将失败的 item 及原因写入日志"""
    with open(CACHE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(cache, ensure_ascii=False) + "\n")

#TODO: 自动padding和自动重试，不是手动重试

def add_padding(image_path, padding=50, color=(255, 255, 255)):
    """给图片添加 padding 并返回字节数据"""
    img = Image.open(image_path)
    padded_img = ImageOps.expand(img, border=padding, fill=color)
    img_byte_arr = io.BytesIO()
    padded_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

# def add_padding(image_path, padding=50, color=(255, 255, 255)):
#     """将图片添加到640×640的白色画布上，并返回字节数据"""
#     # 创建640×640的白色画布
#     canvas_size = (640, 640)
#     canvas = Image.new('RGB', canvas_size, color)
    
#     # 打开原图片
#     img = Image.open(image_path)
    
#     # 计算图片在画布上的居中位置
#     img_width, img_height = img.size
#     x_offset = (canvas_size[0] - img_width) // 2
#     y_offset = (canvas_size[1] - img_height) // 2
    
#     # 将图片粘贴到画布中心
#     canvas.paste(img, (x_offset, y_offset))
    
#     # 转换为字节数据
#     img_byte_arr = io.BytesIO()
#     canvas.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)
#     return img_byte_arr.getvalue()

def safe_request(image_bytes):
    """带全局锁的限速请求"""
    global last_request_time
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < 1.0 / RATE_LIMIT:
            time.sleep((1.0 / RATE_LIMIT) - elapsed)
        last_request_time = time.time()
    response = requests.post(BASE_URL, headers=HEADERS, data=image_bytes)
    return response

def clean_md(md):
    """清理 markdown 数学公式"""
    md = md.replace(r"\( ", "$").replace(r" \)", "$")
    md = md.replace(r"\(", "$").replace(r"\)", "$")
    md = md.replace(r"\[", "$$").replace(r"\]", "$$")
    return md

def clean_ocr_text(text):
    """清理OCR结果中的特殊标记和无关内容"""
    if not text:
        return text
        
    # 清理所有HTML/XML注释 <!-- ... -->
    text = re.sub(r'<!--[^>]*-->', '', text)
    
    # 清理所有注释标记 <!-- ... --> (包括多行注释)
    text = re.sub(r'<!--[\s\S]*?-->', '', text)
    
    # 清理多余的空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # 清理行首行尾的空白字符
    text = text.strip()
    
    return text

def process_single_image(item_idx, img_type, img_path, target_images_dir):
    """
    处理单张图片并返回结果
    img_type: "q" 表示 question 图像, "a" 表示 answer 图像
    target_images_dir: 目标images目录路径
    """
    try:
        resp = safe_request(add_padding(img_path))
        if resp.status_code == 200:
            resp_dict = json.loads(resp.text)
            md_text = resp_dict["data"]["result"]["pages"][0]["md"]
            zip_str = resp_dict["data"]["convert_zip"]
            # 直接解压到目标images目录
            imgs = unzip_base64(zip_str, target_images_dir)
            md = clean_md(clean_ocr_text(md_text))

            if md == "":
                log_cache({"index": item_idx, "type": img_type, "error": "ERROR: 未识别到内容"})
                return (item_idx, img_type, None, imgs)
            else:
                log_cache({"index": item_idx, "type": img_type, "md": md, "image_paths": imgs})
                return (item_idx, img_type, md, imgs)
        else:
            log_cache({"index": item_idx, "type": img_type, "error": f"ERROR: 状态码 {resp.status_code}"})
            return (item_idx, img_type, None, None)
    except Exception as e:
        log_cache({"index": item_idx, "type": img_type, "error": f"ERROR: {e}"})
        return (item_idx, img_type, None, None)

def process_directory(input_path, output_path, resume_mode=False):
    """
    处理指定目录下的所有图片文件
    
    Args:
        input_path: 输入目录路径
        output_path: 输出目录路径
        resume_mode: 是否为断点续传模式
    
    Returns:
        dict: 处理结果统计
    """
    results = {
        "total_directories": 0,
        "total_images": 0,
        "successful_ocr": 0,
        "failed_ocr": 0,
        "processing_time": 0,
        "resume_mode": resume_mode,
        "results": {}
    }
    
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 如果是断点续传模式，检查已存在的结果
    if resume_mode:
        need_retry_images = check_existing_results(output_path)
        if not need_retry_images:
            print("所有图片都已处理完成，无需断点续传")
            return results
    else:
        need_retry_images = []
    
    # 首先统计总图片数量
    print("正在扫描目录结构...")
    total_images_count = 0
    image_info_list = []
    
    for competition_dir in os.listdir(input_path):
        competition_path = osp.join(input_path, competition_dir)
        if not osp.isdir(competition_path):
            continue
            
        results["total_directories"] += 1
        results["results"][competition_dir] = {
            "sub_directories": {},
            "total_images": 0,
            "successful_ocr": 0,
            "failed_ocr": 0
        }
        
        # 遍历竞赛下的子目录（如：联考题（热光近卷））
        for sub_dir in os.listdir(competition_path):
            sub_path = osp.join(competition_path, sub_dir)
            if not osp.isdir(sub_path):
                continue
                
            results["results"][competition_dir]["sub_directories"][sub_dir] = {
                "question_images": {},
                "answer_images": {},
                "analysis_images": {},
                "total_images": 0,
                "successful_ocr": 0,
                "failed_ocr": 0
            }
            
            # 处理题干、答案、解析目录
            for content_dir in ["题干", "答案", "解析"]:
                content_path = osp.join(sub_path, content_dir)
                if not osp.exists(content_path):
                    continue
                    
                # 获取该目录下的所有图片
                image_files = [f for f in os.listdir(content_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    continue
                
                # 记录图片信息
                for img_file in image_files:
                    # 如果是断点续传模式，检查是否需要处理这张图片
                    if resume_mode:
                        img_name_without_ext = osp.splitext(img_file)[0]
                        img_output_dir = osp.join(output_path, competition_dir, sub_dir, content_dir, img_name_without_ext)
                        md_file = osp.join(img_output_dir, f"{img_name_without_ext}.md")
                        
                        # 如果md文件已存在且不在重试列表中，跳过
                        if os.path.exists(md_file):
                            retry_key = f"{competition_dir}_{sub_dir}_{content_dir}_{img_name_without_ext}"
                            if not any(img['img_folder'] == img_name_without_ext and 
                                     img['competition_dir'] == competition_dir and
                                     img['sub_dir'] == sub_dir and
                                     img['content_dir'] == content_dir 
                                     for img in need_retry_images):
                                continue
                    
                    total_images_count += 1
                    image_info_list.append({
                        'competition_dir': competition_dir,
                        'sub_dir': sub_dir,
                        'content_dir': content_dir,
                        'img_file': img_file,
                        'img_path': osp.join(content_path, img_file)
                    })
    
    print(f"扫描完成！共发现 {total_images_count} 张图片需要处理")
    print("开始处理图片OCR...")
    
    # 使用tqdm显示总体进度
    with tqdm(total=total_images_count, desc="OCR处理进度", unit="张") as pbar:
        # 处理图片
        for img_info in image_info_list:
            competition_dir = img_info['competition_dir']
            sub_dir = img_info['sub_dir']
            content_dir = img_info['content_dir']
            img_file = img_info['img_file']
            img_path = img_info['img_path']
            
            # 更新进度条描述
            pbar.set_description(f"处理: {competition_dir}/{sub_dir}/{content_dir}/{img_file}")
            
            results["total_images"] += 1
            results["results"][competition_dir]["total_images"] += 1
            results["results"][competition_dir]["sub_directories"][sub_dir]["total_images"] += 1
            
            # 确定图片类型
            img_type = "question" if content_dir == "题干" else "answer" if content_dir == "答案" else "analysis"
            
            # 为每张图片创建独立的文件夹
            img_name_without_ext = osp.splitext(img_file)[0]
            img_output_dir = osp.join(output_path, competition_dir, sub_dir, content_dir, img_name_without_ext)
            os.makedirs(img_output_dir, exist_ok=True)
            
            # 创建images子文件夹
            images_dir = osp.join(img_output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # 处理图片OCR
            item_idx = f"{competition_dir}_{sub_dir}_{content_dir}_{img_file}"
            result = process_single_image(item_idx, img_type, img_path, images_dir)
            
            if result[2] is not None:  # OCR成功
                results["successful_ocr"] += 1
                results["results"][competition_dir]["successful_ocr"] += 1
                results["results"][competition_dir]["sub_directories"][sub_dir]["successful_ocr"] += 1
                
                # 保存OCR结果到md文件
                md_file = osp.join(img_output_dir, f"{img_name_without_ext}.md")
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(result[2])
                
                # 更新重试计数（成功）
                update_retry_count(competition_dir, sub_dir, content_dir, img_name_without_ext, success=True)
                
                # 图片已经直接解压到images文件夹，记录信息
                img_info = {
                    "original_image": img_file,
                    "ocr_text": result[2],
                    "converted_images": result[3] if result[3] else [],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if img_type == "question":
                    results["results"][competition_dir]["sub_directories"][sub_dir]["question_images"][img_file] = img_info
                elif img_type == "answer":
                    results["results"][competition_dir]["sub_directories"][sub_dir]["answer_images"][img_file] = img_info
                else:
                    results["results"][competition_dir]["sub_directories"][sub_dir]["analysis_images"][img_file] = img_info
                    
            else:  # OCR失败
                results["failed_ocr"] += 1
                results["results"][competition_dir]["failed_ocr"] += 1
                results["results"][competition_dir]["sub_directories"][sub_dir]["failed_ocr"] += 1
                
                # 更新重试计数（失败）
                update_retry_count(competition_dir, sub_dir, content_dir, img_name_without_ext, success=False)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                '成功': results["successful_ocr"],
                '失败': results["failed_ocr"],
                '成功率': f"{results['successful_ocr']/results['total_images']*100:.1f}%"
            })
    
    results["processing_time"] = time.time() - start_time
    
    # 保存处理结果摘要
    summary_file = osp.join(output_path, "processing_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


def main():
    """
    主函数：处理指定输入路径下的所有图片文件
    用法: python 1_doc2x_image_ocr.py <输入路径> [--resume]
    """
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python 1_doc2x_image_ocr.py <输入路径> [--resume]")
        print("示例: python 1_doc2x_image_ocr.py /path/to/source")
        print("示例: python 1_doc2x_image_ocr.py /path/to/source --resume")
        sys.exit(1)
    
    # 输入路径作为命令行参数
    input_path = sys.argv[1]
    
    # 检查是否为断点续传模式
    resume_mode = len(sys.argv) == 3 and sys.argv[2] == "--resume"
    
    # 输出路径自动生成为: 输入路径 + "output"
    output_path = input_path + "_output"
    
    if resume_mode:
        print(f"断点续传模式：检查并重新处理失败的图片...")
    else:
        print(f"开始处理图片OCR...")
    
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"最大重试次数: {MAX_RETRIES}")
    
    try:
        results = process_directory(input_path, output_path, resume_mode)
        
        print("\n" + "="*50)
        print("处理完成！结果摘要：")
        print("="*50)
        print(f"处理模式: {'断点续传' if resume_mode else '完整处理'}")
        print(f"总目录数: {results['total_directories']}")
        print(f"总图片数: {results['total_images']}")
        print(f"OCR成功: {results['successful_ocr']}")
        print(f"OCR失败: {results['failed_ocr']}")
        if results['total_images'] > 0:
            print(f"成功率: {results['successful_ocr']/results['total_images']*100:.2f}%")
        print(f"处理时间: {results['processing_time']:.2f}秒")
        print(f"详细结果已保存到: {output_path}/processing_summary.json")
        print(f"重试日志已保存到: {RETRY_LOG}")
        
        # 显示各竞赛的处理结果
        if results["results"]:
            print("\n各竞赛处理结果:")
            for comp_name, comp_result in results["results"].items():
                print(f"\n{comp_name}:")
                print(f"  总图片: {comp_result['total_images']}, 成功: {comp_result['successful_ocr']}, 失败: {comp_result['failed_ocr']}")
                
                for sub_name, sub_result in comp_result["sub_directories"].items():
                    print(f"    {sub_name}: 图片数 {sub_result['total_images']}, 成功 {sub_result['successful_ocr']}, 失败 {sub_result['failed_ocr']}")
        
        # 显示重试统计
        retry_log = load_retry_log()
        if retry_log:
            print(f"\n重试统计:")
            print(f"仍有 {len(retry_log)} 张图片需要重试")
            print(f"建议使用 --resume 参数继续处理")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()