# coding:utf-8
import os
from pdf2image import convert_from_path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_need_full_files(start_path, file_extensions):
    """
    查找指定目录及其子目录中符合扩展名的文件（返回全路径列表）
    """
    matched_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 过滤临时文件
            if '~$' in os.path.basename(file_path) or '.~' in os.path.basename(file_path):
                continue
            if any(file.lower().endswith(ext) for ext in file_extensions):
                matched_files.append(file_path)
    return matched_files

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 取消像素限制，防止 DecompressionBombError

def save_pdf_pages_as_images(pdf_path, output_dir, num_pages=2, image_format='png', dpi=150):
    """
    将 PDF 文件的前 N 页保存为图片，并自动 resize 到合理尺寸，防止过大图像。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=num_pages, dpi=dpi)
    except Exception as e:
        print(f"❌ 转换失败: {pdf_path}, 错误: {e}")
        return []

    saved_files = []
    for i, image in enumerate(images):
        max_size = (2000, 2000)
        image.thumbnail(max_size, Image.LANCZOS)

        image_filename = os.path.join(output_dir, f'page_{i+1}.{image_format}')
        try:
            image.save(image_filename, image_format.upper())
            saved_files.append(image_filename)
        except Exception as save_error:
            print(f"❌ 保存图片失败: {image_filename}, 错误: {save_error}")

    if saved_files:
        print(f"✅ {pdf_path}：成功保存 {len(saved_files)} 张图片到 '{output_dir}'")
    else:
        print(f"⚠️ {pdf_path}：未保存图片 (可能 PDF 页数不足 {num_pages} 页或转换无内容)")
    return saved_files

def process_pdf_task(pdf_path, source_root, target_root, num_pages=2):
    """
    处理单个 PDF 文件任务，构建输出路径并调用保存函数。
    输出路径格式：target_root + 相对路径（包含“消防各处室资料0801”）+ pdf文件名(不含扩展名)
    """
    try:
        # 使用 source_root 的父目录作为逻辑根目录
        logical_root = os.path.dirname(source_root) 
        relative_path = os.path.relpath(pdf_path, logical_root)
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        relative_dir = os.path.dirname(relative_path)
        output_dir = os.path.join(target_root, relative_dir, pdf_stem)
        return save_pdf_pages_as_images(pdf_path, output_dir, num_pages=num_pages)
    except Exception as e:
        print(f"❌ 处理任务时出错 {pdf_path}: {e}")
        return []

if __name__ == '__main__':
    # ==================== 配置区 ====================
    source_root = "/mnt/data/zzj/data_clean_fire/data/middle_pdf/消防各处室资料0801/03.作战训练处"
    # source_root = "/mnt/data/zzj/data_clean_fire/data/middle_pdf/消防各处室资料0801/03.作战训练处"
    target_root = "/mnt/data/zzj/data_clean_fire/data/extract_img/消防各处室资料0801"

    file_extensions = [".pdf"]
    debug = False
    num_threads = 128
    NUM_PAGES_TO_EXTRACT = 3
    # ===============================================

    found_files = find_need_full_files(source_root, file_extensions)
    print(f"[准备] 共找到 {len(found_files)} 个 PDF 文件。")

    if debug:
        print("⚠️ 调试模式开启（单线程）")
        for pdf_path in tqdm(found_files, desc="Processing PDFs (Debug)"):
            process_pdf_task(pdf_path, source_root, target_root, num_pages=NUM_PAGES_TO_EXTRACT)
    else:
        print(f"🚀 多线程处理开始 (线程数: {num_threads}, 提取页数: {NUM_PAGES_TO_EXTRACT})")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_pdf_task, pdf_path, source_root, target_root, NUM_PAGES_TO_EXTRACT): pdf_path
                for pdf_path in found_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                pdf_path = futures[future]
                try:
                    saved_image_paths = future.result()
                except Exception as e:
                    print(f"❌ 异常处理文件: {pdf_path}，错误: {e}")

    print("[完成] PDF 图片提取任务结束。")