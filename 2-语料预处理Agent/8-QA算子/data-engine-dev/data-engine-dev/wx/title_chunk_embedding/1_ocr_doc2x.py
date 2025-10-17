import os
import zipfile
import time
import shutil
from tqdm import tqdm
from datetime import datetime
from pypdf import PdfReader, PdfWriter
from pdfdeal import Doc2X

# === 配置 ===
API_KEY = "sk-q0skdac8mabmobtkrlsuxdjrv4kkb9uf"
INPUT_DIR = "/home/wangxi/workspace/xiaofang/original"
OUTPUT_DIR = "/home/wangxi/workspace/xiaofang/doc2x_ocred"
MAX_RETRIES = 5
RETRY_DELAY = 10
MAX_FILE_SIZE_MB = 300  # 超过该大小也要拆分
TO_CONSOLE = True

# === 初始化日志 ===
LOG_FILE = os.path.join(OUTPUT_DIR, "doc2x_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n\n========== 运行开始：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========\n")

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")
    if TO_CONSOLE:
        print(full_msg)

# === 初始化 Doc2X 客户端 ===
try:
    client = Doc2X(apikey=API_KEY, debug=True)
except Exception as e:
    log(f"❌ 初始化 Doc2X 客户端失败：{e}")
    exit(1)

# === 工具函数 ===
def get_pdf_page_count(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        log(f"❌ 获取页数失败: {pdf_path} - {e}")
        return -1

def get_file_size_mb(file_path):
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception as e:
        log(f"❌ 获取文件大小失败: {file_path} - {e}")
        return -1

def split_pdf(pdf_path, output_dir):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    file_size_mb = get_file_size_mb(pdf_path)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    split_paths = []

    # 计算拆分份数（页数优先）
    if total_pages > 1000:
        num_parts = (total_pages + 999) // 1000
    elif file_size_mb > MAX_FILE_SIZE_MB:
        # 如果仅是大小超过限制，按平均大小拆分
        # 假设每页大小近似，按页数平均拆
        estimated_pages_per_part = max(1, int(total_pages * (MAX_FILE_SIZE_MB / file_size_mb)))
        num_parts = (total_pages + estimated_pages_per_part - 1) // estimated_pages_per_part
    else:
        num_parts = 1  # 不拆分

    pages_per_part = (total_pages + num_parts - 1) // num_parts

    for i in range(num_parts):
        writer = PdfWriter()
        start_page = i * pages_per_part
        end_page = min(start_page + pages_per_part, total_pages)

        for j in range(start_page, end_page):
            writer.add_page(reader.pages[j])

        split_name = f"{stem}_part{i+1}.pdf"
        split_path = os.path.join(output_dir, split_name)
        with open(split_path, "wb") as f:
            writer.write(f)
        split_paths.append(split_path)
        log(f"📄 拆分生成: {split_path}（页码 {start_page+1}-{end_page}）")

    return split_paths

def merge_md_parts(md_dir, stem):
    part_md_files = sorted([
        os.path.join(md_dir, f) for f in os.listdir(md_dir)
        if f.startswith(f"{stem}_part") and f.endswith(".md")
    ])
    output_md_path = os.path.join(md_dir, f"{stem}.md")
    
    # 检查是否已经合并过
    if os.path.exists(output_md_path) and os.path.getsize(output_md_path) > 0:
        log(f"⏩ 已存在合并后的 Markdown，跳过合并: {output_md_path}")
        return
    
    if not part_md_files:
        log(f"⚠️ 没有找到需要合并的拆分文件: {stem}")
        return
    
    with open(output_md_path, "w", encoding="utf-8") as out_f:
        for md_file in part_md_files:
            with open(md_file, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())
                out_f.write("\n\n")
    log(f"📘 合并 Markdown 完成: {output_md_path}")

    # 🔥 删除合并前的 part md 文件
    for part_md in part_md_files:
        try:
            os.remove(part_md)
            log(f"🗑️ 删除拆分 Markdown: {part_md}")
        except Exception as e:
            log(f"⚠️ 删除失败: {part_md} - {e}")

def process_pdf_with_retry(pdf_file_path, output_dir, retry_count=0):
    stem = os.path.splitext(os.path.basename(pdf_file_path))[0]
    expected_zip_path = os.path.join(output_dir, stem + ".zip")

    try:
        client.pdf2file(pdf_file=pdf_file_path, output_path=output_dir, output_format="md_dollar")
        if not os.path.exists(expected_zip_path):
            raise Exception("未找到 ZIP 输出文件")
        log(f"✅ 转换成功: {pdf_file_path}")
        return True
    except Exception as e:
        if retry_count < MAX_RETRIES:
            log(f"⚠️ 第 {retry_count+1} 次重试: {pdf_file_path} - 错误: {str(e)}")
            time.sleep(RETRY_DELAY)
            return process_pdf_with_retry(pdf_file_path, output_dir, retry_count + 1)
        else:
            log(f"❌ 达最大重试次数: {pdf_file_path} - 错误: {str(e)}")
            return False

def handle_zip_and_md(output_dir, stem):
    zip_path = os.path.join(output_dir, stem + ".zip")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        output_md = os.path.join(output_dir, "output.md")
        final_md = os.path.join(output_dir, stem + ".md")
        if os.path.exists(output_md):
            os.rename(output_md, final_md)
            log(f"✅ Markdown 重命名: {output_md} → {final_md}")
        else:
            log(f"⚠️ 解压后未找到 output.md")
        os.remove(zip_path)
        log(f"🗑️ 删除 ZIP: {zip_path}")
    except Exception as e:
        log(f"❌ ZIP 解压失败: {zip_path} - {e}")

def process_single_pdf(pdf_path, output_root_dir):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_root_dir, stem)
    final_md_path = os.path.join(pdf_output_dir, f"{stem}.md")

    # === 断点续传判断 ===
    if os.path.exists(final_md_path) and os.path.getsize(final_md_path) > 0:
        log(f"⏩ 已存在完整 Markdown，跳过: {pdf_path}")
        return
    elif os.path.exists(pdf_output_dir):
        # 如果文件夹存在但没有完整的md文件，清理残留文件
        log(f"🧹 清理残留文件: {pdf_output_dir}")
        try:
            shutil.rmtree(pdf_output_dir)
        except Exception as e:
            log(f"⚠️ 清理失败: {pdf_output_dir} - {e}")

    os.makedirs(pdf_output_dir, exist_ok=True)

    page_count = get_pdf_page_count(pdf_path)
    file_size_mb = get_file_size_mb(pdf_path)
    if page_count == -1 or file_size_mb == -1:
        log(f"❌ 跳过（获取信息失败）: {pdf_path}")
        return

    need_split = page_count > 1000 or file_size_mb >= MAX_FILE_SIZE_MB

    if not need_split:
        if process_pdf_with_retry(pdf_path, pdf_output_dir):
            handle_zip_and_md(pdf_output_dir, stem)
    else:
        log(f"⚠️ PDF 需拆分: {pdf_path} （{page_count} 页，{file_size_mb:.2f} MB）")
        
        # 检查是否所有拆分部分都已处理完成
        all_parts_completed = True
        split_paths = split_pdf(pdf_path, pdf_output_dir)
        
        for part_path in split_paths:
            part_stem = os.path.splitext(os.path.basename(part_path))[0]
            part_md_path = os.path.join(pdf_output_dir, f"{part_stem}.md")
            
            if not os.path.exists(part_md_path) or os.path.getsize(part_md_path) == 0:
                all_parts_completed = False
                if process_pdf_with_retry(part_path, pdf_output_dir):
                    handle_zip_and_md(pdf_output_dir, part_stem)
            else:
                log(f"⏩ 拆分部分已存在: {part_md_path}")
        
        # 只有当所有部分都完成时才合并
        if all_parts_completed or os.path.exists(final_md_path):
            merge_md_parts(pdf_output_dir, stem)
            for p in split_paths:
                os.remove(p)
            log(f"🧹 拆分 PDF 清理完成")

def run_processing():
    log(f"📂 扫描目录: {INPUT_DIR}")
    for root, _, files in os.walk(INPUT_DIR):
        rel_path = os.path.relpath(root, INPUT_DIR)
        current_output_dir = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)

        for file in tqdm(files, desc=f"📁 {rel_path}", unit="file"):
            source_path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                process_single_pdf(source_path, current_output_dir)
            else:
                dst_path = os.path.join(current_output_dir, file)
                try:
                    shutil.copy2(source_path, dst_path)
                    log(f"✅ 复制非PDF文件: {source_path}")
                except Exception as e:
                    log(f"❌ 复制失败: {source_path} - {e}")

    log("🎉 所有文件处理完成")

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        log(f"❌ 输入目录无效: {INPUT_DIR}")
    else:
        run_processing()