import os
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

SUPPORTED_EXTS = [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".wps", ".et", ".dps"]
PDF_EXT = ".pdf"

def convert_with_libreoffice(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    format_count = defaultdict(int)
    converted_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            format_count[ext] += 1
            input_path = os.path.join(root, file)

            if ext in SUPPORTED_EXTS:
                try:
                    subprocess.run([
                        "libreoffice", "--headless", "--convert-to", "pdf",
                        "--outdir", output_dir, input_path
                    ], check=True)
                    print(f"✅ Converted: {file}")
                    converted_files.append(Path(file).stem + ".pdf")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed: {file} ({e})")
            elif ext == PDF_EXT:
                try:
                    shutil.copy2(input_path, os.path.join(output_dir, file))
                    print(f"📄 Copied PDF: {file}")
                    converted_files.append(file)
                except Exception as e:
                    print(f"❌ Failed to copy PDF: {file} ({e})")

    # 输出统计信息
    print("\n📊 格式统计：")
    for k, v in format_count.items():
        print(f"{k}: {v} 个")

    print(f"\n🎯 目标文件夹中 PDF 文件总数：{len(converted_files)}")

    # 统计目标目录下的文件夹数量
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    print(f"📁 目标文件夹内子文件夹数量：{len(subdirs)}")

    return converted_files

# 示例用法
convert_with_libreoffice(
    "/home/wangxi/workspace/gongye/yijizaojia/xiaofang",
    "/home/wangxi/workspace/gongye/yijizaojia/xiaofang_pdf"
)