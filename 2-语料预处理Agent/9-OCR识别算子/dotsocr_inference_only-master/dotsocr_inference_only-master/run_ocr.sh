#!/bin/bash

# ==============================================================================
#  说明
# ==============================================================================
#
#  这是一个用于运行 Dots.OCR Python 解析器的“一键启动”脚本。
#  所有的常用配置（Python脚本路径、输入输出目录、端口等）都在脚本顶部的
#  “配置区”进行修改。
#
#  用法:
#  1. 直接在下面的“配置区”修改好 INPUT_DIR 和 OUTPUT_DIR 的值。
#  2. 在终端中直接运行 ./run_ocr.sh 即可。注意：chmod +x .run_ocr.sh
#
#  临时覆盖参数:
#  您仍然可以在运行时临时添加参数来覆盖默认配置，例如：
#  ./run_ocr.sh --port 9999 --prompt prompt_layout_all_cn
#
# ==============================================================================

# --- 配置区 ---

# 1. Python 脚本的绝对路径
PYTHON_SCRIPT_PATH="./parser_async_v12.py"

# 2. 【请修改】默认的输入目录
# INPUT_DIR="/home/liujunyi/workspace/检察院/PDF"
INPUT_DIR="./demo_pdf_test"

# INPUT_FILE="/home/liujunyi/workspace/OCR-VLM/dots.ocr/PDF_test/17422-普陀区“15分钟社区生活圈”行动中期评估-正文.pdf"

# 3. 【请修改】默认的输出目录
# OUTPUT_DIR="/home/liujunyi/workspace/检察院/MD"
# OUTPUT_DIR="/home/liujunyi/workspace/OCR-VLM/dots.ocr/output_single_test"
OUTPUT_DIR="./demo_pdf_test_output"

# 4. 【可选】API key for authentication
API_KEY="CdLHpwl3wwjGnWKYUjxybNA0jzYYguysk1gobChTkFo="

# 5. 【可选】如需要用到autopipeline做后续QA生成的，设置为True
ADD_PAGE_TAG=False

# 检查脚本文件是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: Python 脚本未在指定路径找到: '$PYTHON_SCRIPT_PATH'"
    echo "请打开此 shell 脚本并修改 'PYTHON_SCRIPT_PATH' 变量。"
    exit 1
fi

# 检查输入输出目录是否已经配置
if [[ "$INPUT_DIR" == "/path/to/your/default_input_folder" || "$OUTPUT_DIR" == "/path/to/your/default_output_folder" ]]; then
    echo "警告: 输入或输出目录似乎尚未配置。"
    echo "请打开脚本文件修改 INPUT_DIR 和 OUTPUT_DIR 变量。"
    # 等待5秒，给用户一个取消的机会 (按 Ctrl+C)
    echo "将在 5 秒后继续运行..."
    sleep 5
fi


# 将所有从命令行传入的参数作为额外参数
# 这允许您在需要时临时覆盖默认设置
ADDITIONAL_ARGS="$@"
# 如果 ADD_PAGE_TAG 为 True，则添加 --add_page_tag 参数
if [ "$ADD_PAGE_TAG" = "True" ]; then
    echo "将为每页末尾加上<special_page_num_tag>标签"
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --add_page_tag True"
fi

# 打印将要执行的命令，方便调试
echo "================================================="
echo " OCR 任务启动"
echo "================================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "默认参数:  --ip 106.75.235.212 --port 12000 --skip_blank_pages"
echo "其他参数: $ADDITIONAL_ARGS"
if [ ! -z "$API_KEY" ]; then
    echo "API Key: 已设置"
fi
echo "-------------------------------------------------"
echo "执行命令:"
set -x

python "$PYTHON_SCRIPT_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ip 106.75.235.212 \
    --port 12000 \
    --num_cpu_workers 10 \
    --page_concurrency 80 \
    --skip_blank_pages \
    --badcase_collection_dir "/Users/junyi/Downloads" \
    --save_page_json    \
    ${API_KEY:+--api_key "$API_KEY"} \
    $ADDITIONAL_ARGS

# 关闭命令打印
set +x
echo "================================================="
echo " 任务执行完毕"
echo "================================================="