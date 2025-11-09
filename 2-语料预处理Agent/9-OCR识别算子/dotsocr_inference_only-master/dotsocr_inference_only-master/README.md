# Dots.OCR Parser v12

Dots.OCR 是面向长文档的多模态解析流水线，聚焦“稳定可控的结构化输出”。当前主干脚本为 `parser_async_v12.py`，在继承 v7–v11 的布局匹配、跨页文本合并、断点续传等能力的基础上，引入三阶段流式处理、format_transformer_v3 Markdown 管线，以及默认开启的媒体资产治理。

## 启动脚本与配套服务
- `run_ocr.sh`：通用环境一键入口，内部指向 `parser_async_v12.py`，可在脚本顶部调整输入/输出目录、API Key 以及 `ADD_PAGE_TAG`。
- `run_ocr_a100.sh`：面向 8 卡 A100 的批量处理脚本，需要预先配置 API Key。
- `run_ocr_a100_json.sh`：输出分页中间产物和布局截图，适合深度调试（保留旧版框架和后处理逻辑，使用时请结合实际需求）。
- `start_vllm_servers_serial.sh` / `start_vllm_servers.sh`：vLLM 推理服务启动器（串行版本稳健，並行版本不建议 6 卡以上同时拉起）。

## 版本亮点

### v12 新增与优化
- **表格二次解析 (Table Reparsing)**: 新增 `--enable_table_reparse` 参数，当启用时，系统会利用 `paddleocr` 对模型输出的表格进行二次解析。此功能旨在修正和优化表格结构，提高复杂表格的识别准确率。
- **脚注跳过**: 通过 `--skip_footnote` 参数，可以跳过文档中的脚注内容，专注于正文提取。
- **首页摘要裁剪**: `--trim_first_page_summary` 参数用于移除首页可能存在的摘要或元信息，使文档内容更纯净。
- **关键词过滤**: `--keyword_filter_config` 允许用户提供一个 JSON 配置文件，用于过滤包含特定关键词的页面。
- **上标标准化**: `--normalize_superscript` 参数可以将不规范的上标转换为标准`<sup></sup>`格式，统一输出格式。

### v11 新增与优化
- **三阶段流式解析**：PASS-1 并发渲染 + 异步推理，PASS-2 在独立进程完成跨页匹配/合并，PASS-3 采用窗口化顺序落盘，内存与吞吐更均衡。
- **format_transformer_v3 Markdown 管线**：`layoutjson2md_full_robust` 修复 Section-header 标记、规范 LaTeX（`get_formula_in_markdown`、`fix_streamlit_formulas`）、校正 bbox 并与解析器命名逻辑一致。
- **媒体资产治理**：默认过滤二维码/条码（`--no_filter_qr_barcodes` 可关闭）、基于感知哈希的重复图片簇清理（`--no_filter_duplicates` 可关闭），表格截图遵循标题命名。
- **输出友好性**：`--add_page_tag True` 在每页末尾插入 `<special_page_num_tag>`，便于后续 QA/检索流水线；结尾提供性能概要（吞吐、重试、错误类型）。
- **健壮性增强**：竞速重试（`--concurrent_retries`）、改进的断点续传目录映射、`console_write` 与 tqdm 协同的运行概览、`--verbose` / `--save_page_json` / `--save_page_layout` 调试工具。

### 历史能力传承
- **跨页表格“纠错释放”机制**：三阶段匹配防止错误占用标题，并在跨页校验后退回错误关联，提高复杂跨页表格的准确率。
- **页内匹配鲁棒性**：针对内嵌标题、高瘦元素等场景重写匹配逻辑，结合阅读顺序与空间距离提高命中率。
- **跨页文本合并**：`NEW_PARAGRAPH_PATTERN`、TOC 识别与句末判断协同工作，穿透装饰性页眉并在真实章节标题处停下，防止串章节合并。
- **断点续传与资源管理**：递归扫描输出目录匹配历史 `.md`，减少重复工作；`ProcessPoolExecutor` 分离 PDF 渲染与推理，配合全局信号量保护推理端。
- **调试开关**：保留 `--save_page_json`、`--save_page_layout`、`--enable_table_screenshot`、`--skip_uncaptioned_images`、`--keep_page_header/footer`、`--badcase_collection_dir` 等一系列开关。

## 核心功能详解

### 版面分析与后处理
- 跨页表格聚类 + 标题纠错释放，确保表格与标题一一对应。
- 页内图文匹配支持标题位于边界框内部或上下混排的复杂场景。
- 跨页文本合并依据句末标点、章节标记、TOC、列表符号判定边界。
- `_merge_adjacent_text_blocks_in_same_page` 回收检测器切割产生的碎片文本，保证 Markdown 段落连续。

### Markdown 与媒体资产
- format_transformer_v3 负责文本清洗、LaTeX 归一化、Picture 单元落地至 `images/`。
- `_process_base64_images_with_custom_naming` 结合标题命名图片，可选保留/跳过无标题图片，支持表格截图命名。
- `_filter_duplicate_images` 基于感知哈希聚类删除重复图片并同步清理 Markdown 引用。
- `_add_page_tag` 可选在页尾写入 `<special_page_num_tag>`，满足自动化 QA 页级定位。

### 调试与运维
- `--save_page_json`、`--save_page_layout` 持久化模型输出与布局图。
- `--debug_matching` 打印图表标题匹配过程。
- Badcase 自动收集：失败页面截图保存至 `--badcase_collection_dir`。
- 运行结束输出成功率、重试次数、错误分布、吞吐等指标便于排查。

## 架构与处理流程

### 三阶段流式管线
1. **PASS 1：并发渲染与推理**  
   `ProcessPoolExecutor` 渲染 PDF → JPEG，处理空白页后交给异步推理；队列大小 `--queue_size` 防止 backlog。
2. **PASS 2：跨页分析**  
   `_run_all_post_processing_worker` 在独立进程执行页内匹配、跨页表格匹配、跨页文本合并，避免主事件循环阻塞。
3. **PASS 3：Markdown 生成**  
   `_generate_md_for_one_page` 在信号量保护下并发生成单页 Markdown，窗口化策略顺序写盘，处理 base64 图片、表格截图以及重复图片去重。

### 错误处理与降级流程

```
开始处理页面
|
'--> [阶段1: 主路径尝试]
     |
     |   1. 调用 VLM 进行完整版面分析推理
     |
     '--- 如果失败 (网络超时、解析错误等) --> 进入 [阶段2]
     |
     '--- 如果成功 --> [最终状态: 'success'] --> 结束

'--> [阶段2: 文本回退]
     |
     |   1. 改用纯文本提示词请求
     |   2. 对返回文本做基础清洗并写入 Markdown
     |
     '--- 如果失败 --> 进入 [阶段3]
     |
     '--- 如果成功 --> [最终状态: 'success_fallback_text'] --> 结束

'--> [阶段3: 图片兜底]
     |
     |   1. 保存当前页面的原始截图
     |   2. (可选) 同步到 badcase 目录
     |   3. 在 Markdown 中插入截图引用
     |
     '--- [最终状态: 'success_fallback_image'] --> 结束

'--> [最终状态: 'error']（仅在截图保存亦失败时触发）
```

并发竞速重试（`--concurrent_retries`）和串行重试（`--max_retries`）贯穿各阶段；失败页会记录在运行摘要并输出详细错误信息。

## PaddleOCR 安装

当使用 `--enable_table_reparse` 功能时，需要额外安装 PaddleOCR 及其相关依赖。请执行以下命令：

```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

配置PaddleOCR-VL

```bash
docker run -d \
    --name paddlex-ocr-server \
    --gpus device=6 \
    --network host \
    --restart=unless-stopped \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server \
    paddlex_genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 12500 --backend vllm
```

## 参数说明（按类别划分）

### 输入 / 输出
- `--input-dir` / `--input-file`：二选一；支持 PDF、PNG、JPG、TIFF。
- `--output-dir`：输出根目录（默认 `./output`）；每个文件生成独立子目录与 `images/`。
- `--rename`：强制重命名输出 Markdown（与断点续传互斥）。
- `--prompt`：提示模板（默认 `prompt_layout_all_en`）；结合 `dict_promptmode_to_prompt` 中定义。
- `--bbox x1 y1 x2 y2`：对单页进行裁剪解析。

### 模型与连接
- `--ip` / `--port`：推理服务地址（默认 `localhost:8000`）。
- `--model_name`、`--temperature`、`--top_p`、`--max_completion_tokens`：模型相关参数。
- `--api_key`：API 鉴权；脚本也会读取 `API_KEY` 环境变量。
- `--timeout`：单次请求超时时间（默认 900s）。
- `--no_warmup`：禁用 GPU 预热。

### 性能与并发
- `--file_concurrency`：文件级并发度（默认 8）。
- `--page_concurrency`：页级推理信号量（默认 48），同时决定 PASS 1 消费者数量。
- `--md_gen_concurrency`：Markdown 生成并发度（默认 32；设为 0 时按 CPU 核心推断）。
- `--num_cpu_workers`：PDF 渲染进程池大小（默认 32；0 表示自动估算）。
- `--queue_size`：PASS 1 队列上限（默认 300）。
- `--dpi` / `--min_pixels` / `--max_pixels`：输入图片像素护栏，避免过大或过小。

### 内容后处理
- `--enable_table_reparse`: 布尔参数，启用后会调用 PaddleOCR 对表格进行二次解析。
- `--skip_footnote`: 布尔参数，跳过脚注内容。
- `--trim_first_page_summary`: 布尔参数，移除首页摘要。
- `--keyword_filter_config`: 字符串，提供一个 JSON 配置文件路径，用于过滤页面。
- `--normalize_superscript`: 布尔参数，将上标转换为普通文本。

### 媒体与输出控制
- `--enable_table_screenshot`：保存标题驱动的表格截图。
- `--skip_uncaptioned_images`：丢弃未匹配标题的图片。
- `--keep_page_header` / `--keep_page_footer`：保留页眉页脚文本。
- `--skip_blank_pages`：跳过空白页（默认在 `run_ocr.sh` 中启用）。
- `--no_filter_qr_barcodes`：关闭二维码/条码过滤。
- `--no_filter_duplicates`：关闭重复图片去重。
- `--add_page_tag`：布尔参数，启用时在每页末尾追加 `<special_page_num_tag>`。
- `--badcase_collection_dir`：失败页截图目录（默认 `/mnt/dotsocr_badcase`）。

### 重试与断点续传
- `--concurrent_retries`：竞速重试请求数量（默认 4）。
- `--max_retries`：最大串行重试次数（默认 8）。
- `--retry_delay`：初始重试退避时间（默认 2.0s）。
- `--disable_resume`：禁用断点续传。
- `--force_reprocess`：忽略历史输出，强制全量重跑。
- `--show_skipped`：在摘要中展示跳过文件信息（默认开启）。

### 调试与开发
- `--save_page_json`：保存每页 raw JSON。
- `--save_page_layout`：保存每页布局可视化图。
- `--debug_matching`：输出匹配调试日志。
- `--verbose`：打印额外运行信息，配合 `console_write` 与 tqdm。

## 使用示例

### 启动脚本（推荐）
```bash
chmod +x run_ocr.sh
./run_ocr.sh --input-dir /path/to/files --output-dir ./output_dir --verbose
```

### 表格二次解析
```bash
python parser_async_v12.py \
    --input-dir /path/to/files \
    --output-dir ./output_dir \
    --enable_table_reparse
```

### 表格截图与无标题图片过滤
```bash
python parser_async_v12.py \
    --input-dir /path/to/files \
    --output-dir ./output_dir \
    --enable_table_screenshot \
    --skip_uncaptioned_images
```

### 高性能配置（按硬件调整）
```bash
python parser_async_v12.py \
    --input-dir /path/to/large_dataset \
    --output-dir ./output_dir \
    --file_concurrency 24 \
    --page_concurrency 64 \
    --num_cpu_workers 48 \
    --md_gen_concurrency 48 \
    --queue_size 800
```

### 调试复杂版面
```bash
python parser_async_v12.py \
    --input-dir /path/to/complex_layout_files \
    --output-dir ./debug_output \
    --save_page_json \
    --save_page_layout \
    --debug_matching \
    --verbose
```

## 调试与开发建议
- 疑难文档推荐同时启用 `--save_page_json` / `--save_page_layout`，结合 `demo_pdf_test_output` 的 Markdown diff 定位问题。
- 修改 `dots_ocr/utils/format_transformer_v3.py` 时同步补充旁路单测（同目录下命名 `test_*.py`），覆盖 Section-header 标记与 LaTeX 清理逻辑。
- 推理端点不稳时调大 `--concurrent_retries` 或延长 `--timeout`，留意运行末尾的性能摘要。
- 表格截图依赖检测 bbox，调试时组合 `--enable_table_screenshot --save_page_layout` 观察框体是否匹配。

## 验证与日常命令
- 语法检查：`python -m compileall parser_async_v12.py`
- 回归测试：
  ```bash
  python parser_async_v12.py \
      --input-dir demo_pdf_test \
      --output-dir demo_pdf_test_output \
      --verbose \
      --skip_blank_pages
  ```
- 输出对比：与 `demo_pdf_test_output` 或历史基线进行 Markdown diff。

## 注意事项
- 默认开启的二维码/条码过滤会移除相关图片；若需要保留，请带上 `--no_filter_qr_barcodes`。
- 重复图片清理会删除 `images/` 下的高相似图片；若需保留全集，请关闭去重或提前备份。
- `--rename` 会关闭断点续传并覆盖输出，批量模式下谨慎使用。
- 长文档解析完成后会自动清理临时渲染目录；若异常退出，可手动删除输出目录中的 `_tmp_` 文件夹。
- 新增脚本或入口时同步更新 `run_ocr.sh`，保证一键脚本继承最新默认参数。