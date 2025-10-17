# Title Chunk Embedding 项目

这是一个用于处理PDF文档、提取标题结构并生成文本嵌入向量的完整工作流项目。

## 项目概述

该项目实现了一个端到端的文档处理流程，包括：
1. **OCR文档转换** - 将PDF转换为可处理的文本格式
2. **标题结构提取** - 从文档中提取层次化的标题结构
3. **文本分块** - 将文档内容按标题结构进行智能分块
4. **向量嵌入** - 为每个文本块生成向量表示

## 文件结构

```
title_chunk_embedding/
├── 1_ocr_doc2x.py          # Doc2X OCR处理脚本
├── 1_ocr_mineru.py         # Mineru OCR处理脚本
├── 2_extract_tree.py       # 标题结构提取脚本
├── 3_embedding.py          # 文本嵌入生成脚本
└── README.md               # 项目说明文档
```

## 工作流程

### 第一阶段：OCR文档转换

#### 选项1：使用Doc2X (`1_ocr_doc2x.py`)
- 支持PDF文档的OCR识别和转换
- 自动处理大文件拆分（超过1000页或300MB）
- 输出Markdown格式文件
- 支持断点续传和错误重试

**使用方法：**
```bash
python3 1_ocr_doc2x.py
```

**配置参数：**
- `API_KEY`: Doc2X API密钥
- `INPUT_DIR`: 输入PDF目录
- `OUTPUT_DIR`: 输出目录
- `MAX_FILE_SIZE_MB`: 最大文件大小限制

#### 选项2：使用Mineru (`1_ocr_mineru.py`)
- 支持批量PDF处理
- 多端口轮询负载均衡
- 并发处理提高效率
- 支持断点续传

**使用方法：**
```bash
python3 1_ocr_mineru.py input_dir output_dir [options]
```

**主要参数：**
- `--max-workers`: 并发数（默认4）
- `--resume`: 断点续跑
- `--ports`: 端口列表（如：30000-30007）
- `--timeout`: 单任务超时时间

### 第二阶段：标题结构提取 (`2_extract_tree.py`)

从OCR后的Markdown文档中提取层次化的标题结构，这是整个流程的核心环节：

#### 核心算法逻辑

**1. 滑动窗口处理机制**
- 使用可配置的窗口大小（默认80行）和步长（默认60行）
- 将长文档分割成重叠的文本片段，确保标题不被截断
- 每个窗口都包含绝对行号，便于定位和调试

**2. 两阶段标题提取**
- **第一阶段：初始提取** - 从每个文本片段中识别标题
- **第二阶段：结构验证** - 使用更大的验证窗口重新检查标题完整性
- 验证窗口大小 = 提取窗口 × 1.5，步长也相应增大

**3. 智能标题识别**
- 支持多种标题格式：
  - 数字编号：1.、1.1、1.1.1
  - 中文编号：第一章、第一节、一、二、三
  - 字母编号：A.、B.、C.、a.、b.、c.
  - 罗马数字：I.、II.、III.、i.、ii.、iii.
  - 括号编号：(1)、(2)、(3)、（一）、（二）、（三）
  - 方括号编号：[1]、[2]、[3]、【一】、【二】、【三】

**4. 文档内容分块策略**
- 从第一个标题的最后一个匹配位置开始截取正文（避免目录干扰）
- 按标题顺序智能分割文档内容
- 自动跳过标题后的标点符号（句号、顿号等）

**5. 附件检测机制**
- 自动识别文档中的附件、参考文献、附录等非正文内容
- 检测到附件时停止后续标题提取，避免误处理

#### 技术特性

**并发处理**
- 支持多线程并发处理（默认8线程，可配置）
- 使用ThreadPoolExecutor实现高效的并发执行
- 自动处理文件描述符限制问题

**错误处理与重试**
- 指数退避重试机制（1.8^attempt + 随机抖动）
- 智能JSON解析，支持Unicode转义序列清理
- 详细的错误日志和调试信息

**调试支持**
- 可选的调试日志目录，记录所有API调用详情
- 保存原始提示词、响应内容和错误信息
- 支持sanitized JSON的调试输出

#### 使用方法

```bash
python3 2_extract_tree.py input_base_dir --out-dir output_dir [options]
```

**主要参数：**
- `input_base_dir`: 基础目录（自动查找其中所有以`*_output`结尾的文件夹）
- `--out-dir`: 输出目录（默认：out_tree）
- `--toc-window`: 标题提取窗口大小，行数（默认：80）
- `--toc-stride`: 标题提取步长，行数（默认：60）
- `--max-retries`: 最大重试次数（默认：3）
- `--timeout`: Ark API请求超时秒数（默认：120）
- `--max-workers`: 并发线程数（默认：8）
- `--debug-log-dir`: 调试日志目录（可选）

#### 输出结构

**目录结构**
```
output_dir/
├── folder1_output/
│   ├── document1/
│   │   ├── document1.md          # 原始Markdown文件
│   │   ├── structure.json        # 提取的标题结构
│   │   └── sections.jsonl        # 分块后的章节内容
│   └── document2/
│       ├── document2.md
│       ├── structure.json
│       └── sections.jsonl
└── folder2_output/
    └── ...
```

**文件格式**
- `structure.json`: 包含所有提取的标题
- `sections.jsonl`: 每行一个JSON对象，包含章节标题、内容和源文件信息


#### 实际运行示例

**后台运行（推荐用于生产环境）**
```bash
# 使用nohup在后台运行，输出重定向到日志文件
nohup python3 2_extract_tree.py \
  /mnt/data/projects/tmp \
  --out-dir /mnt/data/wx/xiaofang/out_xiaofang_title_$(date +%m%d%H%M) \
  --debug-log-dir /mnt/data/wx/xiaofang/debug_tree_logs_$(date +%m%d%H%M) \
  --toc-window 80 \
  --toc-stride 60 \
  --max-retries 3 \
  --timeout 120 \
  --max-workers 100 \
  > extract_tree_$(date +%m%d%H%M).log 2>&1 &

# 查看运行状态
tail -f extract_tree_$(date +%m%d%H%M).log

# 查看进程
ps aux | grep extract_tree
```

### 第三阶段：文本嵌入生成 (`3_embedding.py`)

为提取的文本块生成向量嵌入：

**功能特性：**
- 支持多种嵌入模型
- 多API端点轮询负载均衡
- 智能文本分块和清理
- 输出标准化的JSONL格式

**使用方法：**
```bash
python3 3_embedding.py --input_path input_dir --output_path output_dir
```

**配置参数：**
- `api_urls`: 嵌入API端点列表
- `model_name`: 使用的嵌入模型名称

## 输出格式

```json
{"section_title": "1. 引言", "content": "章节内容...", "embedding": [0.1, 0.2, ...]}
```

## 依赖安装

```bash
# 基础依赖
pip install tqdm requests

# Doc2X依赖
pip install pypdf pdfdeal

# Ark依赖
pip install 'volcengine-python-sdk[ark]'

# Mineru (需要单独安装)
# 请参考Mineru官方文档进行安装
```

## 环境配置

### API密钥配置
- **Doc2X**: 在`1_ocr_doc2x.py`中配置`API_KEY`
- **Ark**: 在`2_extract_tree.py`中配置`API_KEY`和`MODEL_ID`

### 服务端点配置
- **Mineru**: 在`1_ocr_mineru.py`中配置`--url-host`和`--ports`
- **Embedding**: 在`3_embedding.py`中配置`api_urls`列表

## 使用示例

### 完整流程示例

```bash
# 1. OCR转换（选择一种方式）
python3 1_ocr_doc2x.py
# 或
python3 1_ocr_mineru.py /path/to/pdfs /path/to/output --max-workers 4

# 2. 标题提取
python3 2_extract_tree.py --input_dir /path/to/md_files --output_dir /path/to/titles

# 3. 生成嵌入
python3 3_embedding.py --input_path /path/to/titles --output_path /path/to/embeddings
```

### 后台运行

```bash
# 使用nohup后台运行
nohup python3 3_embedding.py --input_path /path/to/input --output_path /path/to/output > embedding.log 2>&1 &
```

## 注意事项

1. **文件大小限制**: 大PDF文件会自动拆分处理
2. **API配额**: 注意各服务的API调用限制
3. **并发控制**: 根据服务器性能调整并发数
4. **错误处理**: 所有脚本都支持断点续传和错误重试
5. **日志记录**: 详细的操作日志便于调试和监控

## 故障排除

### 常见问题

1. **API调用失败**: 检查API密钥和网络连接
2. **内存不足**: 减少并发数或调整滑动窗口大小
3. **文件权限**: 确保输出目录有写入权限
4. **超时错误**: 调整超时参数或检查网络状况

### 日志文件

- Doc2X: `doc2x_log.txt`
- Mineru: `mineru_batch_*.log`
- 标题提取: 控制台输出
- 嵌入生成: 控制台输出

