# 考试图片分割与链式思维推理 (Exams Split Images Chain-of-Thought)

这个项目是一个专门用于处理考试图片的自动化工具链，主要功能包括图片OCR识别和题目内容提取，最终生成结构化的题目数据用于链式思维推理训练。

## 项目概述

该项目主要解决以下问题：
- 将考试图片进行OCR识别，提取文字内容
- 从OCR结果中提取题目、答案、解析等结构化信息
- 生成标准化的JSON格式数据，便于后续的链式思维推理训练

## 项目结构

```
exams_splitted-images_cot/
├── 1_doc2x_image_ocr.py      # 图片OCR处理脚本
├── 2_extract_questions.py     # 题目内容提取脚本
└── README.md                  # 项目说明文档
```

## 核心组件

### 1. 图片OCR处理 (`1_doc2x_image_ocr.py`)

**功能描述：**
- 使用Doc2X API对考试图片进行OCR识别
- 支持批量处理大量图片文件
- 具备断点续传功能，可处理中断的任务
- 自动重试失败的OCR请求

**主要特性：**
- 多线程并发处理，提高处理效率
- 智能重试机制，最大重试次数可配置
- 详细的处理日志和统计信息
- 支持按竞赛、科目等目录结构组织

**使用方法：**
```bash
# 完整处理模式
python 1_doc2x_image_ocr.py /path/to/source

# 断点续传模式（处理失败的图片）
python 1_doc2x_image_ocr.py /path/to/source --resume
```

**输出结构：**
```
source_output/
├── 竞赛名称/
│   └── 科目名称/
│       ├── 题干/
│       │   └── 图片名称/
│       │       └── output.md
│       ├── 答案/
│       └── 解析/
└── processing_summary.json
```

### 2. 题目内容提取 (`2_extract_questions.py`)

**功能描述：**
- 从OCR结果中提取题目、答案、解析等结构化信息
- 使用大语言模型（豆包）进行智能内容理解
- 生成标准化的JSON格式数据
- 自动处理图片重命名和路径映射

**主要特性：**
- 智能题目类型识别和分类
- 多线程并发处理，支持大量题目
- 自动生成题目ID和索引
- 完整的日志记录和错误处理

**使用方法：**
```bash
python 2_extract_questions.py <OCR输出目录> <原始图片目录> [选项]

# 示例
python 2_extract_questions.py /path/to/source_output /path/to/original_images -j 100
```

**参数说明：**
- `root_path`: OCR输出目录路径
- `original_path`: 原始图片目录路径
- `-o, --output`: 输出目录名（可选）
- `-j, --jobs`: 并发线程数（默认200）

**输出结构：**
```
CoT_输出目录名_时间戳/
├── source/                    # 原始图片副本
├── images/                    # 处理后的图片
├── questions.json            # 结构化题目数据
└── extract_questions.log     # 处理日志
```

## 依赖要求

### Python包依赖
```bash
pip install requests pillow tqdm volcengine-python-sdk[ark]
```

### API密钥配置
- **Doc2X API**: 需要在脚本中配置 `API_KEY`
- **豆包API**: 需要在脚本中配置 `API_KEY` 和 `MODEL_ID`

## 工作流程

1. **图片OCR处理**
   - 扫描输入目录结构
   - 调用Doc2X API进行OCR识别
   - 保存识别结果到markdown文件
   - 生成处理统计报告

2. **题目内容提取**
   - 读取OCR结果文件
   - 使用大语言模型提取结构化信息
   - 生成标准化的题目数据
   - 保存为JSON格式

3. **数据输出**
   - 生成完整的题目数据集
   - 包含图片路径映射
   - 提供详细的处理日志

## 配置说明

### 并发和限流配置
```python
MAX_WORKERS = 8          # 最大并发线程数
RATE_LIMIT = 1           # 每秒最大请求数
MAX_RETRIES = 10         # 最大重试次数
```

### API配置
```python
# Doc2X配置
BASE_URL = "https://v2.doc2x.noedgeai.com/api/v2/parse/img/layout"
HEADERS = {'Authorization': 'Bearer YOUR_API_KEY'}

# 豆包配置
API_KEY = "your_api_key"
MODEL_ID = "doubao-seed-1-6-250615"
```

## 故障排除

### 常见问题
1. **OCR失败率高**：检查图片质量、空白大小
2. **处理速度慢**：调整并发线程数和限流参数
3. **内存不足**：减少并发线程数，分批处理
