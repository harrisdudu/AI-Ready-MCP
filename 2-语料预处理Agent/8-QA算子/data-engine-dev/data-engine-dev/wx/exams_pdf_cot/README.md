# Exams PDF CoT 项目

这是一个专门用于处理考试PDF文档、提取题目和答案、生成推理过程（Chain of Thought, CoT）的完整工作流项目。

## 项目概述

该项目实现了一个端到端的考试文档处理流程，包括：
1. **文档格式转换** - 将各种格式的考试文档转换为PDF
2. **OCR识别** - 使用Mineru进行PDF文档的OCR识别
3. **文档类型判断** - 智能识别试卷的题目和答案组织方式
4. **题目答案分离** - 根据文档类型进行相应的题目答案提取
5. **CoT生成** - 为题目生成详细的推理过程
6. **质量检查** - 验证生成内容的准确性和一致性

## 文件结构

```
exams_pdf_cot/
├── 0_to_pfd.py                    # 文档格式转换脚本
├── 1_ocr.py                       # OCR识别脚本
├── 2_type.py                      # 文档类型判断脚本
├── 3.1_qa_apart/                 # 题目答案分离处理模块
│   ├── README.md                  # 分离处理说明
│   ├── 3.1.1_qa_apart_split.py  # 题目答案分离脚本
│   └── 3.1.2_qa_apart.py        # 分离后的处理脚本
├── 3.2_qa_together/              # 题目答案混合处理模块
│   ├── README.md                  # 混合处理说明
│   ├── 3.2.1_qa_together_without_dep.py  # 无依赖处理脚本
│   ├── 3.2.2_deduplicate_alone.py        # 去重脚本
│   └── 3.2_qa_together_with_dep.py      # 有依赖处理脚本
├── 4_cot_generate_and_check.py   # CoT生成和质量检查脚本
├── 5_extra_check.py              # 额外质量检查脚本
└── README.md                      # 项目说明文档
```

## 工作流程

### 第一阶段：文档格式转换 (`0_to_pfd.py`)

将各种格式的考试文档转换为PDF格式：

**支持格式：**
- Office文档：`.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`
- WPS文档：`.wps`, `.et`, `.dps`
- OpenDocument：`.odt`
- 已有PDF文件直接复制

**使用方法：**
```bash
python3 0_to_pfd.py
```

**功能特性：**
- 使用LibreOffice进行格式转换
- 自动创建输出目录
- 统计转换结果和文件数量
- 支持批量处理

### 第二阶段：OCR识别 (`1_ocr.py`)

使用Mineru对PDF文档进行OCR识别：

**使用方法：**
```bash
python3 1_ocr.py input_dir output_dir [options]
```

**主要参数：**
- `--max-workers`: 并发数（默认4）
- `--resume`: 断点续跑
- `--ports`: 端口列表（如：30000-30007）
- `--timeout`: 单任务超时时间（默认1800秒）
- `--retries`: 失败重试次数（默认1）

**功能特性：**
- 多端口轮询负载均衡
- 并发处理提高效率
- 支持断点续传
- 详细的日志记录

### 第三阶段：文档类型判断 (`2_type.py`)

智能判断试卷的题目和答案组织方式：

**四种类型：**
1. **ONLY_QUESTIONS** - 只有题目，无答案和解析
2. **ONLY_ANSWERS_EXPLANATIONS** - 只有答案和解析，无题目
3. **MIXED_TOGETHER** - 题目和答案解析混合在一起
4. **ANSWERS_AT_END** - 答案和解析集中在文档末尾

**使用方法：**
```bash
python3 2_type.py input_dir [options]
```

**主要参数：**
- `--recursive`: 递归扫描子目录
- `--pattern`: 文件匹配规则（默认*.md）
- `--max-workers`: 并发线程数（默认8）

**技术特性：**
- 使用Ark大模型进行智能判断
- 基于文档前50行和后50行进行分析
- 支持批量处理和并发执行
- 输出详细的分类结果和置信度

### 第四阶段：题目答案处理

#### 选项1：题目答案分离处理 (`3.1_qa_apart/`)

适用于题目和答案分离的文档：

**处理流程：**
1. 智能识别题目和答案的边界
2. 将题目和答案分别提取到不同文件
3. 建立题目和答案的对应关系
4. 生成结构化的JSONL输出

**使用方法：**
```bash
# 分离题目和答案
python3 3.1.1_qa_apart_split.py input_dir output_dir

# 处理分离后的内容
python3 3.1.2_qa_apart.py input_dir output_dir
```

#### 选项2：题目答案混合处理 (`3.2_qa_together/`)

适用于题目和答案混合的文档：

**处理流程：**
1. 识别题目和答案的混合模式
2. 提取完整的题目-答案对
3. 支持去重和依赖处理
4. 生成标准化的输出格式

**使用方法：**
```bash
# 无依赖处理
python3 3.2.1_qa_together_without_dep.py input_dir output_dir

# 有依赖处理
python3 3.2_qa_together_with_dep.py input_dir output_dir

# 去重处理
python3 3.2.2_deduplicate_alone.py input_file output_file
```

### 第五阶段：CoT生成和质量检查 (`4_cot_generate_and_check.py`)

为题目生成详细的推理过程（Chain of Thought）：

**使用方法：**
```bash
python3 4_cot_generate_and_check.py \
  --input-jsonl input.jsonl \
  --out-jsonl output.jsonl \
  --max-workers 256 \
  --temperature 0.25 \
  --top-p 0.8 \
  --thinking enabled \
  --retries 5 \
  --max-correct-rounds 8
```

**主要参数：**
- `--input-jsonl`: 输入JSONL文件路径
- `--out-jsonl`: 输出JSONL文件路径
- `--max-workers`: 并发线程数（默认256）
- `--temperature`: 生成温度（默认0.25）
- `--thinking`: 思维链模式（enabled/disabled/auto）
- `--max-correct-rounds`: 自纠最多轮数（默认8）

**核心功能：**
1. **CoT生成** - 基于标准答案生成推理过程
2. **质量检查** - 验证生成内容与标准答案的一致性
3. **自纠机制** - 自动修正不匹配的内容
4. **多轮优化** - 支持多轮质量检查和修正

### 第六阶段：额外质量检查 (`5_extra_check.py`)

对生成的内容进行额外的质量验证：

**使用方法：**
```bash
python3 5_extra_check.py --in-jsonl input.jsonl
```

**功能特性：**
- 自动生成质量检查报告
- 分离通过和未通过的内容
- 输出详细的QC统计信息
- 支持批量处理和并发执行

## 依赖安装

```bash
# 基础依赖
pip install tqdm rich pandas

# Ark SDK
pip install 'volcengine-python-sdk[ark]'

# 系统依赖
# LibreOffice (用于文档转换)
sudo apt-get install libreoffice

# Mineru (需要单独安装)
# 请参考Mineru官方文档进行安装
```

## 环境配置

### API密钥配置
- **Ark**: 在各个脚本中配置`API_KEY`和`MODEL_ID`
- 建议将API密钥设置为环境变量

### 服务端点配置
- **Mineru**: 在`1_ocr.py`中配置`--url-host`和`--ports`

## 使用示例

### 完整流程示例

```bash
# 1. 文档格式转换
python3 0_to_pfd.py

# 2. OCR识别
python3 1_ocr.py /path/to/pdfs /path/to/output --max-workers 8

# 3. 文档类型判断
python3 2_type.py /path/to/ocr_output --recursive --max-workers 8

# 4. 根据类型选择处理方式
# 如果题目答案分离：
python3 3.1.1_qa_apart_split.py /path/to/input /path/to/output
python3 3.1.2_qa_apart.py /path/to/input /path/to/output

# 如果题目答案混合：
python3 3.2.1_qa_together_without_dep.py /path/to/input /path/to/output

# 5. 生成CoT
python3 4_cot_generate_and_check.py \
  --input-jsonl /path/to/questions.jsonl \
  --out-jsonl /path/to/questions_cot.jsonl \
  --max-workers 256

# 6. 额外质量检查
python3 5_extra_check.py --in-jsonl /path/to/questions_cot.jsonl
```

### 后台运行

```bash
# 使用nohup后台运行
nohup python3 4_cot_generate_and_check.py \
  --input-jsonl input.jsonl \
  --out-jsonl output.jsonl \
  --max-workers 256 \
  > cot_generation.log 2>&1 &

# 查看运行状态
tail -f cot_generation.log
```

## 输出格式

### 文档类型判断输出
```json
{
  "file": "document.md",
  "label": "MIXED_TOGETHER",
  "label_cn": "题和答案解析在一起",
  "confidence": 0.95,
  "rationale": "文档中题目和答案交替出现"
}
```

### 题目答案输出
```json
{
  "question": "题目内容",
  "answer": "标准答案",
  "explanation": "解析说明",
  "source_file": "源文件名"
}
```

### CoT生成输出
```json
{
  "question": "题目内容",
  "answer": "标准答案",
  "cot_content": "详细的推理过程...",
  "qc_passed": true,
  "qc_confidence": 0.98
}
```

