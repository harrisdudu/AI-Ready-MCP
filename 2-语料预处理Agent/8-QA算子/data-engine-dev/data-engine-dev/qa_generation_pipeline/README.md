
# QA_Generation_Pipeline

## 项目概述

本项目提供四个专门的Pipeline，用于从Markdown文档生成高质量的问答对（QA对）。每个Pipeline针对不同的应用场景和需求设计，采用异步处理和并行计算技术，确保高效处理大量文档。

## Pipeline概览

### 1. Book Pipeline
**适用场景**：书籍、技术文档、长篇结构化内容，多生成知识类/推理类问题。

**核心特点**：
- 分层内容处理：使用HierarchicalTextRecall系统进行智能内容召回
- 两阶段生成：先独立生成问题，再基于问题生成答案
- 高质量评估：使用专门的评估机制确保问答对质量
- 适合处理：书籍章节、技术规范、系统文档等长篇内容

**处理流程**：
1. 内容分割与预处理
2. 问题生成
3. 答案生成
4. QA对评估
5. 结果整理与保存

### 2. Policy Pipeline
**适用场景**：政策文件、法规文档、结构化指导文件，多生成概念型问题。

**核心特点**：
- 脑图/目录生成：在生成问题前先创建内容脑图/目录
- 概念导向：基于概念和结构生成相关问题
- 深度内容理解：通过脑图确保问题覆盖完整内容结构
- 适合处理：政策法规、标准规范、指导文件等结构化文档

**特殊功能**：
- 使用`context_extraction_prompt`生成内容脑图/目录
- 基于脑图/目录结构生成概念性问题
- 保持内容层级关系的问答对生成

### 3. Common Pipeline
**适用场景**：通用文档、快速QA对生成、标题结构清晰的文档

**核心特点**：
- 直接标题分割：按Markdown标题直接分割内容
- 一次性生成：直接生成完整的QA对，而非分步生成
- 高效处理：处理速度快，适合大量文档
- 适合处理：各种通用文档，特别是标题结构清晰的Markdown文件

**特殊功能**：
- 使用`split_markdown_by_headers`按标题分割内容
- 使用`QA_generate_prompt_per_block`一次性生成QA对
- 简化处理流程，无需复杂召回系统

### 4. Multiturn Pipeline
**适用场景**：多轮对话、连续问答、教学场景，两轮QA为要求为一轮概念，一轮场景的递进式问题对。

**核心特点**：
- 多轮结构：生成具有逻辑关联的问题对
- 顺序保全：保持问题对的顺序和递进关系
- 对话连贯：确保多轮问答之间的逻辑连贯性
- 适合处理：需要连续对话的场景，如教学问答、技术支持

**特殊功能**：
- 使用`turn_2_prompt`生成两轮问题对
- 保持多轮数据结构：`List[List[Dict[str, Any]]]`
- 特殊评估处理：平铺多轮结构进行评估后重组

## 技术架构

### 共同技术基础
所有Pipeline共享以下技术架构：
- **异步处理**：使用asyncio进行高效异步I/O操作
- **并行计算**：线程池处理I/O密集型任务，进程池处理CPU密集型任务
- **生产者-消费者模式**：高效处理大量文件内容
- **进度监控**：使用tqdm实时显示处理进度
- **错误处理**：完善的异常处理机制

### 差异点
- **召回系统**：在reference 部分 Book/Policy/Multiturn使用HierarchicalTextRecall，Common不使用。 Policy 会额外通过模型在COT部分输出 {ref_i} 标签，进一步筛选reference。
- **提示词模板**：各Pipeline使用不同的提示词优化生成效果
- **数据结构**：Multiturn使用特殊的多轮数据结构

## 安装与使用

### 环境要求
- Python 3.8+
- 依赖库：见`requirements.txt`

### 基本使用

请在 `config.py` 中设置好所有的配置

```bash
# Book Pipeline
python ./core/book_pipeline_batch.py

# Policy Pipeline
python ./core/policy_pipeline_batch.py

# Common Pipeline
python ./core/common_pipeline_batch.py

# Multiturn Pipeline
python ./core/multiturn_pipeline_batch.py
```

## 性能优化建议

1. **资源调整**：根据系统资源调整`max_workers`参数
2. **批量处理**：一次性处理相关文档以提高效率
3. **缓存利用**：Policy Pipeline的脑图生成有缓存机制
4. **模型选择**：根据不同内容类型选择合适的LLM模型, 指令遵循能力强的模型（如`deepseek-v3-250324`），亲测 `doubao-seed-1-6-flash-250715` 在生成质量和评测QA时，效果不佳。

## 输出格式

所有Pipeline生成标准化的JSON输出，包含以下字段：
- `Q`: 问题文本
- `A`: 答案文本
- `ref`: 参考内容列表
- `file_path`: 源文件路径
- `file_name`: 源文件名
- `judge`: 评估结果

Multiturn Pipeline额外保持多轮结构。

## 扩展与定制

### 添加新Pipeline
待后续对Pipeline进行进一步拆分创建基类。

### 自定义提示词
各Pipeline的提示词模板可自定义，位于`core/prompts.py`中。

## 故障排除

常见问题及解决方案：
1. **内存不足**：减少`max_workers`或增加队列大小限制
2. **处理速度慢**：调整线程池/进程池大小
3. **API限制**：增加超时时间或减少并发请求
4. **内容质量差**：调整提示词或模型参数



