# 1.日志处理函数

负责日志的加载与增量写入，确保任务可断点续传
**日志格式：JSONL（每行一个 JSON 对象）**

```
├── load_existing_results_full(log_file: str) → Dict[str, Dict]
│   └── 功能：从日志文件中加载已处理的 `input_path` 及其结果
│       ├── 按行解析 JSONL
│       ├── 缓存 `input_path` → result 的映射
│       └── 异常行仅警告，不中断
│
└── update_or_append_to_jsonl(log_file: str, result: Dict, cache_dict: Dict)
    └── 功能：安全地将新结果写入日志（原子写）
        ├── 使用内存缓存 `cache_dict` 维护所有已处理项
        ├── 写入临时文件 `.tmp` 避免损坏原文件
        ├── 成功后 `os.replace()` 原子替换
        └── 异常时清理临时文件
```

# 2.文本分块函数

将大文件切分为适合模型处理的文本块，避免超出上下文限制

```
└── split_text_into_chunks(text: str, chunk_size=4096) → List[Tuple[start, end, chunk]]
    └── 智能断句策略：
        ├── 优先在 `\n\n`, `\n`, `。`, ` ` 处切分
        ├── 向前查找 50 字符，向后 50 字符内搜索最佳断点
        ├── 保证每块 ≥100 字符，避免过小
        └── 返回 (起始偏移, 结束偏移, 文本块) 三元组
```

# 3.规则提取函数（正则匹配）

基于固定规则提取非姓名类敏感信息（号码类）

```
├── extract_mobiles(content: str) → List[str]
│   └── 正则: `r'1\d{10}'` → 匹配手机号
│
├── extract_phones(content: str) → List[str]
│   └── 正则: `r'\d{3,4}-\d{7,8}'` → 匹配固话（如 010-12345678）
│
└── extract_id_cards(content: str) → List[str]
    └── 正则: `r'\d{17}[\dXx]'` → 匹配身份证号（转大写）
```

# 4.脱敏替换函数

对识别出的敏感信息进行掩码处理

```
└── desensitize_content(original_content: str, sensitive_items: List[str]) → str
    └── 处理流程：
        ├── 去重并按长度降序排序（防止子串误替换）
        ├── 逐项正则转义后替换
        ├── 根据类型选择脱敏方式：
        │   ├── 手机号 → `138****1234`
        │   ├── 固话 → `*******`
        │   ├── 身份证 → `110101********1234`
        │   └── 其他（如姓名）→ `张*` 或 `张***`
        └── 返回脱敏后全文
```

# 5. 文件读取与输入生成

负责扫描 `.md` 文件并生成模型请求任务

```
├── read_md_files(md_root: str, skip_paths: set, target_files: Optional[set]) → Iterator[(path, content)]
│   └── 功能：递归读取所有 `.md` 文件
│       ├── 跳过已成功处理的文件（根据日志）
│       ├── 支持指定目标文件子集（用于调试）
│       └── 遇错打印警告并跳过
│
└── md_input_generator_for_desensitization(...) → Iterator[API 请求 dict]
    └── 输入：目录、缓存、目标文件
    └── 输出：每个 chunk 的模型请求配置
        ├── 每个 chunk 构造一条 `messages` 提示
        ├── system prompt 明确提取规则（姓名 + 职位相关）
        ├── user prompt 包含文件名、块序号、内容片段
        ├── `custom_id = "{path}::chunk_{idx}"` → 用于回溯 chunk 归属
        ├── 附加 `input_path` 和唯一请求头（`CLIENT_REQUEST_HEADER`）
        └── 仅生成未完成的 chunk 任务
```

# 6. Doubao 批处理框架（多进程 + 异步并发）

核心并发执行引擎，支持高吞吐调用大模型 API

└── DoubaoBatchProcessor 类
    ├── __init__(...)
    │   └── 配置：生成器函数、参数、worker 数、并发数、模型、API Key
    │
    ├── run(output_handler: Callable) → None
    │   └── 启动主流程：
    │       ├── 创建 Manager 管理共享队列
    │       ├── 启动 1 个 input producer 进程
    │       ├── 启动 N 个 worker 进程
    │       ├── 主线程消费输出队列，调用 handler
    │       └── 支持中断与清理
    │
    ├── _input_producer(...) → None
    │   └── 预加载所有任务 → 转为 list
    │   └── 使用 tqdm 显示分发进度
    │   └── 入队至 `in_queue`，结束发送 N 个 `None` 信号
    │
    ├── _worker_process(...) → None
    │   └── 每个 worker 启动独立 asyncio event loop
    │   └── 创建 AsyncArk 客户端（带限流）
    │   └── 从队列取任务，异步并发调用 `_process_task`
    │   └── 任务结束发送 `None` 到 `out_queue`
    │
    ├── _process_task(client, record, out_queue, sem) → None
    │   └── 异步调用模型 API
    │   └── 成功：返回完整响应（含 `custom_id`, `input_path`）
    │   └── 失败：构造 error_result 返回
    │   └── 释放信号量
    │
    └── _make_client(max_concurrency) → httpx.AsyncClient
        └── 配置高并发异步客户端
        └── 设置连接池与超时（7200s）

# 7.结果处理器（ResultHandler）

处理模型返回结果，合并 chunk，执行脱敏，写回文件

```
└── ResultHandler(log_file, log_cache, progress_bar)
    ├── handle(result: Dict) → None
    │   └── 解析 `custom_id` 获取 `input_path` 和 `chunk_index`
    │   └── 成功时解析 JSON 响应 → 提取姓名列表
    │   └── 缓存每个文件的所有 chunk 提取结果
    │   └── 更新日志（chunk 级）
    │   └── 打印状态（✅/❌ + 姓名数 + 错误信息）
    │   └── 触发 `_try_merge_file`
    │
    └── _try_merge_file(input_path: str) → None
        └── 当所有 chunk 处理完后触发（实际是每次 handle 都尝试）
        └── 加载原始全文（若未缓存）
        └── 规则提取：手机号、固话、身份证
        └── 模型提取：合并所有 chunk 的姓名 → 去重
        └── 合并所有敏感项 → 脱敏替换
        └── 写回原文件（覆盖）
        └── 记录最终成功日志（含提取详情）
        └── 打印详细提取信息（姓名、号码等）
```

# 8. 主执行入口（`__main__`）

整体流程控制与参数配置

```
└── if __name__ == "__main__":
    ├── 设置 multiprocessing 启动方式为 'spawn'
    ├── 配置参数：
    │   ├── DEBUG_MODE / DEBUG_COUNT：控制调试范围
    │   ├── MD_ROOT_DIR：Markdown 根目录
    │   ├── NUM_WORKERS / MAX_CONCURRENCY：并发控制
    │   ├── LOG_FILE_PATH：日志路径
    │   └── MODEL_NAME：调用模型
    │
    ├── 检查目录是否存在
    ├── 加载历史日志 → 构建 `log_cache`
    ├── 获取所有 .md 文件 → 计算待处理集合 `todo_files`
    ├── DEBUG 模式下限制处理数量
    ├── 若无待处理文件 → 退出
    │
    ├── 初始化 tqdm 进度条（按文件计）
    ├── 创建 ResultHandler（绑定日志、缓存、进度条）
    ├── 创建 DoubaoBatchProcessor（传入生成器与参数）
    ├── 启动 processor.run(handler.handle)
    │
    └── 完成后输出总结信息
```

