# coding:utf-8
"""
批量 Markdown 脱敏处理器（姓名由模型提取，号码由规则提取）
功能：分块调用大模型提取姓名，全文正则提取号码，合并后脱敏写回
日志：JSONL 格式，input_path 为真实路径
所有处理信息均打印输出  
"""

import os
import glob
import asyncio
import multiprocessing
import queue
import json
import re
from typing import Any, Dict, Optional, Callable, Iterator, List, Tuple
import httpx
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime._constants import CLIENT_REQUEST_HEADER
from tqdm import tqdm
from datetime import datetime

# ==================== 日志处理函数 ====================

def load_existing_results_full(log_file: str) -> Dict[str, Dict]:
    results = {}
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    input_path = entry.get("input_path")
                    if input_path:
                        results[input_path] = entry
                except Exception as e:
                    print(f"[警告] 无法解析日志行: {line.strip()[:100]}... 错误: {e}")
    return results

def update_or_append_to_jsonl(log_file: str, result: Dict[str, Any], cache_dict: Dict[str, Dict]):
    input_path = result.get("input_path")
    if not input_path or input_path == "unknown":
        print(f"[警告] 尝试记录日志但缺失有效 input_path: {result}")
        return

    cache_dict[input_path] = result

    temp_log_file = log_file + ".tmp"
    try:
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            for entry in cache_dict.values():
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        os.replace(temp_log_file, log_file)
    except Exception as e:
        print(f"[错误] 写入日志文件失败: {e}")
        if os.path.exists(temp_log_file):
            try:
                os.remove(temp_log_file)
            except OSError as rm_e:
                print(f"[错误] 删除临时日志文件失败: {rm_e}")


# ==================== 分块函数 ====================

def split_text_into_chunks(text: str, chunk_size: int = 4096) -> List[Tuple[int, int, str]]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:]
            chunks.append((start, len(text), chunk))
            break
        else:
            cut_point = end
            look_ahead = text[end - 50:end + 50]
            last_break = -1
            for sep in ['\n\n', '\n', '。', ' ', '']:
                idx = look_ahead.rfind(sep)
                if idx != -1 and start + end - 50 + idx > start + 100:
                    last_break = start + end - 50 + idx + len(sep)
                    break
            if last_break == -1 or last_break <= start:
                last_break = end
            chunk = text[start:last_break]
            chunks.append((start, last_break, chunk))
            start = last_break
    return chunks


# ==================== 规则提取函数 ====================

def extract_mobiles(content: str) -> List[str]:
    """提取手机号"""
    return re.findall(r'1\d{10}', content)

def extract_phones(content: str) -> List[str]:
    """提取固话（区号-号码格式）"""
    return re.findall(r'\d{3,4}-\d{7,8}', content)

def extract_id_cards(content: str) -> List[str]:
    """提取身份证号"""
    return [id_num.upper() for id_num in re.findall(r'\d{17}[\dXx]', content)]


# ==================== 脱敏函数 ====================

def desensitize_content(original_content: str, sensitive_items: List[str]) -> str:
    desensitized_content = original_content
    sorted_items = sorted(set(sensitive_items), key=len, reverse=True)

    for item in sorted_items:
        if not item or item.isspace():
            continue
        escaped_item = re.escape(item)

        if re.fullmatch(r'1\d{10}', item):
            replacement = f"{item[:3]}****{item[-4:]}"
        elif re.fullmatch(r'\d{3,4}-\d{7,8}', item):
            replacement = "*******"
        elif re.fullmatch(r'\d{17}[\dXx]', item):
            replacement = f"{item[:6]}********{item[-4:]}"
        else:
            replacement = item[0] + "*" * (len(item) - 1) if len(item) > 2 else (item[0] + "*")

        desensitized_content = re.sub(escaped_item, replacement, desensitized_content)

    return desensitized_content


# ==================== 文件读取 ====================

def read_md_files(md_root: str, skip_paths: set, target_files: Optional[set] = None) -> Iterator[Tuple[str, str]]:
    files_to_scan = list(target_files) if target_files is not None else glob.glob(os.path.join(md_root, "**/*.md"), recursive=True)
    filtered_files = [f for f in files_to_scan if f not in skip_paths]

    for md_path in filtered_files:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            yield md_path, content
        except Exception as e:
            print(f"❌ 读取文件失败，跳过: {md_path}, 错误: {e}")


# ==================== 输入生成器（使用 custom_id 编码 chunk 信息）====================

def md_input_generator_for_desensitization(md_root_dir: str, existing_log_cache: Dict[str, Dict], target_files: Optional[set] = None):
    successfully_processed_paths = {
        path for path, entry in existing_log_cache.items()
        if entry.get("is_success", False)
    }

    for path, content in read_md_files(md_root_dir, successfully_processed_paths, target_files):
        chunks = split_text_into_chunks(content, chunk_size=4096)
        print(f"📄 {path} -> 长度: {len(content)} 字符，分成 {len(chunks)} 个 chunk")  # ✅ 打印完整路径

        for idx, (start, end, chunk) in enumerate(chunks):
            # 使用 custom_id 编码：路径::chunk_索引
            custom_id = f"{path}::chunk_{idx}"

            yield {
                "messages": [
                    {
                        "role": "system",
                        "content": """你是一个专业的文档信息提取专家，擅长智能识别文档中的姓名类敏感信息。你的任务是准确提取与职位相关的个人姓名，以及符合常见姓名特征的普通个人姓名，并以指定的 JSON 格式列表返回。"""
                    },
                    {
                        "role": "user",
                        "content": f"""
请仔细分析以下 Markdown 内容片段，根据规则提取需要脱敏的**姓名类**敏感信息。

## 提取规则：
- 只提取：普通个人姓名（2-4个汉字）和 职位相关姓名（如：组长、副组长、负责人、联系人、项目经理等）
- 必须提取的职位：组长、副组长、组员、主任、副主任、成员、核心成员、经办人、负责人、执行人、联系人、对接人、项目经理、技术负责人等
- 不提取：手机号、电话、身份证号、公司名、地名、历史人物、公众人物

## 输出格式：
**只返回一个 JSON 数组，仅包含姓名字符串。不要包含任何其他内容。**
例如：["张三", "李四"]

原始内容片段（文件 {os.path.basename(path)} 的第 {idx+1} 块）：
---
{chunk}
---
请开始提取姓名：
"""
                    }
                ],
                "thinking":{
                    "type": "disabled", # 不使用深度思考能力
                    # "type": "enabled", # 使用深度思考能力
                    # "type": "auto", # 模型自行判断是否使用深度思考能力
                },
                "temperature": 0.2,
                "input_path": path,
                "custom_id": custom_id,  # ✅ 关键：chunk 信息编码在 custom_id 中
                "extra_headers": {
                    CLIENT_REQUEST_HEADER: str(uuid.uuid4())
                }
            }


# ==================== 批处理框架（透传 custom_id）====================

class DoubaoBatchProcessor:
    def __init__(
        self,
        input_generator_func: Callable[..., Iterator[Dict[str, Any]]],
        input_generator_args: Dict[str, Any],
        num_workers: int = 4,
        max_concurrency_per_process: int = 32,
        model: str = "doubao-pro-32k",
        api_key: Optional[str] = None,
    ):
        self.input_generator_func = input_generator_func
        self.input_generator_args = input_generator_args
        self.num_workers = num_workers if num_workers > 0 else 1
        self.max_concurrency_per_process = max_concurrency_per_process
        self.model = model
        if api_key is None:
            api_key = os.getenv("ARK_API_KEY")
            if not api_key:
                raise ValueError("请设置 ARK_API_KEY 环境变量")
        self.api_key = api_key

    def run(self, output_handler: Callable[[Dict[str, Any]], None]) -> None:
        print("🚀 启动 Doubao 批量处理器")
        print(f"   工作进程数: {self.num_workers}")
        print(f"   每进程并发数: {self.max_concurrency_per_process}")

        manager = multiprocessing.Manager()
        in_queue: multiprocessing.Queue[Optional[Dict[str, Any]]] = manager.Queue(maxsize=1024)
        out_queue: multiprocessing.Queue[Optional[Dict[str, Any]]] = manager.Queue(maxsize=1024)

        processes = []

        p_in_args = (self.input_generator_func, self.input_generator_args, in_queue, self.num_workers)
        p_in = multiprocessing.Process(target=self._input_producer, args=p_in_args)
        p_in.start()
        processes.append(p_in)

        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=self._worker_process,
                args=(i, self.max_concurrency_per_process, self.api_key, in_queue, out_queue),
            )
            p.start()
            processes.append(p)

        finished_workers = 0
        try:
            while finished_workers < self.num_workers:
                try:
                    result = out_queue.get(timeout=600)
                    if result is None:
                        finished_workers += 1
                        continue
                    output_handler(result)
                except (queue.Empty, EOFError):
                    alive_workers = sum(1 for p in processes[1:] if p.is_alive())
                    if alive_workers == 0:
                        break
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        finally:
            print("⏳ 正在清理和关闭进程...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join(timeout=5)
            print("✅ 所有任务完成！")

    @staticmethod
    def _input_producer(generator_func, generator_args, in_queue, num_workers):
        try:
            records = list(generator_func(**generator_args))
            print(f"📋 总共需要处理 {len(records)} 个 chunk 任务")
            for record in tqdm(records, desc="分发任务"):
                try:
                    in_queue.put(record, block=True, timeout=300)
                except Exception as e:
                    print(f"❌ 任务入队失败: {e}")
                    break
        except Exception as e:
            print(f"❌ 输入生产者出错: {e}")
        finally:
            for _ in range(num_workers):
                in_queue.put(None)

    def _worker_process(self, worker_id, max_concurrency, api_key, in_queue, out_queue):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = AsyncArk(api_key=api_key, http_client=self._make_client(max_concurrency), timeout=7200)

            async def work():
                sem = asyncio.Semaphore(max_concurrency)
                tasks = []
                while True:
                    try:
                        record = await asyncio.to_thread(in_queue.get, block=True, timeout=60)
                        if record is None:
                            break
                        if "model" not in record:
                            record["model"] = self.model
                        await sem.acquire()
                        task = loop.create_task(self._process_task(client, record, out_queue, sem))
                        tasks.append(task)
                    except (queue.Empty, EOFError):
                        break
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await client._client.aclose()
            loop.run_until_complete(work())
        except Exception as e:
            print(f"❌ Worker {worker_id} 异常: {e}")
        finally:
            out_queue.put(None)

    @staticmethod
    async def _process_task(client, record, out_queue, sem):
        api_params = record.copy()
        input_path = api_params.pop("input_path", "unknown")
        custom_id = api_params.pop("custom_id", "")  # ✅ 保留 custom_id
        extra_headers = api_params.pop("extra_headers", {})

        try:
            response = await client.chat.completions.create(**api_params, extra_headers=extra_headers)
            result_dict = response.to_dict()
            result_dict["input_path"] = input_path
            result_dict["custom_id"] = custom_id  # ✅ 确保传回
            result_dict["extra_headers"] = extra_headers
            await asyncio.to_thread(out_queue.put, result_dict)
        except Exception as e:
            error_result = {
                "error": str(e),
                "input_path": input_path,
                "custom_id": custom_id,
                "extra_headers": extra_headers,
                "input": record.get("messages", []),
            }
            await asyncio.to_thread(out_queue.put, error_result)
        finally:
            sem.release()

    @staticmethod
    def _make_client(max_concurrency):
        return httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency),
            timeout=httpx.Timeout(7200.0)
        )


# ==================== 结果处理器（从 custom_id 解析 chunk_index）====================

class ResultHandler:
    def __init__(self, log_file_path: str, log_cache: Dict, progress_bar: tqdm):
        self.log_file_path = log_file_path
        self.log_cache = log_cache
        self.progress_bar = progress_bar
        self.file_name_to_chunks: Dict[str, List[List[str]]] = {}
        self.file_name_to_full_content: Dict[str, str] = {}

    def handle(self, result: Dict[str, Any]):
        input_path = result.get("input_path", "unknown")
        custom_id = result.get("custom_id", "unknown::chunk_unknown")

        # 解析 custom_id 获取 chunk_index
        if "::chunk_" in custom_id:
            try:
                _, chunk_index = custom_id.rsplit("::", 1)
            except:
                chunk_index = "unknown"
        else:
            chunk_index = "0"

        is_success = False
        error_msg = ""
        extracted_names = []

        if input_path == "unknown":
            error_msg = "无效 input_path"
        elif "error" in result:
            error_msg = str(result['error'])
        else:
            try:
                resp = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if resp.startswith("```json"):
                    resp = resp[7:-3].strip()
                names = json.loads(resp)
                if isinstance(names, list):
                    extracted_names = [n.strip() for n in names if isinstance(n, str) and 2 <= len(n.strip()) <= 10]
                # 缓存到原始路径
                if input_path not in self.file_name_to_chunks:
                    self.file_name_to_chunks[input_path] = []
                self.file_name_to_chunks[input_path].append(extracted_names)
                is_success = True
            except Exception as e:
                error_msg = f"解析模型响应失败: {e}, 原始: {resp[:150]}..."

        # 记录 chunk 日志
        log_entry = {
            "input_path": input_path,  # ✅ 完整路径
            "chunk_index": chunk_index,
            "is_success": is_success,
            "error_message": error_msg,
            "extracted_names_count": len(extracted_names),
        }
        update_or_append_to_jsonl(self.log_file_path, log_entry, self.log_cache)

        # 更新进度条
        if self.progress_bar:
            self.progress_bar.update(1)

        # ✅ 打印处理状态
        status = "✅" if is_success else "❌"
        print(f"{status} [{input_path}][块{chunk_index}] 姓名数: {len(extracted_names)}{f' | 错误: {error_msg}' if error_msg else ''}")

        # 尝试合并
        self._try_merge_file(input_path)

    def _try_merge_file(self, input_path: str):
        if input_path not in self.file_name_to_chunks:
            return

        if input_path not in self.file_name_to_full_content:
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    self.file_name_to_full_content[input_path] = f.read()
                print(f"📄 已加载全文: {input_path}")
            except Exception as e:
                print(f"❌ 读取文件失败: {input_path}, 错误: {e}")
                return

        full_content = self.file_name_to_full_content[input_path]

        # ✅ 分别提取手机号、固话、身份证
        extracted_mobiles = list(set(extract_mobiles(full_content)))
        extracted_phones = list(set(extract_phones(full_content)))
        extracted_id_cards = list(set(extract_id_cards(full_content)))

        # 模型提取的姓名（已缓存）
        model_names = [name for chunk in self.file_name_to_chunks[input_path] for name in chunk]
        extracted_names = list(set(model_names))  # 去重

        # 合并所有敏感项用于脱敏
        all_sensitive = list(set(extracted_names + extracted_mobiles + extracted_phones + extracted_id_cards))

        try:
            desensitized = desensitize_content(full_content, all_sensitive)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(desensitized)

            # ✅ 构建最终日志，包含详细提取信息
            final_log = {
                "input_path": input_path,  # ✅ 完整路径
                "is_success": True,
                "error_message": "",
                "total_sensitive_count": len(all_sensitive),
                "extracted_names_count": len(extracted_names),
                "rule_based_count": len(extracted_mobiles) + len(extracted_phones) + len(extracted_id_cards),
                "extracted_names": extracted_names,           # ✅ 模型提取的姓名
                "extracted_mobiles": extracted_mobiles,       # ✅ 手机号
                "extracted_phones": extracted_phones,         # ✅ 固话
                "extracted_id_cards": extracted_id_cards,     # ✅ 身份证
                "timestamp": datetime.now().isoformat(),      # 可选：加时间戳
            }
            update_or_append_to_jsonl(self.log_file_path, final_log, self.log_cache)
            print(f"📄【完成】{input_path} 脱敏写回")
            print(f"   姓名: {extracted_names}")
            print(f"   手机号: {extracted_mobiles}")
            print(f"   固话: {extracted_phones}")
            print(f"   身份证: {extracted_id_cards}")

        except Exception as e:
            print(f"❌ 写入失败: {input_path}, 错误: {e}")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # ==================== 配置信息 ====================
    DEBUG_MODE = False  # ✅ 控制是否打印详细日志
    DEBUG_COUNT = 10   # DEBUG 模式下处理的文件数上限
    MD_ROOT_DIR = "/mnt/data/zzj/data_clean_fire/data/output"
    NUM_WORKERS = 32
    MAX_CONCURRENCY_PER_PROCESS = 64
    LOG_FILE_PATH = os.path.join(MD_ROOT_DIR, "desensitize_results.jsonl")
    MODEL_NAME = "doubao-seed-1-6-flash-250715"

    print(f"📂 处理目录: {MD_ROOT_DIR}")
    if not os.path.isdir(MD_ROOT_DIR):
        print(f"❌ 目录不存在: {MD_ROOT_DIR}")
        exit(1)

    log_cache = load_existing_results_full(LOG_FILE_PATH)
    all_md_files = set(glob.glob(os.path.join(MD_ROOT_DIR, "**/*.md"), recursive=True))
    done_paths = {p for p, e in log_cache.items() if e.get("is_success", False)}
    todo_files = all_md_files - done_paths

    print(f"📚 已加载 {len(log_cache)} 条历史日志")
    print(f"✅ 已成功处理: {len(done_paths)} 个文件")
    print(f"📌 待处理: {len(todo_files)} 个文件")

    # ✅ DEBUG 模式下只处理前 N 个文件
    if DEBUG_MODE:
        todo_files = set(sorted(list(todo_files))[:DEBUG_COUNT])
        print(f"🐞 DEBUG 模式: 只处理前 {len(todo_files)} 个文件")

    if not todo_files:
        print("🎉 所有文件均已处理完毕，无需执行。")
        exit(0)

    with tqdm(total=len(todo_files), desc="处理进度", unit="文件") as pbar:
        handler = ResultHandler(LOG_FILE_PATH, log_cache, pbar)
        processor = DoubaoBatchProcessor(
            input_generator_func=md_input_generator_for_desensitization,
            input_generator_args={
                "md_root_dir": MD_ROOT_DIR,
                "existing_log_cache": log_cache,
                "target_files": todo_files
            },
            num_workers=NUM_WORKERS,
            max_concurrency_per_process=MAX_CONCURRENCY_PER_PROCESS,
            model=MODEL_NAME,
            api_key=os.getenv("ARK_API_KEY")
        )
        processor.run(output_handler=handler.handle)

    print(f"✅ 批量脱敏处理完成！日志已保存至: {LOG_FILE_PATH}")