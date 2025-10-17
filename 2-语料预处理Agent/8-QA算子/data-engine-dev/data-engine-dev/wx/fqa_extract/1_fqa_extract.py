import json
import time
import argparse
import random
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import concurrent.futures as cf

# ========= Ark 配置 =========
API_KEY  = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

# ========= Embedding 配置 =========
EMBEDDING_API_URLS = [
    "http://localhost:7100/v1/embeddings",
    "http://localhost:7101/v1/embeddings", 
    "http://localhost:7102/v1/embeddings",
    "http://localhost:7103/v1/embeddings",
    "http://localhost:7104/v1/embeddings",
    "http://localhost:7105/v1/embeddings",
    "http://localhost:7106/v1/embeddings",
    "http://localhost:7107/v1/embeddings"
]
EMBEDDING_MODEL_NAME = "qwen3-8b-embd"
SIMILARITY_THRESHOLD = 0.99

try:
    from volcenginesdkarkruntime import Ark
except Exception:
    raise SystemExit("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")


try:
    from docx import Document
except Exception:
    raise SystemExit("❌ 请先安装 python-docx: pip install python-docx")


# ========= 参数常量 =========
DEFAULT_WINDOW_LINES = 80    # 稍微保守，减少长窗超时
DEFAULT_STRIDE_LINES = 40
SUPPORTED_TYPES = {"single", "multiple", "judge", "fill"}

# 单次请求最大字符预算（避免超大窗口导致超时/超长）
MAX_CHARS_PER_PROMPT = 9000
# Ark 默认请求超时（秒）
DEFAULT_ARK_TIMEOUT = 120

# ========= System/User Prompt =========
SYSTEM_PROMPT = """\
你是一个精确的问答对抽取助手。
我将给你一段包含问答和参考信息的文本片段。
你的任务：抽取其中的问答对和对应的参考信息。

文本可能包含多种格式：
1) 标准问答格式：明确的问题和答案
2) 采访对话格式：A问B答的对话形式
3) 教学问答格式：老师提问学生回答
4) 其他对话形式：包含问答内容的对话

抽取规则：
1) 每个问答对必须包含：
   - "id"：在当前片段内的局部递增编号（字符串或数字）
   - "question"：问题的完整文字（可能是直接提问，也可能是对话中的询问）
   - "answer"：对应的答案文字（可能是直接回答，也可能是对话中的回应）
   - "reference"：相关的参考信息、来源或上下文（如无则为空字符串）
2) 对于对话格式：
   - 将对话中的询问识别为问题
   - 将对话中的回应识别为答案
   - 保持对话的原始语境和完整性
3) 只输出完整的问答对：必须有完整的问题和答案。
4) 如果问答对被截断或不完整，则不要输出。
5) 严格输出 JSON 格式，UTF-8 编码，不要包含 Markdown、代码块围栏或多余文字。
6) 不要编造内容。参考信息缺失时用空字符串表示。
7) 如果片段中没有符合条件的问答对，返回空数组 []。

输出格式：
JSON 数组，数组中每个元素是一个问答对对象。
"""

CHUNK_PROMPT_TEMPLATE = """\
下面是全文的第 {start_line} 行到第 {end_line} 行的文本内容。
请仔细分析文本结构，提取其中符合要求且完整的问答对。

--- 开始片段 ---
{chunk_text}
--- 结束片段 ---

请记住：
- 识别各种格式的问答：标准问答、对话形式、采访形式等
- 对于对话格式，将询问识别为问题，回应识别为答案
- 保持问答的完整性和原始语境
- 只返回严格 JSON 数组，不要输出任何额外文字
- 每个问答对必须包含完整的问题和答案
"""

# ========= 数据结构 =========
@dataclass
class QAItem:
    qid: Optional[str]
    question: str
    answer: str
    reference: str
    source_window: Tuple[int, int]
    window_local_id: Optional[str]
    source_file: str = ""

@dataclass
class ExtractStats:
    windows: int = 0
    raw_questions: int = 0
    kept_questions: int = 0

@dataclass
class ClusterLog:
    cluster_id: int
    representative_question: str
    cluster_size: int
    similarity_threshold: float
    kept_item: QAItem
    merged_items: List[QAItem]
    similarities: List[float]
    all_cluster_items: List[QAItem]  # 聚类前的所有问题
    answer_groups: Dict[str, List[Tuple[int, str]]]  # 答案分组信息

# ========= Embedding 模型 =========
class EmbeddingModel:
    def __init__(self, api_urls: List[str], model_name: str):
        self.api_urls = api_urls
        self.model_name = model_name
        self.working_urls = []
        self._test_apis()
    
    def _test_apis(self):
        """测试API连接"""
        print(f"🔄 测试embedding API连接: {len(self.api_urls)} 个端点")
        for i, url in enumerate(self.api_urls):
            try:
                test_response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "input": "测试",
                        "model": self.model_name
                    },
                    timeout=10
                )
                test_response.raise_for_status()
                print(f"  ✅ 端点 {i+1} 测试成功: {url}")
                self.working_urls.append(url)
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 端点 {i+1} 测试失败: {url} - {e}")
        
        if not self.working_urls:
            raise Exception("❌ 所有embedding API端点都连接失败，请确保embedding服务正在运行")
        
        print(f"✅ 成功连接 {len(self.working_urls)} 个embedding API端点")
    
    def get_embedding(self, text: str, url_index: int = 0) -> List[float]:
        """获取单个文本的embedding，支持轮询多个API端点"""
        if not text.strip():
            return []
        
        # 轮询尝试所有API端点
        for attempt in range(len(self.working_urls)):
            current_url_index = (url_index + attempt) % len(self.working_urls)
            api_url = self.working_urls[current_url_index]
            
            try:
                response = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "input": text,
                        "model": self.model_name
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['embedding']
                else:
                    print(f"API响应格式错误: {result}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"API调用失败 (端口 {api_url.split(':')[-1]}): {e}")
                continue
        
        print(f"所有API端点都调用失败，文本: {text[:50]}...")
        return []
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本为embedding向量"""
        print(f"🔤 编码 {len(texts)} 个问题文本...")
        embeddings = []
        
        for i, text in enumerate(tqdm(texts, desc="编码文本", ncols=100)):
            if i % 10 == 0:  # 每10个显示一次进度
                print(f"  处理第 {i+1}/{len(texts)} 个文本...")
            
            # 轮询使用不同的API端点
            url_index = i % len(self.working_urls)
            embedding = self.get_embedding(text, url_index)
            embeddings.append(embedding)
        
        # 过滤掉空的embedding
        valid_embeddings = [emb for emb in embeddings if emb]
        if len(valid_embeddings) != len(embeddings):
            print(f"⚠️  警告：{len(embeddings) - len(valid_embeddings)} 个文本的embedding获取失败")
        
        return np.array(valid_embeddings)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算embedding之间的余弦相似度矩阵"""
        # 归一化embedding向量
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(normalized_embeddings)
        return similarity_matrix

# 全局embedding模型实例
_embedding_model = None

def get_embedding_model() -> EmbeddingModel:
    """获取全局embedding模型实例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(EMBEDDING_API_URLS, EMBEDDING_MODEL_NAME)
    return _embedding_model

# ========= Ark 调用 =========
def call_ark(prompt: str, api_key: str, model_id: str,
             temperature: float = 0.0, top_p: float = 0.9,
             timeout: int = DEFAULT_ARK_TIMEOUT) -> str:
    time.sleep(random.uniform(0.05, 0.20))  # 轻微抖动
    client = Ark(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

# ========= JSON 解析 + 清洗 =========
_HEX = set("0123456789abcdefABCDEF")

def _fix_invalid_backslashes(s: str) -> str:
    """
    修复模型返回中的非法反斜杠：对非合法转义前的反斜杠进行二次转义。
    仅用于让 JSON 可被解析，不改动其他内容。
    """
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != '\\':
            out.append(ch)
            i += 1
            continue
        # ch is backslash
        if i + 1 >= n:
            out.append('\\\\'); i += 1; continue
        nxt = s[i+1]
        if nxt in '"\\/bfnrt':
            out.append('\\' + nxt); i += 2; continue
        if nxt == 'u':
            # \uXXXX
            if i + 5 < n and all(c in _HEX for c in s[i+2:i+6]):
                out.append(s[i:i+6]); i += 6; continue
            else:
                out.append('\\\\u'); i += 2; continue
        # 非法转义 -> 双反斜杠
        out.append('\\\\' + nxt)
        i += 2
    return ''.join(out)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    fences = ("```json", "```JSON", "```", "~~~json", "~~~JSON", "~~~")
    for f in fences:
        if s.startswith(f):
            s = s[len(f):].strip()
        if s.endswith(f):
            s = s[:-len(f)].strip()
    return s

def force_json_load_with_sanitize(s: str) -> Tuple[Any, bool]:
    """
    返回 (obj, sanitized_used)
    先按常规路径解析；失败后：
      - 截取首个 [ 到最后一个 ] 的核心
      - 去代码围栏
      - 修复非法反斜杠
    """
    s0 = s.strip()
    try:
        if s0.startswith('[') and s0.endswith(']'):
            return json.loads(s0), False
    except Exception:
        pass

    first = s0.find('['); last = s0.rfind(']')
    core = s0 if (first == -1 or last == -1 or first >= last) else s0[first:last+1]
    core = _strip_code_fences(core)

    # 第一次尝试
    try:
        return json.loads(core), False
    except Exception:
        pass

    # 清洗非法 \
    fixed = _fix_invalid_backslashes(core)
    return json.loads(fixed), True  # 如果仍异常，会抛给上层

# ========= 工具 =========
def norm_text(s: str) -> str:
    return " ".join(s.replace("\u3000", " ").replace("\r", "").split())

def clean_text_lines(text: str, max_consecutive_empty: int = 1) -> str:
    """
    清理文本中的多余空行
    Args:
        text: 输入文本
        max_consecutive_empty: 最大连续空行数，默认为1
    """
    lines = text.splitlines()
    cleaned_lines = []
    empty_count = 0
    
    for line in lines:
        is_empty = not line.strip()
        if is_empty:
            empty_count += 1
            if empty_count <= max_consecutive_empty:
                cleaned_lines.append("")
        else:
            empty_count = 0
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def read_word_document(docx_path: Path) -> str:
    """读取Word文档并返回文本内容"""
    try:
        doc = Document(docx_path)
        text_lines = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # 只添加非空段落
                text_lines.append(paragraph.text.strip())
        return "\n".join(text_lines)
    except Exception as e:
        raise Exception(f"读取Word文档失败: {e}")

def read_markdown_document(md_path: Path) -> str:
    """读取Markdown文档并返回文本内容，删除多余空行"""
    try:
        content = md_path.read_text(encoding="utf-8", errors="ignore")
        # 使用通用清理函数删除多余空行
        return clean_text_lines(content, max_consecutive_empty=1)
    except Exception as e:
        raise Exception(f"读取Markdown文档失败: {e}")

def read_document(file_path: Path) -> str:
    """根据文件类型读取文档内容"""
    suffix = file_path.suffix.lower()
    if suffix in ['.docx', '.doc']:
        return read_word_document(file_path)
    elif suffix in ['.md', '.markdown', '.txt']:
        return read_markdown_document(file_path)
    else:
        raise Exception(f"不支持的文件格式: {suffix}")

def make_windows(lines: List[str], window: int, stride: int) -> List[Tuple[int, int, str]]:
    n = len(lines); out = []; i = 0
    while i < n:
        s = i; e = min(i + window - 1, n - 1)
        out.append((s+1, e+1, "\n".join(lines[s:e+1])))
        if e == n - 1: break
        i += stride
    return out

def make_char_windows(text: str, window_chars: int, stride_chars: int) -> List[Tuple[int, int, str]]:
    """
    按字符数分割文本，创建滑动窗口
    Args:
        text: 输入文本
        window_chars: 窗口字符数
        stride_chars: 滑动步长字符数
    Returns:
        List of (start_pos, end_pos, window_text)
    """
    if not text:
        return []
    
    windows = []
    text_len = len(text)
    start = 0
    
    while start < text_len:
        # 计算窗口结束位置
        end = min(start + window_chars, text_len)
        
        # 获取窗口文本
        window_text = text[start:end]
        
        # 计算行号范围（近似）
        lines_before = text[:start].count('\n')
        lines_in_window = window_text.count('\n')
        start_line = lines_before + 1
        end_line = lines_before + lines_in_window + 1
        
        windows.append((start_line, end_line, window_text))
        
        # 如果已经到达文本末尾，退出
        if end >= text_len:
            break
            
        # 计算下一个窗口的起始位置
        start += stride_chars
    
    return windows

def compress_chunk_text(s: str, max_chars: int = MAX_CHARS_PER_PROMPT) -> str:
    lines = s.replace("\r", "").splitlines()
    out = []
    for ln in lines:
        ln = " ".join(ln.split())
        if ln:  # 只保留非空行
            out.append(ln)
    s2 = "\n".join(out).strip()
    return s2[:max_chars] if len(s2) > max_chars else s2

# ========= Debug 落盘 =========
def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)

# ========= 单窗口自适应调用（带日志） =========
def send_chunk_adaptive(s_line: int, e_line: int, chunk_text: str,
                        max_retries: int, retry_backoff: float, timeout: int,
                        debug_dir: Optional[Path], base_norm: str):
    """
    返回：raw_out, data(list), meta(dict)
    meta: {prompt_len, response_len, parse_error, sanitized_used, compressed_chunk}
    """
    compressed = compress_chunk_text(chunk_text, MAX_CHARS_PER_PROMPT)
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        start_line=s_line, end_line=e_line, chunk_text=compressed
    )
    meta = {
        "prompt_len": len(prompt), "response_len": 0,
        "parse_error": "", "sanitized_used": False,
        "compressed_chunk": compressed
    }

    # 调试目录
    wdir = None
    if debug_dir:
        wdir = debug_dir / base_norm / f"{s_line}-{e_line}"
        write_text(wdir / "prompt.txt", prompt)  # 送给模型的完整用户消息（含片段）

    # 直接重试
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            raw = call_ark(prompt, api_key=API_KEY, model_id=MODEL_ID, timeout=timeout)
            if debug_dir:
                write_text(wdir / "response.txt", raw)
            meta["response_len"] = len(raw)
            obj, sanitized_used = force_json_load_with_sanitize(raw)
            meta["sanitized_used"] = sanitized_used
            if debug_dir and sanitized_used:
                # 保存清洗后的 JSON 文本
                write_text(wdir / "sanitized.json", json.dumps(obj, ensure_ascii=False, indent=2))
            return raw, obj, meta
        except Exception as e:
            last_err = str(e)
            meta["parse_error"] = last_err
            if debug_dir:
                write_text(wdir / "parse_error.txt", last_err)
            if attempt == max_retries:
                break
            time.sleep((retry_backoff ** attempt) + random.uniform(0.1, 0.5))
            attempt += 1

    # 实在失败
    return f"<<FAILED {s_line}-{e_line}>> {last_err}", [], meta

# ========= 保存原文片段对照 =========
def save_raw_chunks(raw_records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ========= 去重逻辑 =========
@dataclass
class DedupLog:
    question: str
    total_count: int
    duplicate_groups: List[Dict[str, Any]]
    kept_item: QAItem
    removed_items: List[QAItem]

def deduplicate_qa_items(items: List[QAItem], log_file: Optional[Path] = None) -> Tuple[List[QAItem], List[DedupLog]]:
    """
    对问答对进行去重处理
    Args:
        items: 问答对列表
        log_file: 去重日志文件路径（可选）
    Returns:
        (去重后的问答对列表, 去重日志列表)
    """
    print("🔄 开始去重处理...")
    
    # 按question分组
    question_groups = {}
    for item in items:
        question_key = norm_text(item.question)
        if question_key not in question_groups:
            question_groups[question_key] = []
        question_groups[question_key].append(item)
    
    print(f"📊 原始题目数：{len(items)}，去重前分组数：{len(question_groups)}")
    
    deduped_items = []
    dedup_logs = []
    total_removed = 0
    
    for question_key, group_items in question_groups.items():
        if len(group_items) == 1:
            # 只有一个，直接保留
            deduped_items.append(group_items[0])
            continue
        
        # 多个相同问题的，需要比较答案
        # 按answer前3个字符分组
        answer_groups = {}
        for item in group_items:
            answer_prefix = item.answer[:3] if len(item.answer) >= 3 else item.answer
            if answer_prefix not in answer_groups:
                answer_groups[answer_prefix] = []
            answer_groups[answer_prefix].append(item)
        
        # 处理每个答案前缀组
        duplicate_groups = []
        kept_item = None
        removed_items = []
        
        for answer_prefix, answer_items in answer_groups.items():
            if len(answer_items) == 1:
                # 该答案前缀只有一个，直接保留
                if kept_item is None or len(answer_items[0].answer) > len(kept_item.answer):
                    if kept_item is not None:
                        removed_items.append(kept_item)
                    kept_item = answer_items[0]
                else:
                    removed_items.append(answer_items[0])
            else:
                # 该答案前缀有多个，保留最长的
                # 按答案长度排序，保留最长的
                answer_items.sort(key=lambda x: len(x.answer), reverse=True)
                kept_in_group = answer_items[0]
                removed_in_group = answer_items[1:]
                
                if kept_item is None or len(kept_in_group.answer) > len(kept_item.answer):
                    if kept_item is not None:
                        removed_items.append(kept_item)
                    kept_item = kept_in_group
                else:
                    removed_items.append(kept_in_group)
                
                removed_items.extend(removed_in_group)
                
                duplicate_groups.append({
                    "answer_prefix": answer_prefix,
                    "count": len(answer_items),
                    "kept_item": {
                        "qid": kept_in_group.qid,
                        "answer_length": len(kept_in_group.answer),
                        "answer_preview": kept_in_group.answer[:100]
                    },
                    "removed_items": [
                        {
                            "qid": item.qid,
                            "answer_length": len(item.answer),
                            "answer_preview": item.answer[:100]
                        } for item in removed_in_group
                    ]
                })
        
        # 记录去重日志
        if duplicate_groups or len(removed_items) > 0:
            dedup_log = DedupLog(
                question=question_key,
                total_count=len(group_items),
                duplicate_groups=duplicate_groups,
                kept_item=kept_item,
                removed_items=removed_items
            )
            dedup_logs.append(dedup_log)
            total_removed += len(removed_items)
        
        deduped_items.append(kept_item)
    
    print(f"🎯 去重完成：原始 {len(items)} 题 → 去重后 {len(deduped_items)} 题，删除 {total_removed} 个重复项")
    
    # 保存去重日志
    if log_file and dedup_logs:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            f.write("# 去重处理日志\n\n")
            f.write(f"## 总体统计\n")
            f.write(f"- 原始题目数：{len(items)}\n")
            f.write(f"- 去重后题目数：{len(deduped_items)}\n")
            f.write(f"- 删除重复项数：{total_removed}\n")
            f.write(f"- 涉及重复的问题数：{len(dedup_logs)}\n\n")
            
            for i, log in enumerate(dedup_logs, 1):
                f.write(f"## 重复问题 {i}\n")
                f.write(f"**问题：** {log.question[:100]}...\n")
                f.write(f"**保留：** {log.kept_item.qid} (答案长度: {len(log.kept_item.answer)})\n")
                f.write(f"**删除：** {len(log.removed_items)} 个重复项\n\n")
        
        print(f"📝 去重日志已保存：{log_file}")
    
    return deduped_items, dedup_logs

# ========= 相似度聚类去重 =========
def cluster_and_deduplicate_questions(items: List[QAItem], similarity_threshold: float = SIMILARITY_THRESHOLD, 
                                    log_file: Optional[Path] = None) -> Tuple[List[QAItem], List[ClusterLog]]:
    """
    基于embedding相似度对问题进行聚类，然后在聚类内进行answer去重
    Args:
        items: 问答对列表
        similarity_threshold: 相似度阈值
        log_file: 聚类日志文件路径（可选）
    Returns:
        (聚类和去重后的问答对列表, 聚类日志列表)
    """
    if not items:
        return items, []
    
    print(f"🔄 开始相似度聚类和去重，相似度阈值：{similarity_threshold}")
    print(f"📊 输入题目数：{len(items)}")
    
    # 获取embedding模型
    embedding_model = get_embedding_model()
    
    # 提取所有问题文本
    questions = [item.question for item in items]
    
    # 编码所有问题
    embeddings = embedding_model.encode_texts(questions)
    
    # 计算相似度矩阵
    print("📐 计算相似度矩阵...")
    similarity_matrix = embedding_model.compute_similarity_matrix(embeddings)
    
    # 聚类算法：使用相似度阈值进行聚类
    print("🎯 执行聚类算法...")
    clusters = []
    visited = set()
    
    for i in range(len(items)):
        if i in visited:
            continue
        
        # 创建新聚类
        cluster = [i]
        visited.add(i)
        
        # 找到所有与当前问题相似的问题
        for j in range(i + 1, len(items)):
            if j in visited:
                continue
            
            similarity = similarity_matrix[i][j]
            if similarity >= similarity_threshold:
                cluster.append(j)
                visited.add(j)
        
        clusters.append(cluster)
    
    print(f"📊 聚类结果：{len(clusters)} 个聚类")
    
    # 处理每个聚类：选择代表问题，然后对答案进行去重
    final_items = []
    cluster_logs = []
    total_merged = 0
    
    for cluster_id, cluster_indices in enumerate(clusters):
        if len(cluster_indices) == 1:
            # 单元素聚类，直接保留
            final_items.append(items[cluster_indices[0]])
            continue
        
        # 多元素聚类，需要选择代表问题并对答案去重
        cluster_items = [items[i] for i in cluster_indices]
        
        # 1. 选择代表问题（选择问题文本最长的，或者第一个）
        representative_item = max(cluster_items, key=lambda x: len(x.question))
        
        # 2. 收集聚类内所有答案
        all_answers = [item.answer for item in cluster_items]
        
        # 3. 对答案进行去重处理（按前3个字符分组）
        answer_groups = {}
        for i, answer in enumerate(all_answers):
            answer_prefix = answer[:3] if len(answer) >= 3 else answer
            if answer_prefix not in answer_groups:
                answer_groups[answer_prefix] = []
            answer_groups[answer_prefix].append((i, answer))
        
        # 4. 在每个答案前缀组内选择最长的答案
        final_answer = None
        merged_items = []
        similarities = []
        
        for answer_prefix, answer_list in answer_groups.items():
            if len(answer_list) == 1:
                # 该答案前缀只有一个，直接使用
                if final_answer is None or len(answer_list[0][1]) > len(final_answer):
                    if final_answer is not None:
                        # 找到被替换的答案对应的item
                        old_item_idx = next(i for i, item in enumerate(cluster_items) 
                                          if item.answer == final_answer)
                        merged_items.append(cluster_items[old_item_idx])
                    final_answer = answer_list[0][1]
                else:
                    # 当前答案较短，标记为合并
                    item_idx = answer_list[0][0]
                    merged_items.append(cluster_items[item_idx])
            else:
                # 该答案前缀有多个，选择最长的
                # 按答案长度排序
                answer_list.sort(key=lambda x: len(x[1]), reverse=True)
                best_answer = answer_list[0][1]
                other_answers = answer_list[1:]
                
                if final_answer is None or len(best_answer) > len(final_answer):
                    if final_answer is not None:
                        # 找到被替换的答案对应的item
                        old_item_idx = next(i for i, item in enumerate(cluster_items) 
                                          if item.answer == final_answer)
                        merged_items.append(cluster_items[old_item_idx])
                    final_answer = best_answer
                else:
                    # 当前最佳答案较短，标记为合并
                    item_idx = answer_list[0][0]
                    merged_items.append(cluster_items[item_idx])
                
                # 标记其他答案为合并
                for item_idx, _ in other_answers:
                    merged_items.append(cluster_items[item_idx])
        
        # 5. 创建最终的QAItem
        final_item = QAItem(
            qid=representative_item.qid,
            question=representative_item.question,
            answer=final_answer,
            reference=representative_item.reference,
            source_window=representative_item.source_window,
            window_local_id=representative_item.window_local_id,
            source_file=representative_item.source_file
        )
        
        # 6. 计算相似度（被合并的item与代表问题之间的相似度）
        representative_idx = next(i for i, item in enumerate(items) if item == representative_item)
        for merged_item in merged_items:
            merged_idx = next(i for i, item in enumerate(items) if item == merged_item)
            similarity = similarity_matrix[representative_idx][merged_idx]
            similarities.append(similarity)
        
        # 7. 记录聚类日志 - 只保留聚类大小>1或答案数量>1的聚类
        if len(cluster_items) > 1 or len(answer_groups) > 1:
            cluster_log = ClusterLog(
                cluster_id=cluster_id + 1,
                representative_question=representative_item.question,
                cluster_size=len(cluster_items),
                similarity_threshold=similarity_threshold,
                kept_item=final_item,
                merged_items=merged_items,
                similarities=similarities,
                all_cluster_items=cluster_items,  # 聚类前的所有问题
                answer_groups=answer_groups  # 答案分组信息
            )
            cluster_logs.append(cluster_log)
            total_merged += len(merged_items)
        
        final_items.append(final_item)
    
    print(f"🎯 聚类和去重完成：原始 {len(items)} 题 → 最终 {len(final_items)} 题，合并 {total_merged} 个相似问题")
    
    # 保存聚类日志
    if log_file and cluster_logs:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            f.write("# 相似度聚类和去重日志\n\n")
            f.write(f"## 总体统计\n")
            f.write(f"- 原始题目数：{len(items)}\n")
            f.write(f"- 最终题目数：{len(final_items)}\n")
            f.write(f"- 合并相似问题数：{total_merged}\n")
            f.write(f"- 聚类数量：{len(clusters)}\n")
            f.write(f"- 相似度阈值：{similarity_threshold}\n\n")
            
            for log in cluster_logs:
                f.write(f"## 聚类 {log.cluster_id}\n")
                f.write(f"**聚类大小：** {log.cluster_size}\n")
                f.write(f"**代表问题：** {log.representative_question}...\n")
                f.write(f"**保留项：** {log.kept_item.qid} (答案长度: {len(log.kept_item.answer)})\n")
                f.write(f"**合并项：** {len(log.merged_items)} 个\n")
                
                # 显示聚类前的所有问题和答案对比（简化版）
                f.write(f"\n**聚类前的问题和答案：**\n")
                for i, item in enumerate(log.all_cluster_items, 1):
                    f.write(f"{i}. {item.qid}: {item.question}...\n")
                    f.write(f"   答案: {item.answer}... (长度: {len(item.answer)})\n")
                
                f.write(f"\n**相似度：** {', '.join([f'{s:.3f}' for s in log.similarities])}\n")
                f.write(f"**答案分组：** {len(log.answer_groups)} 个不同答案前缀\n\n")
                f.write("---\n\n")
        
        print(f"📝 聚类日志已保存：{log_file}")
    
    return final_items, cluster_logs

# ========= 抽取单文件 =========
def extract_from_md(md_path: Path, window: int, stride: int,
                    max_retries: int, retry_backoff: float, timeout: int,
                    source_file_stem: str,
                    debug_dir: Optional[Path]) -> Tuple[List[QAItem], ExtractStats, List[Dict[str, Any]]]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    windows = make_windows(lines, window=window, stride=stride)

    all_items: List[QAItem] = []
    stats = ExtractStats(windows=len(windows))
    raw_records: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(windows), desc=f"抽取 {md_path.name}", ncols=100)
    for (s_line, e_line, chunk) in windows:
        raw_out, data, meta = send_chunk_adaptive(
            s_line=s_line, e_line=e_line, chunk_text=chunk,
            max_retries=max_retries, retry_backoff=retry_backoff, timeout=timeout,
            debug_dir=debug_dir, base_norm=source_file_stem
        )

        raw_records.append({
            "window": f"{s_line}-{e_line}",
            "chunk_text": chunk,
            "compressed_chunk": meta["compressed_chunk"],
            "prompt_length": meta["prompt_len"],
            "model_output": raw_out,
            "response_length": meta["response_len"],
            "sanitized_used": meta["sanitized_used"],
            "parse_error": meta["parse_error"],
            "parsed_questions": data
        })

        stats.raw_questions += len(data)
        for obj in data:
            question = norm_text(str(obj.get("question") or ""))
            answer = norm_text(str(obj.get("answer") or ""))
            reference = norm_text(str(obj.get("reference") or ""))

            # ======= 关键改动：答案为空直接剔除 =======
            if not answer:
                continue
            # =======================================

            window_position = (s_line, e_line)
            window_range = f"{s_line}-{e_line}"

            all_items.append(QAItem(
                qid=None, question=question, answer=answer, reference=reference,
                source_window=(s_line, e_line), window_local_id=str(obj.get("id") or ""),
                source_file=source_file_stem
            ))
            stats.kept_questions += 1

        pbar.update(1)
    pbar.close()

    for i, q in enumerate(all_items, start=1):
        q.qid = f"{source_file_stem}__Q{i:05d}"
    return all_items, stats, raw_records

# ========= IO 辅助 =========
def save_items_per_file(items: List[QAItem], out_dir: Path, base: str):
    out_dir_file = out_dir / base
    out_dir_file.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir_file / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    return jsonl_path

# ========= 单窗口处理（给并发调用） =========
def process_window(md_path: Path, s_line: int, e_line: int, chunk_text: str,
                   window: int, stride: int, max_retries: int, timeout: int,
                   out_root: Path, raw_root: Optional[Path], debug_dir: Optional[Path]) -> Tuple[str, int, int, List[QAItem], int, List[Dict[str, Any]]]:
    """
    处理单个窗口，返回 (source_file_stem, s_line, e_line, items, raw_questions_count, raw_records)
    """
    base_norm = md_path.stem
    compressed = compress_chunk_text(chunk_text, MAX_CHARS_PER_PROMPT)
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        start_line=s_line, end_line=e_line, chunk_text=compressed
    )
    
    # 调试目录
    wdir = None
    if debug_dir:
        wdir = debug_dir / base_norm / f"{s_line}-{e_line}"
        write_text(wdir / "prompt.txt", prompt)

    # 调用Ark处理窗口
    raw_out, data, meta = send_chunk_adaptive(
        s_line=s_line, e_line=e_line, chunk_text=chunk_text,
        max_retries=max_retries, retry_backoff=1.8, timeout=timeout,
        debug_dir=debug_dir, base_norm=base_norm
    )

    # 处理返回的数据
    items = []
    
    for obj in data:
        question = norm_text(str(obj.get("question") or ""))
        answer = norm_text(str(obj.get("answer") or ""))
        reference = norm_text(str(obj.get("reference") or ""))

        # 答案为空直接剔除
        if not answer:
            continue

        window_position = (s_line, e_line)
        window_range = f"{s_line}-{e_line}"

        items.append(QAItem(
            qid=None, question=question, answer=answer, reference=reference,
            source_window=(s_line, e_line), window_local_id=str(obj.get("id") or ""),
            source_file=base_norm
        ))

    return base_norm, s_line, e_line, items, len(data), [{
        "window": f"{s_line}-{e_line}",
        "chunk_text": chunk_text,
        "compressed_chunk": meta["compressed_chunk"],
        "prompt_length": meta["prompt_len"],
        "model_output": raw_out,
        "response_length": meta["response_len"],
        "sanitized_used": meta["sanitized_used"],
        "parse_error": meta["parse_error"],
        "parsed_questions": data
    }]


# ========= 主程序 =========
def main():
    ap = argparse.ArgumentParser(description="问答对抽取：处理Word文档或Markdown文档，抽取问答对")
    ap.add_argument("input_file", type=str, help="输入的文档路径（支持.docx, .doc, .md, .markdown, .txt）")
    ap.add_argument("--out-dir", type=str, default="out_qas", help="输出根目录")
    ap.add_argument("--save-raw-dir", type=str, default="", help="保存原文片段对照的目录（每文件 raw_chunks.jsonl）")
    ap.add_argument("--debug-log-dir", type=str, default="", help="保存每个窗口的 prompt/response/parse_error/sanitized.json")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_LINES, help="滑窗大小（行）")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE_LINES, help="滑窗步长（行）")
    ap.add_argument("--window-chars", type=int, default=3000, help="字符窗口大小（字符数）")
    ap.add_argument("--stride-chars", type=int, default=1800, help="字符窗口步长（字符数）")
    ap.add_argument("--use-char-window", action="store_true", help="使用字符窗口而不是行窗口")
    ap.add_argument("--max-retries", type=int, default=3, help="Ark 调用失败最大重试次数")
    ap.add_argument("--timeout", type=int, default=DEFAULT_ARK_TIMEOUT, help="单次 Ark 请求超时（秒）")
    ap.add_argument("--max-workers", type=int, default=256, help="并发线程数")
    args = ap.parse_args()

    input_file = Path(args.input_file)
    out_root = Path(args.out_dir)
    raw_root = Path(args.save_raw_dir) if args.save_raw_dir else None
    debug_dir = Path(args.debug_log_dir) if args.debug_log_dir else None

    if not input_file.exists():
        print(f"❌ 输入文件不存在：{input_file}"); return
    
    supported_formats = ['.docx', '.doc', '.md', '.markdown', '.txt']
    if not input_file.suffix.lower() in supported_formats:
        print(f"❌ 输入文件格式不支持，支持格式：{supported_formats}，当前文件：{input_file}"); return

    print(f"📄 处理文件：{input_file}")
    
    # 读取文档
    print("📥 读取文档...")
    try:
        text_content = read_document(input_file)
        # 清理多余空行
        cleaned_content = clean_text_lines(text_content, max_consecutive_empty=1)
        
        if args.use_char_window:
            # 使用字符窗口
            windows = make_char_windows(cleaned_content, args.window_chars, args.stride_chars)
            print(f"✅ 文档字符数：{len(cleaned_content)}，窗口数：{len(windows)}（字符窗口）")
        else:
            # 使用行窗口
            lines = cleaned_content.splitlines()
            windows = make_windows(lines, window=args.window, stride=args.stride)
            print(f"✅ 文档行数：{len(lines)}，窗口数：{len(windows)}（行窗口）")
    except Exception as e:
        print(f"❌ 读取文档失败：{e}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    combined_jsonl = out_root / "combined_questions.jsonl"
    combined_jsonl_before_dedup = out_root / "combined_questions_before_dedup.jsonl"

    # 窗口级并发处理
    print(f"🚀 使用窗口级并发，最大线程数：{args.max_workers}")
    
    # 准备窗口任务
    base_norm = input_file.stem
    
    file_stats = {
        base_norm: {
            'path': input_file,
            'windows': len(windows),
            'items': [],
            'raw_records': [],
            'stats': ExtractStats(windows=len(windows))
        }
    }
    
    all_window_tasks = []
    for s_line, e_line, chunk in windows:
        all_window_tasks.append((input_file, s_line, e_line, chunk))
    
    print(f"📊 总窗口数：{len(all_window_tasks)}")
    
    # 全局窗口级并发处理
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
                            executor.submit(
                    process_window,
                    md_path=task[0], s_line=task[1], e_line=task[2], chunk_text=task[3],
                    window=args.window, stride=args.stride, max_retries=args.max_retries, timeout=args.timeout,
                    out_root=out_root, raw_root=raw_root, debug_dir=debug_dir
                ): (task[0].stem, task[1], task[2])  # 用 (文件名, 起始行, 结束行) 作为key
            for task in all_window_tasks
        }

        # 处理完成的窗口
        for fut in tqdm(cf.as_completed(futures), total=len(futures), ncols=100, desc="窗口级并发抽取"):
            file_stem, s_line, e_line = futures[fut]
            try:
                source_file_stem, s_line, e_line, items, raw_questions_count, raw_records = fut.result()
                
                # 更新文件统计
                file_stats[source_file_stem]['items'].extend(items)
                file_stats[source_file_stem]['raw_records'].extend(raw_records)
                file_stats[source_file_stem]['stats'].raw_questions += raw_questions_count
                file_stats[source_file_stem]['stats'].kept_questions += len(items)
                
            except Exception as e:
                print(f"\n❌ 窗口处理失败 {file_stem} {s_line}-{e_line}: {e}")
    
    # 收集所有题目，准备全局重新分配qid
    print("\n📝 收集所有题目并准备重新分配qid...")
    all_items_for_qid = []
    combined_rows: List[dict] = []
    
    for base_norm, file_info in file_stats.items():
        if file_info['items']:
            all_items_for_qid.extend(file_info['items'])
    
    # 先保存去重前的数据
    print(f"\n💾 保存去重前的数据...")
    if all_items_for_qid:
        # 为去重前的数据分配临时qid
        source_file = all_items_for_qid[0].source_file if all_items_for_qid else "问答整理"
        for i, item in enumerate(all_items_for_qid, start=1):
            item.qid = f"{source_file}__Q{i:05d}"
        
        # 保存去重前的combined文件
        with combined_jsonl_before_dedup.open("w", encoding="utf-8") as f:
            for it in all_items_for_qid:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        print(f"✅ 去重前数据已保存：{combined_jsonl_before_dedup} ({len(all_items_for_qid)} 题)")
    
    # 进行相似度聚类和去重处理
    print(f"\n🔄 开始相似度聚类和去重处理，原始题目数：{len(all_items_for_qid)}")
    cluster_log_file = out_root / "cluster_log.md"
    clustered_items, cluster_logs = cluster_and_deduplicate_questions(all_items_for_qid, SIMILARITY_THRESHOLD, cluster_log_file)
    
    # 对聚类后的数据重新分配qid（按文件顺序，然后按项目顺序）
    print("🔢 对聚类后的数据重新分配qid...")
    source_file = clustered_items[0].source_file if clustered_items else "问答整理"
    for i, item in enumerate(clustered_items, start=1):
        item.qid = f"{source_file}__Q{i:05d}"
    
    # 保存文件级结果（使用聚类后的数据）
    print("💾 保存文件级结果...")
    for base_norm, file_info in file_stats.items():
        if file_info['items']:
            # 从聚类后的数据中筛选属于当前文件的题目
            file_clustered_items = [item for item in clustered_items if item.source_file == base_norm]
            
            # 保存文件级结果
            jsonl_path = save_items_per_file(file_clustered_items, out_root, base_norm)
            raw_path = None
            if raw_root:
                raw_path = raw_root / base_norm / "raw_chunks.jsonl"
                save_raw_chunks(file_info['raw_records'], raw_path)
            
            print(f"✅ {file_info['path'].name} → 抽取 {len(file_info['items'])} 题 → 聚类后 {len(file_clustered_items)} 题")
            if raw_path:
                print(f"📂 原文片段对照：{raw_path}")
            
            # 收集所有题目用于批量写入combined文档
            for it in file_clustered_items:
                combined_rows.append({
                    "qid": it.qid, "source_file": it.source_file,
                    "question_preview": it.question[:160].replace("\n", " "),
                    "has_answer": 1 if it.answer else 0,
                    "has_reference": 1 if it.reference else 0,
                })
    
    
    # 批量写入combined文档（聚类后）
    print("\n💾 批量写入combined文档（聚类后）...")
    if clustered_items:
        # 写入聚类后的JSONL
        with combined_jsonl.open("w", encoding="utf-8") as f:
            for it in clustered_items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        
        print(f"✅ 已批量写入combined文档，总计：{len(clustered_items)} 题（聚类去重后）")

    print("\n=== 处理完成 ===")
    print(f"📊 统计：原始 {len(all_items_for_qid)} 题 → 聚类后 {len(clustered_items)} 题")
    print(f"🗂 结果文件：{combined_jsonl.resolve()}")
    if cluster_logs:
        print(f"📝 聚类日志：{cluster_log_file.resolve()}")
    

if __name__ == "__main__":
    main()


# 运行示例
'''
按照字符窗口抽取
python /home/wangxi/workspace/data-engine/wx/fqa_extract/1_fqa_extract.py \
  /home/wangxi/workspace/xiaofang/fqa/问答整理.docx \
  --use-char-window --window-chars 3000 --stride-chars 1800 \
  --max-workers 256 \
  --timeout 120 \
  --out-dir out_qas_$(date +%m%d%H%M) \
  --save-raw-dir raw_logs_$(date +%m%d%H%M) \
  --debug-log-dir debug_logs_$(date +%m%d%H%M)
'''

'''
按照行窗口抽取（不推荐）
python /home/wangxi/workspace/data-engine/wx/fqa_extract/1_fqa_extract.py \
  /home/wangxi/workspace/xiaofang/fqa/问答整理.md \
  --window 80 --stride 40 \
  --max-workers 256 \
  --timeout 120 \
  --out-dir out_qas_$(date +%m%d%H%M) \
  --save-raw-dir raw_logs_$(date +%m%d%H%M) \
  --debug-log-dir debug_logs_$(date +%m%d%H%M)
'''