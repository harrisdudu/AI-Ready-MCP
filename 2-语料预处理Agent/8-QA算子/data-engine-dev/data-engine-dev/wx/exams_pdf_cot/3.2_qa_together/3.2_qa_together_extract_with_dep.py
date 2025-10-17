#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import pandas as pd
import concurrent.futures as cf

# ========= Ark 配置 =========
API_KEY  = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

try:
    from volcenginesdkarkruntime import Ark
except Exception:
    raise SystemExit("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
except Exception:
    raise SystemExit("❌ 请先安装 sentence-transformers: pip install sentence-transformers")

# ========= 参数常量 =========
DEFAULT_WINDOW_LINES = 80    # 稍微保守，减少长窗超时
DEFAULT_STRIDE_LINES = 40
SUPPORTED_TYPES = {"single", "multiple", "judge", "fill"}

# 单次请求最大字符预算（避免超大窗口导致超时/超长）
MAX_CHARS_PER_PROMPT = 9000
# Ark 默认请求超时（秒）
DEFAULT_ARK_TIMEOUT = 120

# Embedding 配置
DEFAULT_EMBEDDING_MODEL = "/home/wangxi/workspace/gongye/yijizaojia/Qwen3-Embedding-0.6B"  # 本地模型路径
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # 余弦相似度阈值

# ========= System/User Prompt =========
SYSTEM_PROMPT = """\
你是一个精确的试题抽取助手。
我将给你一段从考试 PDF 转成 Markdown 的文本片段。
你的任务：只抽取四种类型的完整试题：
- single（单选题）
- multiple（多选题）
- judge（判断题）
- fill（填空题）

规则：
1) 每道题必须包含：
   - "id"：在当前片段内的局部递增编号（字符串或数字）
   - "type"：取值只能是 ["single","multiple","judge","fill"]
   - "question"：题干的完整文字（保留题干中的图片链接或公式）和选项列表（按原顺序）
   - "answer"：简洁的答案文字；选择题用字母（如 "A" 或 "ACD"）；判断题用“对/错”或“True/False”；填空题用实际填空内容（如有）
   - "explanation"：解析说明（如无则为空字符串）
   - "knowledge_points"：知识点标签（如无则为空字符串）
2) 只输出完整的试题：必须有完整的问题和答案。
3) 如果题目被截断或不完整，则不要输出。
4) 严格输出 JSON 格式，UTF-8 编码，不要包含 Markdown、代码块围栏或多余文字。
5) 不要编造内容。答案、解析或知识点缺失时用空字符串表示。
6) 如果片段中没有符合条件的试题，返回空数组 []。

输出格式：
JSON 数组，数组中每个元素是一个试题对象。
"""

CHUNK_PROMPT_TEMPLATE = """\
下面是全文的第 {start_line} 行到第 {end_line} 行的 Markdown 内容。
请仅提取其中符合要求且完整的试题（题型限定在 single/multiple/judge/fill）。

--- 开始片段 ---
{chunk_text}
--- 结束片段 ---

请记住：
- 只保留 ["single","multiple","judge","fill"] 这四种题型。
- 只返回严格 JSON 数组，不要输出任何额外文字。
"""

# ========= 数据结构 =========
@dataclass
class QAItem:
    qid: Optional[str]
    type: str
    question: str
    answer: str
    explanation: str
    knowledge_points: str
    source_window: Tuple[int, int]
    window_local_id: Optional[str]
    source_file: str = ""

@dataclass
class ExtractStats:
    windows: int = 0
    raw_questions: int = 0
    kept_questions: int = 0
    dedup_dropped: int = 0
    embedding_dedup_dropped: int = 0  # embedding相似度去重丢弃的数量

@dataclass
class DedupLog:
    """去重日志记录"""
    timestamp: str
    source_file: str
    window_range: str
    question_text: str
    dedup_type: str  # "md5" 或 "embedding"
    reason: str  # 去重原因描述
    similarity_score: Optional[float] = None  # 仅embedding去重时有值
    duplicate_with: Optional[str] = None  # 与哪个题目重复（题目预览）

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

def type_filter(qtype: str) -> bool:
    return qtype in SUPPORTED_TYPES

def hash_key(question: str) -> str:
    import hashlib
    base = norm_text(question)
    return hashlib.md5(base.encode("utf-8")).hexdigest()

# ========= 去重日志记录器 =========
class DedupLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
    def log_dedup(self, dedup_log: DedupLog):
        """记录去重日志"""
        with self.lock:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(dedup_log), ensure_ascii=False) + "\n")
    
    def log_md5_dedup(self, source_file: str, window_range: str, question: str, 
                     duplicate_with: str):
        """记录MD5去重"""
        dedup_log = DedupLog(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            source_file=source_file,
            window_range=window_range,
            question_text=question[:200] + "..." if len(question) > 200 else question,
            dedup_type="md5",
            reason="MD5哈希值重复",
            duplicate_with=duplicate_with[:200] + "..." if len(duplicate_with) > 200 else duplicate_with
        )
        self.log_dedup(dedup_log)
    
    def log_embedding_dedup(self, source_file: str, window_range: str, question: str,
                          duplicate_with: str, similarity_score: float):
        """记录embedding去重"""
        # 确保similarity_score是Python原生float类型
        if hasattr(similarity_score, 'item'):
            similarity_score = float(similarity_score.item())
        else:
            similarity_score = float(similarity_score)
            
        dedup_log = DedupLog(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            source_file=source_file,
            window_range=window_range,
            question_text=question[:200] + "..." if len(question) > 200 else question,
            dedup_type="embedding",
            reason=f"embedding相似度 {similarity_score:.4f} 超过阈值",
            similarity_score=similarity_score,
            duplicate_with=duplicate_with[:200] + "..." if len(duplicate_with) > 200 else duplicate_with
        )
        self.log_dedup(dedup_log)

# ========= Embedding 相关工具 ========
import threading
import time

class EmbeddingDeduplicator:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, 
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        # 支持本地模型路径
        if Path(model_name).exists():
            print(f"🔍 加载本地embedding模型：{model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            print(f"🔍 下载embedding模型：{model_name}")
            self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.embeddings = []
        self.questions = []
        self.lock = threading.Lock()  # 线程锁
        
    def add_question(self, question: str) -> Tuple[bool, Optional[Tuple[str, float]]]:
        """
        添加问题，如果与已有问题相似度超过阈值则返回(False, (重复题目, 相似度))
        否则返回(True, None)（表示新增）
        """
        with self.lock:  # 线程安全
            if not self.questions:
                # 第一个问题，直接添加
                embedding = self.model.encode([question])[0]
                self.embeddings.append(embedding)
                self.questions.append(question)
                return True, None
                
            # 计算与所有已有问题的相似度
            new_embedding = self.model.encode([question])[0]
            similarities = cosine_similarity([new_embedding], self.embeddings)[0]
            
            # 如果最大相似度超过阈值，认为是重复
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= self.threshold:
                duplicate_question = self.questions[max_similarity_idx]
                return False, (duplicate_question, max_similarity)
                
            # 不重复，添加到列表
            self.embeddings.append(new_embedding)
            self.questions.append(question)
            return True, None

def make_windows(lines: List[str], window: int, stride: int) -> List[Tuple[int, int, str]]:
    n = len(lines); out = []; i = 0
    while i < n:
        s = i; e = min(i + window - 1, n - 1)
        out.append((s+1, e+1, "\n".join(lines[s:e+1])))
        if e == n - 1: break
        i += stride
    return out

def compress_chunk_text(s: str, max_chars: int = MAX_CHARS_PER_PROMPT) -> str:
    lines = s.replace("\r", "").splitlines()
    out = []; blank = 0
    for ln in lines:
        ln = " ".join(ln.split())
        if not ln:
            blank += 1
            if blank > 1: continue
        else:
            blank = 0
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

# ========= 抽取单文件 =========
def extract_from_md(md_path: Path, window: int, stride: int,
                    max_retries: int, retry_backoff: float, timeout: int,
                    source_file_stem: str,
                    debug_dir: Optional[Path],
                    embedding_dedup: Optional[EmbeddingDeduplicator] = None,
                    dedup_logger: Optional[DedupLogger] = None) -> Tuple[List[QAItem], ExtractStats, List[Dict[str, Any]]]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    windows = make_windows(lines, window=window, stride=stride)

    all_items: List[QAItem] = []
    seen = set()
    seen_questions = {}  # 记录已见过的题目，用于去重日志
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
            explanation = norm_text(str(obj.get("explanation") or ""))
            knowledge = norm_text(str(obj.get("knowledge_points") or ""))
            qtype = norm_text(str(obj.get("type") or ""))

            # 题型过滤
            if not qtype or not type_filter(qtype):
                continue

            # ======= 关键改动：答案为空直接剔除 =======
            if not answer:
                continue
            # =======================================

            # hash 去重
            key = hash_key(question)
            if key in seen:
                stats.dedup_dropped += 1
                # 记录MD5去重日志
                if dedup_logger:
                    window_range = f"{s_line}-{e_line}"
                    duplicate_question = seen_questions.get(key, "未知题目")
                    dedup_logger.log_md5_dedup(source_file_stem, window_range, question, duplicate_question)
                continue
            seen.add(key)
            seen_questions[key] = question

            # Embedding 相似度去重
            if embedding_dedup is not None:
                is_unique, duplicate_info = embedding_dedup.add_question(question)
                if not is_unique:
                    stats.embedding_dedup_dropped += 1
                    # 记录embedding去重日志
                    if dedup_logger:
                        window_range = f"{s_line}-{e_line}"
                        duplicate_question, similarity_score = duplicate_info
                        dedup_logger.log_embedding_dedup(source_file_stem, window_range, question, 
                                                       duplicate_question, similarity_score)
                    continue

            all_items.append(QAItem(
                qid=None, type=qtype, question=question, answer=answer,
                explanation=explanation, knowledge_points=knowledge,
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
    rows = []
    for it in items:
        rows.append({
            "qid": it.qid, "source_file": it.source_file, "type": it.type,
            "question_preview": it.question[:160].replace("\n", " "),
            "has_answer": 1 if it.answer else 0,
            "has_explanation": 1 if it.explanation else 0,
            "has_kp": 1 if it.knowledge_points else 0,
            "source_window": f"{it.source_window[0]}-{it.source_window[1]}",
            "window_local_id": it.window_local_id or ""
        })
    pd.DataFrame(rows).to_csv(out_dir_file / "summary.csv", index=False, encoding="utf-8-sig")
    return jsonl_path, out_dir_file / "summary.csv"

def append_combined(items: List[QAItem], combined_jsonl: Path, combined_rows: List[dict]):
    combined_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with combined_jsonl.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
            combined_rows.append({
                "qid": it.qid, "source_file": it.source_file, "type": it.type,
                "question_preview": it.question[:160].replace("\n", " "),
                "has_answer": 1 if it.answer else 0,
                "has_explanation": 1 if it.explanation else 0,
                "has_kp": 1 if it.knowledge_points else 0,
            })

# ========= 标注读取 & 文件查找 =========
def load_target_names(label_json_path: Path) -> set:
    targets = set()

    def add_name(v: str):
        v = (v or "").strip()
        if not v:
            return
        name = Path(v).name        # 最后一层文件名
        stem = Path(name).stem     # 去掉扩展名
        if stem.endswith("_content_list"):
            stem = stem[:-len("_content_list")].rstrip("_- ")
        targets.add(stem)

    with open(label_json_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # 只处理 label 为 MIXED_TOGETHER 的记录
            if obj.get("label") == "MIXED_TOGETHER":
                add_name(obj.get("file") or obj.get("path") or obj.get("File") or "")

    return targets

def find_md_files(md_root: Path, target_names: set) -> List[Path]:
    exts = {'.md', '.markdown'}
    files = []
    unmatched = []

    for p in md_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            stem = Path(p.name).stem  # 直接取去扩展名的部分
            if stem in target_names:
                files.append(p)

    return sorted(files)

# ========= 单窗口处理（给并发调用） =========
def process_window(md_path: Path, s_line: int, e_line: int, chunk_text: str,
                   window: int, stride: int, max_retries: int, timeout: int,
                   out_root: Path, raw_root: Optional[Path], debug_dir: Optional[Path],
                   embedding_dedup: Optional[EmbeddingDeduplicator] = None,
                   dedup_logger: Optional[DedupLogger] = None) -> Tuple[str, int, int, List[QAItem], List[Dict[str, Any]]]:
    """
    处理单个窗口，返回 (source_file_stem, s_line, e_line, items, raw_records)
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
    seen = set()  # 每个窗口内的去重
    seen_questions = {}  # 记录已见过的题目，用于去重日志
    
    for obj in data:
        question = norm_text(str(obj.get("question") or ""))
        answer = norm_text(str(obj.get("answer") or ""))
        explanation = norm_text(str(obj.get("explanation") or ""))
        knowledge = norm_text(str(obj.get("knowledge_points") or ""))
        qtype = norm_text(str(obj.get("type") or ""))

        # 题型过滤
        if not qtype or not type_filter(qtype):
            continue

        # 答案为空直接剔除
        if not answer:
            continue

        # hash 去重（窗口内）
        key = hash_key(question)
        if key in seen:
            # 记录MD5去重日志
            if dedup_logger:
                window_range = f"{s_line}-{e_line}"
                duplicate_question = seen_questions.get(key, "未知题目")
                dedup_logger.log_md5_dedup(base_norm, window_range, question, duplicate_question)
            continue
        seen.add(key)
        seen_questions[key] = question

        # Embedding 相似度去重
        if embedding_dedup is not None:
            is_unique, duplicate_info = embedding_dedup.add_question(question)
            if not is_unique:
                # 记录embedding去重日志
                if dedup_logger:
                    window_range = f"{s_line}-{e_line}"
                    duplicate_question, similarity_score = duplicate_info
                    dedup_logger.log_embedding_dedup(base_norm, window_range, question, 
                                                   duplicate_question, similarity_score)
                continue

        items.append(QAItem(
            qid=None, type=qtype, question=question, answer=answer,
            explanation=explanation, knowledge_points=knowledge,
            source_window=(s_line, e_line), window_local_id=str(obj.get("id") or ""),
            source_file=base_norm
        ))

    return base_norm, s_line, e_line, items, [{
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

# ========= 单文件处理（给并发调用） =========
def process_one(md_path: Path,
                window: int,
                stride: int,
                max_retries: int,
                timeout: int,
                out_root: Path,
                raw_root: Optional[Path],
                debug_dir: Optional[Path],
                embedding_dedup: Optional[EmbeddingDeduplicator] = None,
                dedup_logger: Optional[DedupLogger] = None) -> Tuple[str, List[QAItem], ExtractStats, List[Dict[str, Any]], Path, Path, Optional[Path]]:
    base_norm = md_path.stem
    items, stats, raw_records = extract_from_md(
        md_path=md_path,
        window=window, stride=stride,
        max_retries=max_retries, retry_backoff=1.8, timeout=timeout,
        source_file_stem=base_norm, debug_dir=debug_dir,
        embedding_dedup=embedding_dedup,
        dedup_logger=dedup_logger
    )

    jsonl_path, csv_path = save_items_per_file(items, out_root, base_norm)
    raw_path = None
    if raw_root:
        raw_path = raw_root / base_norm / "raw_chunks.jsonl"
        save_raw_chunks(raw_records, raw_path)

    return base_norm, items, stats, raw_records, jsonl_path, csv_path, raw_path


# ========= 主程序 =========
def main():
    ap = argparse.ArgumentParser(description="批量：按标注JSON筛选 .md，Ark 抽取试题（带请求/返回调试日志）")
    ap.add_argument("md_dir", type=str, help="Markdown 根目录（递归）")
    ap.add_argument("label_json", type=str, help="标注行式 JSONL，只处理 label 为 MIXED_TOGETHER 的记录")
    ap.add_argument("--out-dir", type=str, default="out_qas", help="输出根目录")
    ap.add_argument("--save-raw-dir", type=str, default="", help="保存原文片段对照的目录（每文件 raw_chunks.jsonl）")
    ap.add_argument("--debug-log-dir", type=str, default="", help="保存每个窗口的 prompt/response/parse_error/sanitized.json")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_LINES, help="滑窗大小（行）")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE_LINES, help="滑窗步长（行）")
    ap.add_argument("--max-retries", type=int, default=3, help="Ark 调用失败最大重试次数")
    ap.add_argument("--timeout", type=int, default=DEFAULT_ARK_TIMEOUT, help="单次 Ark 请求超时（秒）")
    ap.add_argument("--max-workers", type=int, default=256, help="并发线程数")
    ap.add_argument("--concurrency-level", type=str, choices=["window", "file"], default="window", 
                    help="并发级别：window(窗口级并发) 或 file(文件级并发)")
    ap.add_argument("--enable-embedding-dedup", action="store_true", help="启用embedding相似度去重")
    ap.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="embedding模型名称或本地模型路径")
    ap.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="余弦相似度阈值")
    ap.add_argument("--enable-dedup-log", action="store_true", help="启用去重过程日志记录")
    args = ap.parse_args()

    md_root = Path(args.md_dir)
    label_json_path = Path(args.label_json)
    out_root = Path(args.out_dir)
    raw_root = Path(args.save_raw_dir) if args.save_raw_dir else None
    debug_dir = Path(args.debug_log_dir) if args.debug_log_dir else None

    if not md_root.exists():
        print(f"❌ md_dir 不存在：{md_root}"); return
    if not label_json_path.exists():
        print(f"❌ label_json 不存在：{label_json_path}"); return

    print("📥 读取标注 ...")
    target_names = load_target_names(label_json_path)
    print(f"✅ 目标文件数量：{len(target_names)}")

    print("🔍 扫描 .md ...")
    md_files = find_md_files(md_root, target_names)
    print(f"📄 待处理文件数：{len(md_files)}")
    if not md_files:
        print("🛑 未找到可处理的 md"); return

    out_root.mkdir(parents=True, exist_ok=True)
    combined_jsonl = out_root / "combined_questions.jsonl"
    combined_csv = out_root / "combined_summary.csv"
    
    # 创建combined文件并写入表头
    if combined_jsonl.exists(): combined_jsonl.unlink()
    with combined_jsonl.open("w", encoding="utf-8") as f:
        pass  # 创建空文件
    
    # 创建CSV文件并写入表头
    csv_headers = ["qid", "source_file", "type", "question_preview", "has_answer", "has_explanation", "has_kp"]
    with combined_csv.open("w", encoding="utf-8-sig") as f:
        f.write(",".join(csv_headers) + "\n")

    # 初始化embedding去重器（如果启用）
    embedding_dedup = None
    if args.enable_embedding_dedup:
        print(f"🔍 初始化embedding去重器，模型：{args.embedding_model}，阈值：{args.similarity_threshold}")
        embedding_dedup = EmbeddingDeduplicator(
            model_name=args.embedding_model,
            similarity_threshold=args.similarity_threshold
        )

    # 初始化去重日志记录器（如果启用）
    dedup_logger = None
    if args.enable_dedup_log:
        dedup_log_file = out_root / "dedup_log.jsonl"
        dedup_logger = DedupLogger(dedup_log_file)
        print(f"📝 启用去重日志记录：{dedup_log_file}")
        if dedup_log_file.exists():
            dedup_log_file.unlink()  # 清空之前的日志

    # 根据并发级别选择处理方式
    combined_rows: List[dict] = []
    
    if args.concurrency_level == "window":
        # 窗口级并发处理
        print(f"🚀 使用窗口级并发，最大线程数：{args.max_workers}")
        
        # 收集所有窗口任务
        all_window_tasks = []
        file_stats = {}  # 用于跟踪每个文件的统计信息
        
        print("📋 准备窗口任务...")
        for md_path in md_files:
            base_norm = md_path.stem
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            windows = make_windows(lines, window=args.window, stride=args.stride)
            
            file_stats[base_norm] = {
                'path': md_path,
                'windows': len(windows),
                'items': [],
                'raw_records': [],
                'stats': ExtractStats(windows=len(windows))
            }
            
            for s_line, e_line, chunk in windows:
                all_window_tasks.append((md_path, s_line, e_line, chunk))
        
        print(f"📊 总窗口数：{len(all_window_tasks)}")
        
        # 全局窗口级并发处理
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_window,
                    md_path=task[0], s_line=task[1], e_line=task[2], chunk_text=task[3],
                    window=args.window, stride=args.stride, max_retries=args.max_retries, timeout=args.timeout,
                    out_root=out_root, raw_root=raw_root, debug_dir=debug_dir,
                    embedding_dedup=embedding_dedup,
                    dedup_logger=dedup_logger
                ): (task[0].stem, task[1], task[2])  # 用 (文件名, 起始行, 结束行) 作为key
                for task in all_window_tasks
            }

            # 处理完成的窗口
            for fut in tqdm(cf.as_completed(futures), total=len(futures), ncols=100, desc="窗口级并发抽取"):
                file_stem, s_line, e_line = futures[fut]
                try:
                    source_file_stem, s_line, e_line, items, raw_records = fut.result()
                    
                    # 更新文件统计
                    file_stats[source_file_stem]['items'].extend(items)
                    file_stats[source_file_stem]['raw_records'].extend(raw_records)
                    file_stats[source_file_stem]['stats'].raw_questions += len(items)
                    file_stats[source_file_stem]['stats'].kept_questions += len(items)
                    
                    # 实时写入combined文档
                    if items:
                        # 写入JSONL
                        with combined_jsonl.open("a", encoding="utf-8") as f:
                            for it in items:
                                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
                        
                        # 写入CSV
                        with combined_csv.open("a", encoding="utf-8-sig") as f:
                            for it in items:
                                row = [
                                    it.qid, it.source_file, it.type,
                                    it.question[:160].replace("\n", " "),
                                    "1" if it.answer else "0",
                                    "1" if it.explanation else "0", 
                                    "1" if it.knowledge_points else "0"
                                ]
                                f.write(",".join(str(cell) for cell in row) + "\n")
                        
                        # 同时维护内存中的汇总数据（用于最终统计）
                        for it in items:
                            combined_rows.append({
                                "qid": it.qid, "source_file": it.source_file, "type": it.type,
                                "question_preview": it.question[:160].replace("\n", " "),
                                "has_answer": 1 if it.answer else 0,
                                "has_explanation": 1 if it.explanation else 0,
                                "has_kp": 1 if it.knowledge_points else 0,
                            })
                            
                    # 每处理100个窗口显示一次进度
                    if len(combined_rows) % 100 == 0:
                        print(f"💾 已实时写入combined文档，当前总计：{len(combined_rows)} 题")
                        
                except Exception as e:
                    print(f"\n❌ 窗口处理失败 {file_stem} {s_line}-{e_line}: {e}")
        
        # 为每个文件的题目分配qid并保存文件级结果
        print("\n📝 保存文件级结果...")
        for base_norm, file_info in file_stats.items():
            if file_info['items']:
                # 为题目分配qid
                for i, q in enumerate(file_info['items'], start=1):
                    q.qid = f"{base_norm}__Q{i:05d}"
                
                # 保存文件级结果
                jsonl_path, csv_path = save_items_per_file(file_info['items'], out_root, base_norm)
                raw_path = None
                if raw_root:
                    raw_path = raw_root / base_norm / "raw_chunks.jsonl"
                    save_raw_chunks(file_info['raw_records'], raw_path)
                
                print(f"✅ {file_info['path'].name} → 抽取 {len(file_info['items'])} 题；窗口数={file_info['stats'].windows}, "
                      f"模型返回={file_info['stats'].raw_questions}, 保留={file_info['stats'].kept_questions}")
                print(f"📝 {jsonl_path}")
                print(f"🧾 {csv_path}")
                if raw_path:
                    print(f"📂 原文片段对照：{raw_path}")
    
    else:
        # 文件级并发处理
        print(f"🚀 使用文件级并发，最大线程数：{args.max_workers}")
        
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_one,
                    md_path=md_path,
                    window=args.window,
                    stride=args.stride,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    out_root=out_root,
                    raw_root=raw_root,
                    debug_dir=debug_dir,
                    embedding_dedup=embedding_dedup,
                    dedup_logger=dedup_logger
                ): md_path
                for md_path in md_files
            }

            for fut in tqdm(cf.as_completed(futures), total=len(futures), ncols=100, desc="文件级并发抽取"):
                md_path = futures[fut]
                try:
                    base_norm, items, stats, raw_records, jsonl_path, csv_path, raw_path = fut.result()
                    print(f"\n✅ {md_path.name} → 抽取 {len(items)} 题；窗口数={stats.windows}, 模型返回={stats.raw_questions}, "
                          f"保留={stats.kept_questions}, hash去重丢弃={stats.dedup_dropped}, embedding去重丢弃={stats.embedding_dedup_dropped}")
                    print(f"📝 {jsonl_path}")
                    print(f"🧾 {csv_path}")
                    if raw_path:
                        print(f"📂 原文片段对照：{raw_path}")

                    # 实时写入combined文档
                    if items:
                        # 写入JSONL
                        with combined_jsonl.open("a", encoding="utf-8") as f:
                            for it in items:
                                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
                        
                        # 写入CSV
                        with combined_csv.open("a", encoding="utf-8-sig") as f:
                            for it in items:
                                row = [
                                    it.qid, it.source_file, it.type,
                                    it.question[:160].replace("\n", " "),
                                    "1" if it.answer else "0",
                                    "1" if it.explanation else "0", 
                                    "1" if it.knowledge_points else "0"
                                ]
                                f.write(",".join(str(cell) for cell in row) + "\n")
                        
                        # 同时维护内存中的汇总数据（用于最终统计）
                        for it in items:
                            combined_rows.append({
                                "qid": it.qid, "source_file": it.source_file, "type": it.type,
                                "question_preview": it.question[:160].replace("\n", " "),
                                "has_answer": 1 if it.answer else 0,
                                "has_explanation": 1 if it.explanation else 0,
                                "has_kp": 1 if it.knowledge_points else 0,
                            })
                            
                    print(f"💾 已实时写入combined文档，当前总计：{len(combined_rows)} 题")
                except Exception as e:
                    print(f"\n❌ 处理失败：{md_path} -> {e}")

    print("\n=== 全量汇总完成 ===")
    print(f"🗂 题目汇总 JSONL：{combined_jsonl.resolve()}")
    print(f"📊 题目汇总 CSV ：{combined_csv.resolve()}")
    
    # 显示去重统计信息
    if args.enable_dedup_log and dedup_logger and dedup_logger.log_file.exists():
        print(f"\n📝 去重日志文件：{dedup_logger.log_file.resolve()}")
        
        # 统计去重情况
        md5_count = 0
        embedding_count = 0
        total_dedup = 0
        
        with dedup_logger.log_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        log_entry = json.loads(line)
                        total_dedup += 1
                        if log_entry.get("dedup_type") == "md5":
                            md5_count += 1
                        elif log_entry.get("dedup_type") == "embedding":
                            embedding_count += 1
                    except:
                        continue
        
        print(f"📊 去重统计：")
        print(f"   - 总去重数量：{total_dedup}")
        print(f"   - MD5去重：{md5_count}")
        print(f"   - Embedding去重：{embedding_count}")
        
        if embedding_count > 0:
            # 计算embedding去重的相似度分布
            similarities = []
            with dedup_logger.log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            log_entry = json.loads(line)
                            if log_entry.get("dedup_type") == "embedding" and log_entry.get("similarity_score"):
                                similarities.append(log_entry["similarity_score"])
                        except:
                            continue
            
            if similarities:
                # 确保所有相似度都是Python原生float类型
                similarities = [float(s.item()) if hasattr(s, 'item') else float(s) for s in similarities]
                print(f"   - Embedding相似度统计：")
                print(f"     * 平均相似度：{np.mean(similarities):.4f}")
                print(f"     * 最高相似度：{np.max(similarities):.4f}")
                print(f"     * 最低相似度：{np.min(similarities):.4f}")
                print(f"     * 相似度阈值：{args.similarity_threshold}")

if __name__ == "__main__":
    main()


# 运行示例
'''python 3_qa.py \
  /home/wangxi/workspace/gongye/zejun \
  /home/wangxi/workspace/gongye/shiti/classification_results_20250815_110753.jsonl \
  --max-workers 256 \
  --window 80 --stride 40 \
  --timeout 120 \
  --out-dir shiti/out_qas \
  --save-raw-dir shiti/raw_logs \
  --debug-log-dir shiti/debug_logs \
  --enable-embedding-dedup \
  --embedding-model /home/wangxi/workspace/gongye/yijizaojia/Qwen3-Embedding-0.6B \
  --concurrency-level window \
  --similarity-threshold 0.99 \
  --enable-dedup-log'''