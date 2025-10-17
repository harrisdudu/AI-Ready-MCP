#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法示例：
python 4_cot_check.py \
  --input-jsonl  /home/wangxi/workspace/gongye/yijizaojia/out_qas_0814/combined_questions.jsonl \
  --out-jsonl    /home/wangxi/workspace/gongye/yijizaojia/out_qas_0814/combined_questions_cot.jsonl \
  --max-workers  256 \
  --temperature  0.25 \
  --top-p        0.8 \
  --thinking     enabled \
  --retries      5 \
  --max-correct-rounds 8
"""

import argparse
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import concurrent.futures as cf
from tqdm import tqdm

# ========== Ark 固定配置（请按需修改/置于环境变量） ==========
API_KEY   = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID  = "doubao-seed-1-6-250615"
BACKEND   = "ark"

# 参数默认值
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P       = 0.9
DEFAULT_RETRIES     = 5
DEFAULT_BACKOFF     = 1.6
DEFAULT_MAX_WORKERS = 256
DEFAULT_THINKING    = "disabled"  # disabled | enabled | auto
DEFAULT_MAX_CORRECT = 8           # 自纠最多轮数

# Ark SDK
try:
    from volcenginesdkarkruntime import Ark
    from volcenginesdkarkruntime._constants import CLIENT_REQUEST_HEADER
except Exception:
    raise SystemExit("❌ 需要安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

# ========= 生成阶段：严格“按标准答案去推理”的提示词 =========
PROMPT_SYSTEM = (
    "你是一个严谨的解题助手。"
    "请在“最终答案必须与给定的标准答案完全一致”的前提下，给出自洽、可验证、逐步展开的推理过程。"
    "禁止质疑或更改标准答案；如题干信息不足，可以补充常识性解释，但不得与标准答案矛盾。"
    "直接输出推理文本，不要 JSON、不用代码围栏。"
)

PROMPT_USER_TEMPLATE = """请根据以下信息，给出严密、逐步、可验证的推理过程，并最终得出与标准答案一致的结论。

【题目与上下文】
{question_md}

【选项】（如无可留空数组）
{options_md}

【标准答案（必须严格一致）】
{given_answer}

要求：
1) 以“{given_answer}”为唯一正确结论进行反向与正向结合的推理：先解释该答案为何成立，再说明其他选项为何不成立（若适用）。
2) 过程要条理清晰，避免空洞复述与无关扩写；必要时可引用关键定义/定理/公式/法规点，但要简洁准确。
3) 只输出推理过程文本与最终一次性结论，不要输出 JSON、不要多次给出不同结论。
"""

# ========= 自纠阶段：当质检失败时的纠偏提示 =========
CORRECTION_USER_TEMPLATE = """【纠偏重写】
你之前的推理/输出与标准答案不一致或存在歧义。请在不改变标准答案前提下，重写更严密、更具可验证性的推理过程，并只给出一次明确结论。

【题目与上下文】
{question_md}

【选项】
{options_md}

【标准答案（必须严格一致）】
{given_answer}

请勿质疑或更改标准答案；请通过充分、条理清晰的论证，使最终结论严格为“{given_answer}”。只输出推理文本。
"""

# ========= 质检（模型判断）提示 =========
QC_SYSTEM_PROMPT = (
    "你是一个严格的试题质检员。"
    "给定模型输出的 content 和标准答案 answer，你需要判断它们是否一致。"
    "只输出 JSON，字段如下："
    "{\"pass\": <bool>, \"confidence\": <0.0-1.0>, \"rationale\": \"简短说明理由\"}"
    "只输出 JSON，不要其他内容。"
)

QC_USER_TEMPLATE = """请判断以下模型输出与标准答案是否一致，仅输出 JSON：
[标准答案]
{gold_answer}

[模型输出]
{content}
"""

# ---------------- 工具函数 ----------------
def _safe_header_token(s) -> str:
    s = str(s) if s is not None else ""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    txt = s.strip()
    fence = re.findall(r"```(?:json|text)?\s*([\s\S]*?)\s*```", txt, flags=re.I)
    if fence:
        txt = fence[0].strip()
    txt = re.sub(r"^`+|`+$", "", txt).strip()
    return txt

def clean_model_text(s: str) -> str:
    return _strip_code_fences(s or "")

def count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total

# ---------------- Ark 同步调用（带重试 & thinking） ----------------
def call_model_ark(client: Ark,
                   model_id: str,
                   messages: list[dict],
                   qid: str,
                   temperature: float,
                   top_p: float,
                   retries: int,
                   backoff: float,
                   thinking_type: str) -> Tuple[bool, str, str]:
    """
    返回 (ok, content, reasoning_content)
    - ok=False 时，content 存放错误字符串（带 [ERROR]），reasoning_content 为空
    """
    headers = {CLIENT_REQUEST_HEADER: f"reasoning-{_safe_header_token(qid)}"} if qid else None

    thinking_payload = None
    t = (thinking_type or "").lower()
    if t in ("disabled", "enabled", "auto"):
        thinking_payload = {"type": t}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                extra_headers=headers,
                thinking=thinking_payload if thinking_payload else None,
            )
            final_text = ""
            reasoning_text = ""

            if resp and getattr(resp, "choices", None):
                msg = resp.choices[0].message

                # 最终文本
                try:
                    final_text = getattr(msg, "content", "") or msg.get("content", "")
                except Exception:
                    try:
                        final_text = msg["content"]
                    except Exception:
                        final_text = ""

                # Reasoning 内容（尽力获取）
                candidates = []
                for key in ("reasoning_content", "reasoning", "thoughts", "chain_of_thought"):
                    try:
                        v = getattr(msg, key, None)
                    except Exception:
                        v = msg.get(key) if isinstance(msg, dict) else None
                    if isinstance(v, str) and v.strip():
                        candidates.append(v)
                    elif isinstance(v, list):
                        parts = []
                        for it in v:
                            if isinstance(it, dict):
                                txt = it.get("text") or it.get("content") or ""
                                if isinstance(txt, str) and txt.strip():
                                    parts.append(txt.strip())
                        if parts:
                            candidates.append("\n".join(parts))
                if candidates:
                    reasoning_text = max(candidates, key=len)
            else:
                final_text = str(resp)

            return True, clean_model_text(final_text), clean_model_text(reasoning_text)
        except BaseException as e:
            last_err = e
            if attempt < retries:
                time.sleep((backoff ** (attempt - 1)))

    return False, f"[ERROR] {type(last_err).__name__}: {last_err}", ""

# ---------------- 质检调用 ----------------
def qc_build_messages(answer: str, content: str):
    user = QC_USER_TEMPLATE.format(gold_answer=answer or "", content=content or "")
    return [
        {"role": "system", "content": QC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def qc_parse_json_any(s: str):
    if not s:
        return None
    s = _strip_code_fences(s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def qc_call_model(client: Ark, messages, qid: str,
                  retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF,
                  temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    headers = {CLIENT_REQUEST_HEADER: f"qc-{_safe_header_token(qid)}"} if qid else None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                extra_headers=headers
            )
            text = getattr(resp.choices[0].message, "content", "") if resp.choices else ""
            obj = qc_parse_json_any(text)
            if not isinstance(obj, dict):
                return False, {"pass": False, "confidence": 0.0, "rationale": f"[PARSE] {text[:120]}"}
            return True, {
                "pass": bool(obj.get("pass", False)),
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "rationale": (obj.get("rationale") or "")[:300]
            }
        except BaseException as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** (attempt - 1))
    return False, {"pass": False, "confidence": 0.0, "rationale": f"[ERROR] {last_err}"}

# ---------------- 单条任务处理（含自纠环路） ----------------
def make_gen_messages(question_md: str, given_answer: str = "", options_md: Optional[str] = None) -> list[dict]:
    user = PROMPT_USER_TEMPLATE.format(
        question_md=(question_md or "").strip(),
        options_md=(options_md if options_md is not None else "[]"),
        given_answer=(given_answer or "").strip()
    )
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user},
    ]

def make_correction_messages(question_md: str, given_answer: str, options_md: Optional[str] = None) -> list[dict]:
    user = CORRECTION_USER_TEMPLATE.format(
        question_md=(question_md or "").strip(),
        options_md=(options_md if options_md is not None else "[]"),
        given_answer=(given_answer or "").strip()
    )
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user},
    ]

def process_one_line(idx: int,
                     raw_line: str,
                     temperature: float,
                     top_p: float,
                     retries: int,
                     backoff: float,
                     thinking_type: str,
                     max_correct_rounds: int,
                     client: Ark) -> Tuple[int, Dict[str, Any]]:
    s = raw_line.strip()
    if not s:
        return idx, {"content": "[PARSE] 空行", "reasoning_content": ""}

    try:
        obj = json.loads(s)
    except Exception:
        return idx, {"content": "[PARSE] 无法解析该行 JSON", "reasoning_content": ""}

    qid = obj.get("qid", "") or obj.get("id", "") or str(idx)
    question_md = obj.get("question", "") or ""
    given_answer = (obj.get("answer", "") or "").strip()
    options_md  = obj.get("options_md", "[]")

    # 若没有标准答案，则不走质检自纠；直接生成一次
    if not given_answer:
        ok, content, reasoning = call_model_ark(
            client=client,
            model_id=MODEL_ID,
            messages=make_gen_messages(question_md, given_answer, options_md),
            qid=qid,
            temperature=temperature,
            top_p=top_p,
            retries=retries,
            backoff=backoff,
            thinking_type=thinking_type
        )
        merged = dict(obj)
        merged["content"] = content
        merged["reasoning_content"] = reasoning if ok else ""
        merged["_qc"] = {"skipped": True, "rounds": 0}
        return idx, merged

    # 有标准答案：生成 -> 质检 -> 如不通过则纠偏重写，直至通过或达上限
    rounds = 0
    last_verdict = {}
    content = ""
    reasoning = ""
    ok_generate = False

    # 第一次生成（严格按答案去推理）
    ok_generate, content, reasoning = call_model_ark(
        client=client,
        model_id=MODEL_ID,
        messages=make_gen_messages(question_md, given_answer, options_md),
        qid=qid,
        temperature=temperature,
        top_p=top_p,
        retries=retries,
        backoff=backoff,
        thinking_type=thinking_type
    )

    # 质检
    qc_ok, verdict = qc_call_model(
        client=client,
        messages=qc_build_messages(given_answer, content),
        qid=str(qid),
        retries=retries,
        backoff=backoff,
        temperature=0.0,
        top_p=0.9
    )
    last_verdict = verdict
    pass_flag = verdict.get("pass", False)

    # 自纠环路
    while not pass_flag and rounds < max_correct_rounds:
        rounds += 1
        # 用纠偏提示重写推理
        ok_generate, content, reasoning = call_model_ark(
            client=client,
            model_id=MODEL_ID,
            messages=make_correction_messages(question_md, given_answer, options_md),
            qid=f"{qid}-fix{rounds}",
            temperature=temperature,
            top_p=top_p,
            retries=retries,
            backoff=backoff,
            thinking_type=thinking_type
        )
        qc_ok, verdict = qc_call_model(
            client=client,
            messages=qc_build_messages(given_answer, content),
            qid=f"{qid}-qc{rounds}",
            retries=retries,
            backoff=backoff,
            temperature=0.0,
            top_p=0.9
        )
        last_verdict = verdict
        pass_flag = verdict.get("pass", False)

    merged = dict(obj)
    merged["content"] = content
    merged["reasoning_content"] = reasoning if ok_generate else ""
    merged["_qc"] = {
        "pass": bool(last_verdict.get("pass", False)),
        "confidence": float(last_verdict.get("confidence", 0.0)),
        "rounds": rounds,
        "rationale": last_verdict.get("rationale", "")[:300]
    }
    return idx, merged

# ---------------- 主流程：顺序写出 ----------------
def main():
    ap = argparse.ArgumentParser(description="按标准答案生成推理 + 模型质检自纠（线程池并发，顺序写出）")
    ap.add_argument("--input-jsonl", required=True, help="输入 jsonl（原题目）")
    ap.add_argument("--out-jsonl",   required=True, help="输出 jsonl（原对象 + content + reasoning_content + _qc）")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="线程池大小")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p",       type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--retries",     type=int,   default=DEFAULT_RETRIES, help="每条请求重试次数")
    ap.add_argument("--backoff",     type=float, default=DEFAULT_BACKOFF, help="指数退避系数")
    ap.add_argument("--thinking",    choices=["disabled","enabled","auto"], default=DEFAULT_THINKING,
                    help="深度思考：disabled|enabled|auto（默认 disabled）")
    ap.add_argument("--max-correct-rounds", type=int, default=DEFAULT_MAX_CORRECT,
                    help="当与标准答案不一致时的自纠最多轮数（默认 8）")
    args = ap.parse_args()

    in_path  = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)

    if not in_path.exists():
        raise SystemExit(f"❌ 输入不存在：{in_path}")

    total = count_lines(in_path)
    print(f"[INFO] 待处理条目总数: {total}")
    print(f"[INFO] thinking = {args.thinking}  | max_correct_rounds = {args.max_correct_rounds}")

    if not (API_KEY or "").strip():
        raise SystemExit("❌ 缺少 Ark API Key")
    client = Ark(api_key=(API_KEY or "").strip())

    next_to_write = 0
    buffer: Dict[int, Dict[str, Any]] = {}

    success_cnt = 0
    parse_fail_cnt = 0
    error_cnt = 0
    corrected_cnt = 0
    qc_fail_after_limit = 0

    with in_path.open("r", encoding="utf-8") as fr, out_path.open("w", encoding="utf-8") as fw:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {}
            idx = 0
            for line in fr:
                if not line.strip():
                    # 依旧并发提交，让顺序写更简单
                    futures[ex.submit(
                        process_one_line, idx, line,
                        args.temperature, args.top_p,
                        args.retries, args.backoff,
                        args.thinking, args.max_correct_rounds, client
                    )] = idx
                    idx += 1
                    continue
                futures[ex.submit(
                    process_one_line, idx, line,
                    args.temperature, args.top_p,
                    args.retries, args.backoff,
                    args.thinking, args.max_correct_rounds, client
                )] = idx
                idx += 1

            with tqdm(total=total, desc="COT + QC self-correct", unit="q", dynamic_ncols=True) as pbar:
                for fut in cf.as_completed(futures):
                    i, out_obj = fut.result()
                    buffer[i] = out_obj
                    # 按序写出
                    while next_to_write in buffer:
                        obj = buffer.pop(next_to_write)
                        c = obj.get("content", "")
                        qc = obj.get("_qc", {}) or {}
                        rounds = int(qc.get("rounds", 0))

                        if isinstance(c, str) and c.startswith("[PARSE]"):
                            parse_fail_cnt += 1
                        elif isinstance(c, str) and c.startswith("[ERROR]"):
                            error_cnt += 1
                        else:
                            success_cnt += 1
                            if rounds > 0:
                                corrected_cnt += 1
                            if qc.get("pass") is False and rounds >= args.max_correct_rounds:
                                qc_fail_after_limit += 1

                        fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        next_to_write += 1
                        pbar.update(1)

    print("\n===== 汇总统计 =====")
    print(f"总条目: {total}")
    print(f"成功写出: {success_cnt}")
    print(f"解析失败: {parse_fail_cnt}  （空行/JSON解析失败等）")
    print(f"请求错误: {error_cnt}      （请求异常/服务端错误等）")
    print(f"触发自纠: {corrected_cnt}  （至少1轮纠偏）")
    print(f"仍未通过且达上限: {qc_fail_after_limit}  （建议人工复核或增大 --max-correct-rounds）")
    print(f"输出: {out_path}")

if __name__ == "__main__":
    main()