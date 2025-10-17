#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qc_content_vs_answer_model_autoout.py

用模型判断 content 与标准答案 answer 是否一致
- 输入：--in-jsonl 传入路径
- 输出：自动生成同目录下的 .ok.jsonl / .bad.jsonl / .qc_report.csv
"""

import json
import re
import time
import concurrent.futures as cf
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import hashlib

# Ark 配置
API_KEY   = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID  = "doubao-seed-1-6-250615"
MAX_WORKERS = 256
RETRIES     = 5
BACKOFF     = 1.6
TEMPERATURE = 0.0
TOP_P       = 0.9

try:
    from volcenginesdkarkruntime import Ark
    from volcenginesdkarkruntime._constants import CLIENT_REQUEST_HEADER
except Exception:
    raise SystemExit("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

SYSTEM_PROMPT = (
    "你是一个严格的试题质检员。"
    "给定模型输出的 content 和标准答案 answer，你需要判断它们是否一致。"
    "只输出 JSON，字段如下："
    "{\"pass\": <bool>, \"confidence\": <0.0-1.0>, \"rationale\": \"简短说明理由\"}"
    "只输出 JSON，不要其他内容。"
)

USER_TEMPLATE = """请判断以下模型输出与标准答案是否一致，仅输出 JSON：
[标准答案]
{gold_answer}

[模型输出]
{content}
"""



def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    m = re.findall(r"```(?:json|text)?\s*([\s\S]*?)\s*```", s.strip(), flags=re.I)
    return m[0].strip() if m else s.strip()

def parse_json_any(s: str):
    if not s:
        return None
    s = strip_code_fences(s)
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

def build_messages(answer: str, content: str):
    user = USER_TEMPLATE.format(gold_answer=answer or "", content=content or "")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def _safe_header_token(s: str) -> str:
    s = s or ""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]  # 纯ASCII

def call_model(client: Ark, messages, qid: str):
    headers = {CLIENT_REQUEST_HEADER: f"qc-{_safe_header_token(qid)}"} if qid else None
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                extra_headers=headers
            )
            text = getattr(resp.choices[0].message, "content", "") if resp.choices else ""
            obj = parse_json_any(text)
            if not isinstance(obj, dict):
                return False, {"pass": False, "confidence": 0.0, "rationale": f"[PARSE] {text[:100]}"}
            return True, {
                "pass": bool(obj.get("pass", False)),
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "rationale": (obj.get("rationale") or "")[:200]
            }
        except BaseException as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(BACKOFF ** (attempt - 1))
    return False, {"pass": False, "confidence": 0.0, "rationale": f"[ERROR] {last_err}"}

def process_one(idx: int, raw_line: str, client: Ark):
    try:
        obj = json.loads(raw_line)
    except Exception:
        return idx, {"_error": "[PARSE] 非法JSON"}
    qid = obj.get("qid") or obj.get("id") or f"idx{idx}"
    answer = obj.get("answer", "")
    content = obj.get("content", "")
    ok, verdict = call_model(client, build_messages(answer, content), str(qid))
    obj["_qc_verdict"] = verdict
    return idx, obj

def main():
    import argparse
    ap = argparse.ArgumentParser(description="模型判断 content 与 answer 是否一致（输出路径自动生成）")
    ap.add_argument("--in-jsonl", required=True, help="输入文件路径（JSONL）")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"❌ 输入文件不存在：{in_path}")

    ok_path  = in_path.with_suffix("").with_name(in_path.stem + ".ok.jsonl")
    bad_path = in_path.with_suffix("").with_name(in_path.stem + ".bad.jsonl")
    csv_path = in_path.with_suffix("").with_name(in_path.stem + ".qc_report.csv")

    total = sum(1 for _ in in_path.open("r", encoding="utf-8"))
    print(f"[INFO] 待质检总数: {total}")

    client = Ark(api_key=(API_KEY or "").strip())
    buf = {}
    next_write = 0
    pass_cnt = 0
    rows = []

    with in_path.open("r", encoding="utf-8") as fr, \
         ok_path.open("w", encoding="utf-8") as fok, \
         bad_path.open("w", encoding="utf-8") as fbad:
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, i, line.strip(), client): i for i, line in enumerate(fr) if line.strip()}
            with tqdm(total=total, desc="QC model judge", unit="q", dynamic_ncols=True) as pbar:
                for fut in cf.as_completed(futures):
                    i, obj = fut.result()
                    buf[i] = obj
                    while next_write in buf:
                        cur = buf.pop(next_write)
                        next_write += 1
                        pbar.update(1)
                        verdict = cur.get("_qc_verdict", {})
                        is_pass = verdict.get("pass", False)
                        if is_pass:
                            pass_cnt += 1
                            fok.write(json.dumps(cur, ensure_ascii=False) + "\n")
                        else:
                            fbad.write(json.dumps(cur, ensure_ascii=False) + "\n")
                        rows.append({
                            "qid": cur.get("qid"),
                            "pass": is_pass,
                            "confidence": verdict.get("confidence", 0.0),
                            "rationale": verdict.get("rationale"),
                            "answer": cur.get("answer"),
                            "content": cur.get("content")
                        })

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n===== 模型质检统计 =====")
    print(f"总样本: {total}")
    print(f"通过数: {pass_cnt}  通过率: {pass_cnt/total:.1%}")
    print(f"OK 输出： {ok_path}")
    print(f"Bad 输出：{bad_path}")
    print(f"报告 CSV：{csv_path}")

if __name__ == "__main__":
    main()