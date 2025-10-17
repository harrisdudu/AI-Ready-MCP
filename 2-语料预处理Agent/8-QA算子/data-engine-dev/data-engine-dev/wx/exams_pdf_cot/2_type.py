#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 *.md 文件的前50行 + 后50行文本片段，判断四类试题版式：
1) ONLY_QUESTIONS                -> 只有题
2) ONLY_ANSWERS_EXPLANATIONS     -> 只有答案和解析
3) MIXED_TOGETHER                -> 题和答案解析在一起
4) ANSWERS_AT_END                -> 答案和解析在最后

用法示例：
  python 2_type.py "/home/wangxi/workspace/gongye/一级造价/mineru_ocred" \
    --recursive --pattern "*.md" --max-workers 8

依赖：
  pip install tqdm rich pandas "volcengine-python-sdk[ark]"
"""

import os
import re
import json
import argparse
import concurrent.futures as cf
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from tqdm import tqdm
from rich import print
import pandas as pd
import time
from datetime import datetime


# ========== 写死的 Ark 配置 ==========
API_KEY   = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID  = "doubao-seed-1-6-250615"
BACKEND   = "ark"  # 固定

# 参数默认值
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P       = 0.9
DEFAULT_TIMEOUT     = 60
DEFAULT_MAX_CHARS   = 6000

# Ark SDK
try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    raise SystemExit("❌ 需要安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

LABELS = {
    "ONLY_QUESTIONS": "只有题",
    "ONLY_ANSWERS_EXPLANATIONS": "只有答案和解析",
    "MIXED_TOGETHER": "题和答案解析在一起",
    "ANSWERS_AT_END": "答案和解析在最后",
}

SYSTEM_PROMPT = (
    "你是一个严谨的试卷版式分类助手。你将看到某个Markdown文件的片段（来自前50行和后50行）。"
    "请仅基于这些片段判断整份试卷的题与答案解析组织形式，并输出JSON："
    '{"type":"<四选一标签>","confidence":0.0~1.0,"rationale":"简述依据要点"}'
    "四个合法标签："
    "ONLY_QUESTIONS（只有题），"
    "ONLY_ANSWERS_EXPLANATIONS（只有答案或解析），"
    "MIXED_TOGETHER（题和答案解析在一起），"
    "ANSWERS_AT_END（答案和解析在最后）。"
)

USER_PROMPT_TEMPLATE = """请阅读以下片段（按原始行顺序拼接）：

[前50行片段]
----------------
{front_excerpt}

[后50行片段]
----------------
{back_excerpt}

判定标准提示：
- 若只有题干、题号、选项等，无答案/解析等描述，倾向 ONLY_QUESTIONS。
- 若几乎只有答案/解析/参考答案/参考等类似描述，无题干，倾向 ONLY_ANSWERS_EXPLANATIONS。
- 若同一页或紧邻处既有提干又有解析，倾向 MIXED_TOGETHER。
- 若前部多为题干，末尾集中出现答案/解析/参考答案等描述，倾向 ANSWERS_AT_END。
- 答案有可能在表格里出现，需要结合表格内容判断。
- 有一些前面有一些前言介绍的，可能不是题，实际还是只有答案，要注意。

请只输出一个 JSON，且字段完整，不要额外文本。
"""

@dataclass
class ClassificationResult:
    file: str
    label: str
    label_cn: str
    confidence: float
    backend: str
    heuristic_used: bool
    elapsed_sec: float
    rationale: str
    front_preview: str
    back_preview: str
    raw_model_output: Optional[str] = None

# ---------------- Markdown 读与抽取 ----------------
def load_markdown(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def pick_front_back_lines(lines: List[str], n_each: int = 50) -> Tuple[str, str]:
    if not lines:
        return "", ""
    
    # 前n_each行
    front_lines = lines[:n_each]
    front = "".join(front_lines).strip()
    
    # 后n_each行
    back_lines = lines[-n_each:] if len(lines) > n_each else []
    back = "".join(back_lines).strip()
    
    return front, back

# ---------------- 模型调用（Ark） ----------------
def call_model_ark(model_id: str, system: str, user: str,
                   temperature: float, top_p: float, timeout: int) -> str:
    key = (API_KEY or "").strip()
    if not key:
        raise RuntimeError("缺少 Ark API Key。")
    client = Ark(api_key=key)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        top_p=top_p,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def parse_model_json(s: str) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    txt = s.strip()
    m = re.findall(r"```json(.*?)```", txt, flags=re.S | re.I)
    if m:
        txt = m[0].strip()
    m2 = re.search(r"\{.*\}", txt, flags=re.S)
    if m2:
        cand = m2.group(0)
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(txt)
    except Exception:
        return None

# ---------------- 启发式兜底 ----------------
def heuristic_guess(front: str, back: str) -> Tuple[str, float, str]:
    def score(tokens: List[str], text: str) -> int:
        s = 0
        for t in tokens:
            s += len(re.findall(re.escape(t), text, flags=re.I))
        return s

    q_tokens = ["题目", "试题", "单选", "多选", "判断", "案例", "问答", "A.", "B.", "C.", "D.", "【题干】"]
    a_tokens = ["答案", "参考答案", "解析", "解答", "【答案】", "【解析】", "评分标准", "正确答案"]

    fq, fa = score(q_tokens, front), score(a_tokens, front)
    bq, ba = score(q_tokens, back), score(a_tokens, back)
    rationale = f"计数→ 前:题={fq} 答/解={fa}; 后:题={bq} 答/解={ba}"

    if (fq + bq) > 0 and (fa + ba) == 0:
        return "ONLY_QUESTIONS", 0.6, rationale
    if (fa + ba) > 0 and (fq + bq) == 0:
        return "ONLY_ANSWERS_EXPLANATIONS", 0.6, rationale
    if (fq > 0 and fa > 0) or (bq > 0 and ba > 0):
        return "MIXED_TOGETHER", 0.55, rationale
    if fq > 0 and (ba > fa + 1):
        return "ANSWERS_AT_END", 0.65, rationale

    return "MIXED_TOGETHER", 0.5, rationale + "（默认猜测）"

# ---------------- 主分类逻辑 ----------------
def classify_one(md_path: Path,
                 temperature: float = DEFAULT_TEMPERATURE,
                 top_p: float = DEFAULT_TOP_P,
                 timeout: int = DEFAULT_TIMEOUT,
                 max_chars: int = DEFAULT_MAX_CHARS) -> ClassificationResult:
    t0 = time.time()
    try:
        lines = load_markdown(md_path)
    except Exception as e:
        raise RuntimeError(f"读取 Markdown 失败：{md_path} -> {e}")

    front, back = pick_front_back_lines(lines, 50)

    front_cut = (front or "")[:max_chars]
    back_cut  = (back or "")[:max_chars]

    user_prompt = USER_PROMPT_TEMPLATE.format(front_excerpt=front_cut, back_excerpt=back_cut)

    raw = None
    label, conf, rationale = None, 0.0, ""

    try:
        raw = call_model_ark(MODEL_ID, SYSTEM_PROMPT, user_prompt, temperature, top_p, timeout)
        obj = parse_model_json(raw)
        if isinstance(obj, dict) and "type" in obj:
            t = obj.get("type")
            c = float(obj.get("confidence", 0.0) or 0.0)
            r = str(obj.get("rationale", ""))[:2000]
            if t in LABELS:
                label, conf, rationale = t, max(0.0, min(1.0, c)), r
    except Exception as e:
        raw = f"[MODEL_ERROR] {e}"

    heuristic_used = False
    if label is None:
        heuristic_used = True
        label, conf, heur_r = heuristic_guess(front_cut, back_cut)
        rationale = (rationale + " | " if rationale else "") + f"[heuristic] {heur_r}"

    elapsed = time.time() - t0
    return ClassificationResult(
        file=str(md_path),
        label=label,
        label_cn=LABELS.get(label, label),
        confidence=conf,
        backend=BACKEND,
        heuristic_used=heuristic_used,
        elapsed_sec=elapsed,
        rationale=rationale,
        front_preview=front_cut[:1000],
        back_preview=back_cut[:1000],
        raw_model_output=(raw[:4000] if isinstance(raw, str) else None),
    )

# ---------------- 扫描与主函数 ----------------
def iter_md_files(path: Path, recursive: bool, pattern: str) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".md":
        return [path]
    glob_pat = f"**/{pattern}" if recursive else pattern
    return sorted(path.glob(glob_pat))

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="基于 *.md 文件的前50行+后50行判定试卷类型（四类, Ark 固定）")
    ap.add_argument("path", help="单个 Markdown 文件或目录")
    ap.add_argument("--recursive", action="store_true", help="递归扫描目录")
    ap.add_argument("--pattern", default="*.md", help="文件名匹配模式（默认 *.md）")
    ap.add_argument("--max-workers", type=int, default=256, help="并发线程数")
    ap.add_argument("--out-csv", default=f"classification_results_{timestamp}.csv")
    ap.add_argument("--out-jsonl", default=f"classification_results_{timestamp}.jsonl")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="前/后片段截断长度")
    args = ap.parse_args()

    root = Path(args.path)
    files = iter_md_files(root, args.recursive, args.pattern)
    if not files:
        print(f"[red]没有匹配到 Markdown 文件：{root} (pattern={args.pattern})[/red]")
        return

    results: List[ClassificationResult] = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(
                classify_one,
                md_path=fp,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                max_chars=args.max_chars,
            )
            for fp in files
        ]
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc="Classifying"):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[yellow]处理失败：{e}[/yellow]")

    rows = [asdict(r) for r in results]
    if rows:
        # CSV（核心字段）
        df = pd.DataFrame([{
            "file": r["file"],
            "label": r["label"],
            "label_cn": r["label_cn"],
            "confidence": r["confidence"],
            "backend": r["backend"],
            "heuristic_used": r["heuristic_used"],
            "elapsed_sec": r["elapsed_sec"],
        } for r in rows])
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

        # JSONL（全量）
        with open(args.out_jsonl, "w", encoding="utf-8") as fw:
            for r in rows:
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[green]✅ 完成：{len(rows)} 个文件[/green]")
        print(f"CSV → {args.out_csv}")
        print(f"JSONL → {args.out_jsonl}")

        # ===== 新增：分类统计 =====
        label_counts = Counter(r["label"] for r in rows)
        total_files = len(rows)
        print("\n[cyan]📊 分类统计：[/cyan]")
        # 固定顺序输出，方便对齐
        order = ["ONLY_QUESTIONS","ONLY_ANSWERS_EXPLANATIONS","MIXED_TOGETHER","ANSWERS_AT_END"]
        for lbl in order:
            cnt = label_counts.get(lbl, 0)
            print(f"  {lbl:<28} {LABELS.get(lbl, lbl):<20} {cnt} 个 ({(cnt/total_files if total_files else 0):.1%})")

    else:
        print("[yellow]没有结果可导出[/yellow]")

if __name__ == "__main__":
    main()


# python 2_type.py "/home/wangxi/workspace/gongye/yijizaojia/mineru_ocred" --recursive --pattern "*.md" --max-workers 8