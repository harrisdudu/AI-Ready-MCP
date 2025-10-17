#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

from tqdm import tqdm

# ==== 固定 Ark 配置 ====
API_KEY  = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

try:
    from volcenginesdkarkruntime import Ark
except Exception:
    print("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'", file=sys.stderr)
    sys.exit(1)

LLM_SYSTEM_PROMPT = """你是一名严谨的“题/答分界”判定专家。
输入是一段文档片段（Markdown，带绝对行号，形如 000123│）。
请判断这个片段里是否包含 **题区到答案区的分界点**。

如果包含，请返回答案区的**第一行绝对行号**；
如果不包含，请返回 null。

严格输出 JSON：
{
  "boundary_line": 整数或 null,
  "confidence": 0.0~1.0,
  "rationale": "简要原因"
}
"""

log_lock = Lock()
log_file_path = Path("json_parse_errors.log")
PAGE_SIZE = 300  # 每页行数


def log_json_error(file: str, attempt: int, raw_output: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | file: {file} | attempt: {attempt}\n{raw_output}\n{'-'*80}\n"
    with log_lock:
        log_file_path.open("a", encoding="utf-8").write(entry)


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_markdown_json(raw: str) -> str:
    raw = raw.replace("`", "'").replace("|", "-")
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\*{1,3}", "", raw)
    return raw


def safe_json_loads(raw: str, file: str = "", attempt: int = 0):
    raw_cleaned = clean_text(raw)
    raw_cleaned = sanitize_markdown_json(raw_cleaned)
    try:
        data = json.loads(raw_cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    s = raw_cleaned.find("{")
    e = raw_cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            data = json.loads(raw_cleaned[s:e + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    log_json_error(file, attempt, raw)
    return {}


def _format_with_line_numbers(lines: List[str]) -> str:
    return "\n".join(f"{i:06d}│{lines[i]}" for i in range(len(lines)))


def safe_write_json(path: Path, data: dict):
    def sanitize(obj):
        if isinstance(obj, str):
            return clean_text(obj)
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    path.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def classify_full_file(client: Ark, lines: List[str], args, file_name: str) -> (Optional[int], str):
    n = len(lines)
    # start_index = n // 2  # 只处理后一半
    start_index = 0
    raw_output = ""
    boundary_line = None

    for start in range(start_index, n, PAGE_SIZE):
        window_lines = lines[start:start+PAGE_SIZE]
        snippet = _format_with_line_numbers(window_lines)
        user_prompt = f"下面是文档片段（Markdown，行号为绝对0-based）：\n\n```\n{snippet}\n```"

        for i in range(args.repeat):
            try:
                print(f"📡 调用模型，尝试 {i+1}/{args.repeat}, 行 {start}-{start+len(window_lines)} ...")
                resp = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                )
                content = (resp.choices[0].message.content or "").strip()
                raw_output += content + "\n"
                data = safe_json_loads(content, file=file_name, attempt=i+1)
                bl = data.get("boundary_line", None)
                if bl is not None:
                    boundary_line = start + int(bl)  # 转成绝对行号
                    print(f"✅ {file_name} 找到分界线: {boundary_line}")
                    return boundary_line, raw_output  # 发现分界线立即返回，停止分页
            except Exception as e:
                print(f"❌ 模型调用异常: {e}")
                log_json_error(file_name, i+1, f"Exception: {e}\nRaw: {raw_output}")
                continue

    return boundary_line, raw_output


def process_file(md_path: Path, args, client: Ark):
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    n = len(lines)
    stem = md_path.stem

    file_out_dir = Path(args.out_dir) / stem
    file_out_dir.mkdir(parents=True, exist_ok=True)

    boundary_line, raw_model_output = classify_full_file(client, lines, args, md_path.name)

    if boundary_line is None:
        questions_text = clean_text("\n".join(lines))
        answers_text = ""
    else:
        questions_text = clean_text("\n".join(lines[:boundary_line]))
        answers_text = clean_text("\n".join(lines[boundary_line:]))

    (file_out_dir / "questions.md").write_text(questions_text, encoding="utf-8")
    (file_out_dir / "answers.md").write_text(answers_text, encoding="utf-8")

    report = {
        "file": str(md_path),
        "total_lines": n,
        "boundary_line": boundary_line,
        "raw_model_output": raw_model_output
    }
    safe_write_json(file_out_dir / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--jsonl-file", required=True, help="JSONL 文件，每行一个 JSON")
    p.add_argument("--pattern", default="*.md")
    p.add_argument("--out-dir", default="qna_split_out")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--file-concurrency", type=int, default=16)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    allowed_files = set()
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("label") == "ANSWERS_AT_END" and "file" in rec:
                    fname = Path(rec["file"]).stem
                    if fname.endswith("_content_list"):
                        fname = fname[:-len("_content_list")]
                    allowed_files.add(fname)
            except json.JSONDecodeError:
                continue

    client = Ark(api_key=API_KEY)
    root = Path(args.folder).resolve()
    all_md_files = sorted(root.rglob(args.pattern))
    files = [f for f in all_md_files if f.stem in allowed_files]

    if not files:
        print("⚠️ 没有找到符合 JSONL 条件的文件")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.file_concurrency) as ex:
        future_to_file = {ex.submit(process_file, f, args, client): f for f in files}
        for fut in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="文件级并发处理中"):
            try:
                rep = fut.result()
                results.append(rep)
            except Exception as e:
                print(f"❌ {future_to_file[fut]} 处理失败: {e}")

    (Path(args.out_dir) / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()   

# python3 3.1_qa_apart_split.py \
#   /home/wangxi/workspace/gongye/yijizaojia/mineru_ocred \
#   --jsonl-file /home/wangxi/workspace/gongye/yijizaojia/classification_results.jsonl \
#   --out-dir yijizaojia/qna_split \
#   --file-concurrency 256 \
#   --repeat 3 \
#   --timeout 120 \
#   --temperature 0.0 \
#   --top-p 0.9