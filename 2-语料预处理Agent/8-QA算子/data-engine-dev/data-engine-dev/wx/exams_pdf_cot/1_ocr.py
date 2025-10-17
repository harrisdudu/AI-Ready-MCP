#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_mineru.py
批量并发运行 mineru，对输入目录下的 PDF 逐个处理，并把请求轮询分发到端口池（多 Docker 实例）。

示例：
  python batch_mineru.py /data/pdfs /data/mineru_out \
    --pattern "*.pdf" --max-workers 8 --resume --retries 2 --timeout 1800 \
    --url-host 172.16.0.28 --ports 30000-30007

也支持离散端口：
    --ports 30000,30002,30005
"""

import os
import sys
import shlex
import csv
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
from threading import Lock

DEFAULT_BACKEND = "vlm-sglang-client"

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Batch run mineru on a folder of PDFs (port-pool round-robin)")
    p.add_argument("input_dir", help="输入目录（递归扫描）")
    p.add_argument("output_dir", help="输出根目录")
    p.add_argument("--pattern", default="*.pdf", help="文件匹配规则（默认 *.pdf）")
    p.add_argument("--max-workers", type=int, default=4, help="并发数（默认4）")
    p.add_argument("--resume", action="store_true", help="断点续跑：若目标已存在且非空则跳过")
    p.add_argument("--retries", type=int, default=1, help="失败重试次数（默认1）")
    p.add_argument("--timeout", type=int, default=1800, help="单任务超时秒数（默认1800=30min）")
    p.add_argument("--log-file", default=f"mineru_batch_{timestamp}.log", help="批处理日志文件")
    p.add_argument("--summary-csv", default=f"mineru_summary_{timestamp}.csv", help="结果汇总CSV")
    p.add_argument("--extra", default="", help="给 mineru 追加的原样参数字符串，如：\"--xx 1 --yy abc\"")
    # 新增：端口池 & 主机
    p.add_argument("--url-host", default="172.16.0.28", help="mineru 服务主机/IP（默认 172.16.0.28）")
    p.add_argument("--ports", default="30000-30007",
                   help="端口列表或区间，例：'30000-30007' 或 '30000,30002,30005'")
    return p.parse_args()

def which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        c = Path(p) / cmd
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None

def parse_ports(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    # comma
    return [int(x) for x in spec.split(",") if x.strip()]

def is_done(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.iterdir())

def run_once(pdf_path: Path, out_dir: Path, timeout: int, extra_args: str,
             log_f: Path, backend_url: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_log = out_dir / "mineru_run.log"

    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "-b", DEFAULT_BACKEND,
        "-u", backend_url,
    ]
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    start = time.time()
    with per_log.open("a", encoding="utf-8") as plog, log_f.open("a", encoding="utf-8") as blog:
        plog.write(f"[{datetime.now()}] URL={backend_url} CMD: {' '.join(shlex.quote(c) for c in cmd)}\n")
        blog.write(f"[{datetime.now()}] START {pdf_path} url={backend_url}\n")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
            duration = time.time() - start
            plog.write(proc.stdout or "")
            plog.write(f"\n[duration_sec]={duration:.2f}, [returncode]={proc.returncode}\n")
            blog.write(f"[{datetime.now()}] END {pdf_path} code={proc.returncode} dur={duration:.2f}s url={backend_url}\n")
            return proc.returncode, duration, proc.stdout
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            msg = f"[TIMEOUT] after {duration:.2f}s URL={backend_url}\n"
            with per_log.open("a", encoding="utf-8") as plog2:
                plog2.write(msg)
            with log_f.open("a", encoding="utf-8") as blog2:
                blog2.write(f"[{datetime.now()}] TIMEOUT {pdf_path} after {duration:.2f}s url={backend_url}\n")
            return 124, duration, msg
        except Exception as e:
            duration = time.time() - start
            err = f"[EXCEPTION] {type(e).__name__}: {e} URL={backend_url}\n"
            with per_log.open("a", encoding="utf-8") as plog2:
                plog2.write(err)
            with log_f.open("a", encoding="utf-8") as blog2:
                blog2.write(f"[{datetime.now()}] EXCEPTION {pdf_path}: {e} url={backend_url}\n")
            return 1, duration, err

class PortRoundRobin:
    def __init__(self, host: str, ports: list[int]):
        self._cycle = cycle([f"http://{host}:{p}" for p in ports])
        self._lock = Lock()
    def next(self) -> str:
        with self._lock:
            return next(self._cycle)

def make_worker(args, in_root: Path, out_root: Path, log_f: Path, rr: PortRoundRobin):
    def worker(pdf_path: Path):
        rel = pdf_path.relative_to(in_root)
        out_dir = (out_root / rel).with_suffix("")
        if args.resume and is_done(out_dir):
            return {
                "pdf": str(pdf_path),
                "out_dir": str(out_dir),
                "status": "skipped_resume",
                "returncode": 0,
                "duration_sec": 0.0,
                "tries": 0,
                "url": ""
            }
        last_rc = None
        total_dur = 0.0
        tries = 0
        for i in range(args.retries + 1):
            tries = i + 1
            backend_url = rr.next()             # 轮询一个 URL
            rc, dur, _ = run_once(pdf_path, out_dir, args.timeout, args.extra, log_f, backend_url)
            total_dur += dur
            last_rc = rc
            if rc == 0:
                break
            time.sleep(min(5, i * 2))           # 简单退避
        status = "ok" if last_rc == 0 else "failed"
        return {
            "pdf": str(pdf_path),
            "out_dir": str(out_dir),
            "status": status,
            "returncode": last_rc,
            "duration_sec": round(total_dur, 2),
            "tries": tries,
            "url": backend_url if last_rc == 0 else ""
        }
    return worker

if __name__ == "__main__":
    args = parse_args()

    mineru_path = which("mineru")
    if not mineru_path:
        print("❌ 未找到 mineru 可执行文件，请确认已安装并在 PATH 中。", file=sys.stderr)
        sys.exit(2)

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
        
    out_root.mkdir(parents=True, exist_ok=True)
    log_f = Path(args.log_file).resolve()

    ports = parse_ports(args.ports)
    if not ports:
        print("❌ 端口池为空，请检查 --ports 参数。", file=sys.stderr)
        sys.exit(2)

    print(f"▶️ URL host: {args.url_host}, ports: {ports}")
    rr = PortRoundRobin(args.url_host, ports)

    pdfs = sorted(in_root.rglob(args.pattern))
    if not pdfs:
        print(f"⚠️ 未找到匹配文件：{args.pattern} in {in_root}", file=sys.stderr)
        sys.exit(0)

    print(f"▶️ 发现 {len(pdfs)} 个待处理文件，并发执行（workers={args.max_workers}）")

    worker = make_worker(args, in_root, out_root, log_f, rr)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(worker, pdf) for pdf in pdfs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            mark = "✅" if res["status"] == "ok" else ("⏭️" if res["status"] == "skipped_resume" else "❌")
            print(f"{mark} {res['pdf']} -> {res['status']} (rc={res['returncode']}, tries={res['tries']}, {res['duration_sec']}s)")

    csv_path = Path(args.summary_csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=["pdf", "out_dir", "status", "returncode", "tries", "duration_sec", "url"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skipped_resume")
    fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n📊 总结：total={total}, ok={ok}, skipped={skip}, failed={fail}")
    print(f"🧾 汇总表：{csv_path}")
    print(f"🗒️ 日志：{log_f}")


# python 1_ocr.py /home/wangxi/workspace/gongye/一级造价 /home/wangxi/workspace/gongye/一级造价/mineru_ocred --pattern "*.pdf" --max-workers 8 --resume --retries 2 --timeout 1800 --url-host 172.16.0.28 --ports 30000-30007