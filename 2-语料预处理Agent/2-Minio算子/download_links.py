import csv
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import requests

BASE = "http://172.16.0.144:8088/annotation-platform-api/api"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Authorization": "51HEyiMzNH5Bb90erCGwZFpChC9looJOCBMgum7uGGsJyspkqPgi5iA9dL1E",
    "Cookie": "corpus-token=ca536612-e2d0-439c-b011-22b979625798",
    "Origin": "http://172.16.0.144:8088",
    "Referer": "http://172.16.0.144:8088/project",
    "Uid": "28",
    "User-Agent": "client/1.0",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def fetch_projects(page_size: int = 50) -> List[Dict[str, Any]]:
    url = f"{BASE}/project/list"
    params = {
        "keyword": "",
        "order_field": "created_at",
        "order_desc": "true",
        "page": 1,
        "page_size": page_size,
        "project_type": 1,
    }
    r = SESSION.post(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not (data.get("successful") and data.get("code") == 200000):
        raise RuntimeError(f"获取项目列表失败: {data}")
    return data["data"]["results"]


def choose_project_ids() -> List[int]:
    projects = fetch_projects()
    if not projects:
        print("当前无项目可选"); return []
    print("\n可选项目：")
    for idx, p in enumerate(projects, 1):
        print(f"{idx}. {p['name']}  (id={p['id']})")
    raw = input("\n请输入序号，可多选，逗号分隔： ").strip()
    nums = {int(x) for x in raw.split(",") if x.strip().isdigit()}
    return [projects[i - 1]["id"] for i in nums if 1 <= i <= len(projects)]


def fetch_all_assignments(project_id: int, page_size: int = 50) -> List[int]:
    url = f"{BASE}/assignment/list"
    params = {"page": 1, "page_size": page_size, "project_id": project_id}
    r = SESSION.post(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    if not (data.get("successful") and data.get("code") == 200000):
        raise RuntimeError(f"assignment/list 接口异常: {data}")
    return [itm["id"] for itm in data["data"]["results"]]


def fetch_all_datasets_for_assignment(assignment_id: int, page_size: int = 50) -> List[int]:
    url = f"{BASE}/assignment/datasets"
    params = {"page": 1, "page_size": page_size, "keyword": "", "assignment_id": assignment_id}
    r = SESSION.post(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    if not (data.get("successful") and data.get("code") == 200000):
        raise RuntimeError(f"assignment/datasets 接口异常: {data}")
    return [d["id"] for d in data["data"]["results"]]


def find_all_assignment_dataset_pairs(project_id: int) -> List[Dict[str, int]]:
    pairs = []
    for aid in fetch_all_assignments(project_id):
        for dsid in fetch_all_datasets_for_assignment(aid):
            pairs.append({"project_id": project_id, "assignment_id": aid, "dataset_id": dsid})
    return pairs


def pack_datasets(pairs: List[Dict[str, int]], status: int = 0, download_file_type: int = 0) -> List[Dict[str, Any]]:
    url = f"{BASE}/assignment/datasets/pack"
    out = []
    for p in pairs:
        payload = {"assignment_id": p["assignment_id"], "dataset_id": p["dataset_id"],
                   "status": status, "download_file_type": download_file_type}
        r = SESSION.post(url, json=payload, timeout=30); r.raise_for_status()
        data = r.json()
        if not (data.get("successful") and data.get("code") == 200000):
            raise RuntimeError(f"pack 接口异常: {data}")
        out.append({**p, "download_id": data["data"]["download_id"]})
    return out


def resolve_download_url(download_id: str, poll_interval: int = 2, timeout: int = 180) -> str:
    url = f"{BASE}/assignment/audios/package/{download_id}"
    start = time.time()
    while True:
        r = SESSION.get(url, timeout=15); r.raise_for_status()
        data = r.json()
        if not (data.get("successful") and data.get("code") == 200000):
            raise RuntimeError(f"package 解析异常: {data}")
        if data["data"]["is_ok"]:
            return data["data"]["url"]
        if time.time() - start > timeout:
            raise TimeoutError(f"{download_id} 等待超时")
        time.sleep(poll_interval)


def export_to_csv(records: List[Dict[str, Any]], csv_path: str):
    fieldnames = ["project_id", "assignment_id", "dataset_id", "download_id", "url"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(records)


# ==================== 读取 CSV 并下载 ====================
def safe_name(name: str, used: set) -> str:
    base = _illegal.sub("_", name).strip()
    n, idx = base, 1
    while n.lower() in used:
        n = f"{base}-{idx}"; idx += 1
    used.add(n.lower()); return n + ".zip"


def fetch_assignment_name_map(project_id: int) -> Dict[int, str]:
    url = f"{BASE}/assignment/list"
    params = {"page": 1, "page_size": 50, "project_id": project_id}
    r = SESSION.post(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    if not (data.get("successful") and data.get("code") == 200000):
        raise RuntimeError(f"assignment/list 失败: {data}")
    return {itm["id"]: itm["name"] for itm in data["data"]["results"]}


def download_file(url: str, dst: Path, chunk: int = 8192):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with SESSION.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for blk in r.iter_content(chunk):
                if blk: f.write(blk)

_illegal = re.compile(r'[\\/:*?"<>|]')

def _safe(text: str) -> str:
    """清理文件/目录非法字符"""
    return _illegal.sub("_", text).strip()

def download_by_csv(csv_path: str, out_root: str = "./zip_downloads"):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV 为空"); return

    proj_name_map = _project_name_map()          # {id: name}
    aid_name_cache: Dict[int, str] = {}          # assignment_id -> name

    # 按 project 分组，减少接口调用
    proj_groups: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        proj_groups.setdefault(int(r["project_id"]), []).append(r)

    existed_files = set()
    out_root = Path(out_root)

    for pid, group in proj_groups.items():
        proj_dir = out_root / _safe(proj_name_map.get(pid, f"project_{pid}"))
        # 缓存 assignment 名字
        if pid not in aid_name_cache:
            aid_name_cache.update(fetch_assignment_name_map(pid))

        for row in group:
            aid = int(row["assignment_id"])
            url = row["url"]
            ass_name = _safe(aid_name_cache.get(aid, f"assignment_{aid}"))

            # 避免同一项目下重名覆盖
            fname, idx = ass_name + ".zip", 1
            while (proj_dir / fname).exists():
                fname = f"{ass_name}-{idx}.zip"; idx += 1

            dst = proj_dir / fname
            print("下载 →", dst)
            try:
                download_file(url, dst)
            except Exception as e:
                print("  失败:", e)


def _project_name_map() -> Dict[int, str]:
    """返回 {project_id: project_name}"""
    url = f"{BASE}/project/list"
    params = {"keyword": "", "order_field": "created_at", "order_desc": "true",
              "page": 1, "page_size": 50, "project_type": 1}
    r = SESSION.post(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    if not (data.get("successful") and data.get("code") == 200000):
        raise RuntimeError(f"project/list 失败: {data}")
    return {p["id"]: p["name"] for p in data["data"]["results"]}

if __name__ == "__main__":
    selected = choose_project_ids()
    for pid in selected:
        pairs = find_all_assignment_dataset_pairs(pid)
        enriched = pack_datasets(pairs)
        for r in enriched:
            r["url"] = resolve_download_url(r["download_id"])
        csv_name = f"{pid}_download_links.csv"
        export_to_csv(enriched, csv_name)
        print("CSV 已生成:", csv_name)

        download_by_csv(csv_name, "./zip_downloads")
