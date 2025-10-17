#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
import random
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import concurrent.futures as cf


# ========= Ark 配置 =========
API_KEY  = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

try:
    from volcenginesdkarkruntime import Ark
except Exception:
    raise SystemExit("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

# ========= 参数常量 =========
DEFAULT_WINDOW_LINES = 80
DEFAULT_STRIDE_LINES = 60

# 单次请求最大字符预算
MAX_CHARS_PER_PROMPT = 9000
# Ark 默认请求超时（秒）
DEFAULT_ARK_TIMEOUT = 120
# 默认并发线程数（降低以避免文件描述符问题）
DEFAULT_MAX_WORKERS = 8

# ========= System/User Prompt =========
TOC_EXTRACT_PROMPT = """\
你是一个精确的文档目录结构提取助手。
我将给你一段从文档 PDF 转成 Markdown 的文本片段。
你的任务：从正文中提取标题结构。

要求：
1) 要以行为单位判断标题，整行作为整体判断，标题一般都比较短，不要提取长文本
2) 识别所有标题，提取完整的标题文本，包括编号和标题内容
    - 标题都是标题编号数字+标题内容的结构，不要提取没有数字的标题
    - 标题编号数字比如：1.、1.1、1.1.1、第一章、第一节、一、二、三、1、2、3、(1)、(2)、(3)、（一）、（二）、（三）、[1]、[2]、[3]、【一】、【二】、【三】、A.、B.、C.、a.、b.、c.、I.、II.、III.、i.、ii.、iii.等
3) 表格里的标题不要提取
4) 不要提取目录，只提取正文中的实际章节标题，如果遇到目录部分，直接跳过，不提取任何内容
5) 如果检测到附件、参考文献、附录等非正文内容，返回特殊标记 {"has_attachment": true}

输出格式：
- 正常情况下返回JSON数组，每个元素包含：
  {"title": 完整的这行标题文本（包含编号和标题内容）}
- 如果没有找到任何标题，返回空数组 []
- 如果检测到附件，返回 {"has_attachment": true}

只输出严格的JSON格式，不要包含任何额外文字。
不要编造内容，只提取实际存在的标题。
"""

TOC_VERIFY_PROMPT = """\
你是一个精确的文档目录验证助手。
我将给你一个已提取的标题结构列表和一段文档文本片段。
你的任务：检查这个目录结构是否完整和准确，并输出最终目录结构。

要求：
1) 仔细检查文档片段中是否还有遗漏的标题
2) 检查已提取的标题是否真的存在（不是误判）
3) 标题是滑动窗口拼接起来的，检查标题是否连贯，如果存在重复，只保留一个
3) 重点关注：
   - 是否有新的标题被遗漏
   - 是否有非标题内容被误判为标题
4) 只关注正文中的实际章节标题，忽略：
   - 目录部分
   - 表格中的标题
   - 附件、参考文献等非正文部分

输出格式：
- 返回JSON数组，包含最终的准确完整的标题列表
- 每个标题项包含：
  {"title": 完整的标题文本}

只输出严格的JSON格式，不要包含任何额外文字。
直接输出最终的标题数组，不要分类。
"""

# ========= 数据结构 =========
@dataclass
class TOCItem:
    """目录项"""
    title: str

@dataclass
class Section:
    """文档章节"""
    section_title: str
    content: str
    source_file: str

@dataclass
class ExtractStats:
    files_processed: int = 0
    files_copied: int = 0
    files_with_titles: int = 0
    title_items: int = 0
    sections_extracted: int = 0
    # 新增验证相关统计
    missing_titles_found: int = 0
    extra_titles_found: int = 0
    correct_titles_found: int = 0
    # 新增附件检测相关统计
    files_with_attachments: int = 0

# ========= Ark 调用 =========
def call_ark(system_prompt: str, user_prompt: str, api_key: str, model_id: str,
             temperature: float = 0.0, top_p: float = 0.9,
             timeout: int = DEFAULT_ARK_TIMEOUT) -> str:
    time.sleep(random.uniform(0.05, 0.20))  # 轻微抖动
    client = Ark(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

# ========= JSON 解析 + 清洗 =========
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    fences = ("```json", "```JSON", "```", "~~~json", "~~~JSON", "~~~")
    for f in fences:
        if s.startswith(f):
            s = s[len(f):].strip()
        if s.endswith(f):
            s = s[:-len(f)].strip()
    return s

def _clean_unicode_escapes(s: str) -> str:
    """
    清理Unicode转义序列，将\\u3001等转换为实际字符
    """
    # 处理常见的Unicode转义序列
    unicode_mappings = {
        '\\u3001': '、',  # 中文顿号
        '\\u3002': '。',  # 中文句号
        '\\u300a': '《',  # 中文左书名号
        '\\u300b': '》',  # 中文右书名号
        '\\u2014': '—',  # 破折号
        '\\u2018': ''',  # 左单引号
        '\\u2019': ''',  # 右单引号
        '\\u201c': '"',  # 左双引号
        '\\u201d': '"',  # 右双引号
    }
    
    for escape, char in unicode_mappings.items():
        s = s.replace(escape, char)
    
    return s

def force_json_load_with_sanitize(s: str) -> Tuple[Any, bool]:
    """
    返回 (obj, sanitized_used)
    """
    s0 = s.strip()
    
    # 清理Unicode转义序列
    s0 = _clean_unicode_escapes(s0)
    
    try:
        if s0.startswith('[') and s0.endswith(']'):
            return json.loads(s0), False
    except Exception:
        pass

    first = s0.find('['); last = s0.rfind(']')
    core = s0 if (first == -1 or last == -1 or first >= last) else s0[first:last+1]
    core = _strip_code_fences(core)

    try:
        return json.loads(core), False
    except Exception:
        return [], True  # 如果解析失败，返回空数组

# ========= 工具函数 =========
def make_windows(lines: List[str], window: int, stride: int) -> List[Tuple[int, int, str]]:
    # 确保参数是整数
    window = int(window)
    stride = int(stride)
    
    n = len(lines); out = []; i = 0
    while i < n:
        s = i; e = min(i + window - 1, n - 1)
        out.append((s+1, e+1, "\n".join(lines[s:e+1])))
        if e == n - 1: break
        i += stride
    return out

def split_document_by_titles(lines: List[str], toc_items: List[TOCItem], source_file: str) -> List[Section]:
    """
    根据标题结构拆分文档内容（使用清洗后的文本）
    从第一个标题在全文中的最后一个匹配位置开始作为全文起始点，避免目录干扰
    """
    if not toc_items:
        return []
    
    sections = []
    full_text = "\n".join(lines)
    
    # 找到第一个标题在全文中的最后一个匹配位置，作为正文起始点
    first_title = toc_items[0].title
    first_title_pos = full_text.find(first_title)
    if first_title_pos == -1:
        # 如果找不到第一个标题，返回空结果
        return []
    
    # 从第一个标题的最后一个匹配位置开始
    # 如果第一个标题在文档中出现多次，取最后一个位置
    last_occurrence_pos = first_title_pos
    current_pos = first_title_pos
    while True:
        next_pos = full_text.find(first_title, current_pos + 1)
        if next_pos == -1:
            break
        last_occurrence_pos = next_pos
        current_pos = next_pos
    
    # 从第一个标题的最后一个匹配位置开始截取正文
    body_text_start = last_occurrence_pos
    body_text = full_text[body_text_start:]
    
    # 为每个标题创建章节
    for i, toc_item in enumerate(toc_items):       
        # 在正文部分中查找标题（使用相对位置）
        title_pos = body_text.find(toc_item.title)
        if title_pos == -1:
            # 如果找不到，跳过这个标题
            continue
        
        # 找到下一个标题位置
        next_title_pos = len(body_text)
        if i + 1 < len(toc_items):
            # 在当前标题之后查找下一个标题
            for j in range(i + 1, len(toc_items)):
                next_candidate_pos = body_text.find(toc_items[j].title, title_pos + 1)
                
                if next_candidate_pos != -1:
                    next_title_pos = next_candidate_pos
                    break
        
        # 提取内容
        content_start = title_pos + len(toc_item.title)
        # 跳过标题后的标点符号（如句号、顿号等）
        while content_start < next_title_pos and body_text[content_start] in '。、，；：':
            content_start += 1
        content = body_text[content_start:next_title_pos].strip()
        
        # 创建章节对象
        section = Section(
            section_title=toc_item.title,
            content=content,
            source_file=source_file
        )
        sections.append(section)
    
    return sections

# ========= 目录提取 =========
def extract_toc_from_chunk(s_line: int, e_line: int, chunk_text: str,
                          max_retries: int, timeout: int,
                          debug_dir: Optional[Path], base_norm: str) -> Tuple[List[TOCItem], Dict[str, Any]]:
    """
    从文本片段中提取目录结构
    """    
    lines_with_numbers = []
    for i, line in enumerate(chunk_text.split('\n')):
        # 使用绝对行号，基于原始文档的行号
        absolute_line_num = s_line + i
        lines_with_numbers.append(f"{absolute_line_num:06d}│{line}")
    
    numbered_text = '\n'.join(lines_with_numbers)
    
    user_prompt = f"""请仔细阅读下方给出的文档片段（第 {s_line} 行到第 {e_line} 行），并从中提取出清晰的目录结构。

--- 文档片段开始 ---
{numbered_text}
--- 文档片段结束 ---
"""

    meta = {
        "prompt_len": len(user_prompt), "response_len": 0,
        "parse_error": "", "sanitized_used": False,
        "has_attachment": False
    }

    # 调试目录
    wdir = None
    if debug_dir:
        wdir = debug_dir / base_norm / f"toc_{s_line}-{e_line}"
        wdir.mkdir(parents=True, exist_ok=True)
        try:
            with (wdir / "system_prompt.txt").open("w", encoding="utf-8") as f:
                f.write(TOC_EXTRACT_PROMPT)
            with (wdir / "user_prompt.txt").open("w", encoding="utf-8") as f:
                f.write(user_prompt)
        except Exception as e:
            print(f"⚠️ 写入调试文件失败：{e}")

    # 重试机制
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            raw = call_ark(TOC_EXTRACT_PROMPT, user_prompt, api_key=API_KEY, model_id=MODEL_ID, timeout=timeout)
            if debug_dir:
                try:
                    with (wdir / "response.txt").open("w", encoding="utf-8") as f:
                        f.write(raw)
                except Exception as e:
                    print(f"⚠️ 写入响应调试文件失败：{e}")
            meta["response_len"] = len(raw)
            
            # 检查是否检测到附件
            try:
                obj = json.loads(raw.strip())
                if isinstance(obj, dict) and obj.get("has_attachment", False):
                    meta["has_attachment"] = True
                    return [], meta
            except Exception:
                pass
            
            obj, sanitized_used = force_json_load_with_sanitize(raw)
            meta["sanitized_used"] = sanitized_used
            
            # 转换为TOCItem对象
            toc_items = []
            for item in obj:
                if isinstance(item, dict):
                    toc_items.append(TOCItem(
                        title=item.get("title", "")
                    ))
            
            if debug_dir and sanitized_used:
                try:
                    with (wdir / "sanitized.json").open("w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️ 写入sanitized调试文件失败：{e}")
            
            return toc_items, meta
            
        except Exception as e:
            last_err = str(e)
            meta["parse_error"] = last_err
            if debug_dir:
                try:
                    with (wdir / "parse_error.txt").open("w", encoding="utf-8") as f:
                        f.write(last_err)
                except Exception as write_err:
                    print(f"⚠️ 写入错误调试文件失败：{write_err}")
            if attempt == max_retries:
                break
            time.sleep((1.8 ** attempt) + random.uniform(0.1, 0.5))
            attempt += 1

    return [], meta

def verify_toc_structure(s_line: int, e_line: int, chunk_text: str, 
                        existing_toc_items: List[TOCItem],
                        max_retries: int, timeout: int,
                        debug_dir: Optional[Path], base_norm: str) -> Tuple[List[TOCItem], Dict[str, Any]]:
    """
    验证已提取的标题结构，输出最终的完整标题列表
    """    
    # 为每一行添加绝对行号
    lines_with_numbers = []
    for i, line in enumerate(chunk_text.split('\n')):
        # 使用绝对行号，基于原始文档的行号
        absolute_line_num = s_line + i
        lines_with_numbers.append(f"{absolute_line_num:06d}│{line}")
    
    numbered_text = '\n'.join(lines_with_numbers)
    
    # 准备现有标题结构
    existing_titles_text = ""
    for item in existing_toc_items:
        existing_titles_text += f"{item.title}\n"
    
    user_prompt = f"""请输出最终的完整标题结构。

--- 已提取的标题结构 ---
{existing_titles_text.strip()}

--- 文档片段（第 {s_line} 行到第 {e_line} 行） ---
{numbered_text}
--- 文档片段结束 ---

请仔细检查并输出最终的完整标题列表。
"""

    meta = {
        "prompt_len": len(user_prompt), "response_len": 0,
        "parse_error": "", "sanitized_used": False,
    }

    # 调试目录
    wdir = None
    if debug_dir:
        wdir = debug_dir / base_norm / f"verify_{s_line}-{e_line}"
        wdir.mkdir(parents=True, exist_ok=True)
        try:
            with (wdir / "system_prompt.txt").open("w", encoding="utf-8") as f:
                f.write(TOC_VERIFY_PROMPT)
            with (wdir / "user_prompt.txt").open("w", encoding="utf-8") as f:
                f.write(user_prompt)
        except Exception as e:
            print(f"⚠️ 写入调试文件失败：{e}")

    # 重试机制
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            raw = call_ark(TOC_VERIFY_PROMPT, user_prompt, api_key=API_KEY, model_id=MODEL_ID, timeout=timeout)
            if debug_dir:
                try:
                    with (wdir / "response.txt").open("w", encoding="utf-8") as f:
                        f.write(raw)
                except Exception as e:
                    print(f"⚠️ 写入响应调试文件失败：{e}")
            meta["response_len"] = len(raw)
            
            # 解析JSON响应
            try:
                obj = json.loads(raw.strip())
            except Exception:
                # 尝试清理和重新解析
                obj, sanitized_used = force_json_load_with_sanitize(raw)
                meta["sanitized_used"] = sanitized_used
            else:
                meta["sanitized_used"] = False
            
            # 转换为TOCItem对象
            final_toc_items = []
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        final_toc_items.append(TOCItem(
                            title=item.get("title", "")
                        ))
            
            if debug_dir and meta["sanitized_used"]:
                try:
                    with (wdir / "sanitized.json").open("w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️ 写入sanitized调试文件失败：{e}")
            
            return final_toc_items, meta
            
        except Exception as e:
            last_err = str(e)
            meta["parse_error"] = last_err
            if debug_dir:
                try:
                    with (wdir / "parse_error.txt").open("w", encoding="utf-8") as f:
                        f.write(last_err)
                except Exception as write_err:
                    print(f"⚠️ 写入错误调试文件失败：{write_err}")
            if attempt == max_retries:
                break
            time.sleep((1.8 ** attempt) + random.uniform(0.1, 0.5))
            attempt += 1

    # 如果验证失败，返回空结果
    return [], meta

# ========= 单文件处理 =========
def process_single_file(md_path: Path,
                       toc_window: int, toc_stride: int,
                       max_retries: int, timeout: int,
                       debug_dir: Optional[Path],
                       out_dir: Path, input_dir: Path) -> Tuple[str, List[TOCItem], List[Section], ExtractStats]:
    """
    处理单个文件：复制文件并提取标题结构和拆分文档
    """
    base_norm = md_path.stem
    
    # 获取*_output文件夹名称
    output_folder_name = input_dir.name
    
    # 创建输出目录（只按*_output文件夹名称分类）
    out_dir_file = out_dir / output_folder_name / base_norm
    out_dir_file.mkdir(parents=True, exist_ok=True)
    
    # 复制原始markdown文件到输出目录
    dest_file = out_dir_file / md_path.name
    copy_success = False
    try:
        shutil.copy2(md_path, dest_file)
        print(f"📁 已复制原始文件：{md_path.name} -> {dest_file}")
        copy_success = True
    except Exception as e:
        print(f"❌ 复制原始文件失败 {md_path}: {e}")
    
    # 使用原始文件进行处理
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    
    # 清洗文本：去掉每行开头的#号和中间的空格
    cleaned_lines = []
    for line in lines:
        # 去掉开头的#号
        cleaned_line = re.sub(r'^#+\s*', '', line)
        # 去掉中间的空格
        cleaned_line = re.sub(r'\s+', '', cleaned_line)
        # 跳过空行
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    lines = cleaned_lines
    
    stats = ExtractStats(files_processed=1, files_copied=1 if copy_success else 0)
    
    # 第一步：提取文档结构
    print(f"🔍 提取文档结构：{md_path.name}")
    toc_windows = make_windows(lines, toc_window, toc_stride)
    
    all_toc_items = []
    attachment_detected = False
    # 提取结构进度跟踪
    total_windows = len(toc_windows)
    for i, (s_line, e_line, chunk) in enumerate(toc_windows):
        if i % 5 == 0 or i == total_windows - 1:  # 每5个或最后一个显示进度
            print(f"🔍 提取结构进度: {i+1}/{total_windows} ({(i+1)/total_windows*100:.1f}%)")
        toc_items, meta = extract_toc_from_chunk(
            s_line, e_line, chunk, max_retries, timeout, debug_dir, base_norm
        )
        
        # 检查是否检测到附件
        if meta.get("has_attachment", False):
            attachment_detected = True
            print(f"📎 在第 {s_line}-{e_line} 行检测到附件，停止后续标题提取")
            stats.files_with_attachments = 1
            break
        
        all_toc_items.extend(toc_items)
    
    # 保留所有标题项（不去重，允许相同标题在不同位置出现）
    unique_toc_items = all_toc_items  # 直接使用所有标题，不去重（否则同样标题不同级出现可能会被当作重复）
    
    stats.title_items = len(unique_toc_items)
    
    if unique_toc_items:
        stats.files_with_titles = 1
        print(f"✅ 找到 {len(unique_toc_items)} 个标题")
        
        # 第二步：验证标题结构（新增步骤）
        print(f"🔍 验证标题结构：{md_path.name}")
        # 验证窗口增大一倍
        verify_window = int(toc_window * 1.5)
        verify_stride = int(toc_stride * 1.5)
        verify_windows = make_windows(lines, verify_window, verify_stride)
        
        all_verified_titles = []
        
        # 验证结构进度跟踪
        total_verify_windows = len(verify_windows)
        for i, (s_line, e_line, chunk) in enumerate(verify_windows):
            if i % 5 == 0 or i == total_verify_windows - 1:  # 每5个或最后一个显示进度
                print(f"🔍 验证结构进度: {i+1}/{total_verify_windows} ({(i+1)/total_verify_windows*100:.1f}%)")
            # 使用所有标题进行验证
            window_toc_items = unique_toc_items
            
            # 验证所有窗口，不管是否有标题
            verified_titles, meta = verify_toc_structure(
                s_line, e_line, chunk, window_toc_items,
                max_retries, timeout, debug_dir, base_norm
            )
            
            all_verified_titles.extend(verified_titles)
        
        # 保留所有验证后的标题（不去重）
        final_toc_items = all_verified_titles  # 直接使用所有验证后的标题，不去重
        
        print(f"✅ 验证后最终标题数：{len(final_toc_items)} 个")
        
        # 第三步：根据标题结构拆分文档
        print(f"📄 拆分文档内容：{md_path.name}")
        sections = split_document_by_titles(lines, final_toc_items, md_path.name)
        
        # 去重章节（基于标题，优先保留有内容的）
        title_to_sections = {}
        for section in sections:
            if section.section_title not in title_to_sections:
                title_to_sections[section.section_title] = []
            title_to_sections[section.section_title].append(section)
        
        unique_sections = []
        for title, section_list in title_to_sections.items():
            if len(section_list) == 1:
                # 只有一个章节，直接保留
                unique_sections.append(section_list[0])
            else:
                # 多个章节，优先保留有内容的
                sections_with_content = [s for s in section_list if s.content.strip()]
                if sections_with_content:
                    # 有内容的章节，保留第一个
                    best_section = sections_with_content[0]
                else:
                    # 都没有内容，保留第一个
                    best_section = section_list[0]
                unique_sections.append(best_section)
        
        stats.sections_extracted = len(unique_sections)
        print(f"✅ 拆分为 {len(unique_sections)} 个章节（去重后）")
        
        return base_norm, final_toc_items, unique_sections, stats
    else:
        print(f"❌ 未找到文档结构：{md_path.name}")
        return base_norm, [], [], stats

# ========= 保存结果 =========
def save_results(base_norm: str, toc_items: List[TOCItem], sections: List,
                out_dir: Path, relative_path: Path = None, output_folder_name: str = None) -> Tuple[Path, Path]:
    """
    保存提取的标题结构
    """
    # 如果提供了输出文件夹名称，只按*_output文件夹名称分类
    if output_folder_name:
        out_dir_file = out_dir / output_folder_name / base_norm
    else:
        out_dir_file = out_dir / base_norm
    
    # 保存文档结构
    toc_path = out_dir_file / "structure.json"
    toc_data = []
    for item in toc_items:
            toc_data.append({
                "title": item.title
            })
    try:
        with toc_path.open("w", encoding="utf-8") as f:
            json.dump(toc_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存文档结构失败：{e}")
        toc_path = None
    
    # 保存章节内容
    sections_path = out_dir_file / "sections.jsonl"
    try:
        with sections_path.open("w", encoding="utf-8") as f:
            for section in sections:
                f.write(json.dumps(asdict(section), ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ 保存章节内容失败：{e}")
        sections_path = None
    
    return toc_path, sections_path

# ========= 主程序 =========
def main():
    ap = argparse.ArgumentParser(description="从文档中提取标题结构")
    ap.add_argument("input_dir", type=str, help="基础目录（将处理其中所有以*_output结尾的文件夹）")
    ap.add_argument("--out-dir", type=str, default="out_tree", help="输出目录")
    ap.add_argument("--debug-log-dir", type=str, default="", help="调试日志目录")
    ap.add_argument("--toc-window", type=int, default=DEFAULT_WINDOW_LINES, help="结构提取窗口大小（行）")
    ap.add_argument("--toc-stride", type=int, default=DEFAULT_STRIDE_LINES, help="结构提取步长（行）")
    ap.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    ap.add_argument("--timeout", type=int, default=DEFAULT_ARK_TIMEOUT, help="请求超时（秒）")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="并发线程数")
    args = ap.parse_args()

    # 处理输入目录：查找指定目录下所有以*_output结尾的文件夹
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"❌ 基础目录不存在：{base_dir}")
        return
    
    # 查找所有以*_output结尾的文件夹
    import glob
    output_pattern = str(base_dir / "*_output")
    input_dirs = glob.glob(output_pattern)
    
    # 过滤出目录（排除文件）
    input_dirs = [d for d in input_dirs if Path(d).is_dir()]
    
    if not input_dirs:
        print(f"❌ 在 {base_dir} 中未找到以*_output结尾的文件夹")
        return
    
    print(f"📂 找到匹配的文件夹：{input_dirs}")
    
    out_dir = Path(args.out_dir)
    debug_dir = Path(args.debug_log_dir) if args.debug_log_dir else None

    # 查找所有.md文件
    md_files = []
    for input_dir_path in input_dirs:
        input_dir = Path(input_dir_path)
        dir_md_files = list(input_dir.rglob("*.md"))
        md_files.extend(dir_md_files)
        print(f"📄 在 {input_dir} 中找到 {len(dir_md_files)} 个.md文件")
    
    if not md_files:
        print(f"❌ 未找到.md文件")
        return
    print(f"📄 总共找到 {len(md_files)} 个.md文件")

    out_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    all_toc_items = []
    all_sections = []
    total_stats = ExtractStats()

    # 并发处理
    print(f"🚀 使用并发处理，线程数：{args.max_workers}")
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for md_path in md_files:
            # 确定文件属于哪个输入目录
            file_input_dir = None
            for input_dir_path in input_dirs:
                input_dir = Path(input_dir_path)
                try:
                    md_path.relative_to(input_dir)
                    file_input_dir = input_dir
                    break
                except ValueError:
                    continue
            
            if file_input_dir is None:
                print(f"⚠️ 无法确定文件 {md_path} 的输入目录")
                continue
            
            future = executor.submit(
                process_single_file,
                md_path=md_path,
                toc_window=args.toc_window,
                toc_stride=args.toc_stride,
                max_retries=args.max_retries,
                timeout=args.timeout,
                debug_dir=debug_dir,
                out_dir=out_dir,
                input_dir=file_input_dir
            )
            futures[future] = (md_path, file_input_dir)

        # 使用更兼容的方式处理进度条
        completed = 0
        for fut in cf.as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(futures):  # 每10个或最后一个显示进度
                print(f"📊 处理进度: {completed}/{len(futures)} ({completed/len(futures)*100:.1f}%)")
            md_path, file_input_dir = futures[fut]
            try:
                base_norm, toc_items, sections, stats = fut.result()
                
                # 更新统计
                total_stats.files_processed += stats.files_processed
                total_stats.files_copied += stats.files_copied
                total_stats.files_with_titles += stats.files_with_titles
                total_stats.title_items += stats.title_items
                total_stats.sections_extracted += stats.sections_extracted
                total_stats.missing_titles_found += stats.missing_titles_found
                total_stats.extra_titles_found += stats.extra_titles_found
                total_stats.correct_titles_found += stats.correct_titles_found
                total_stats.files_with_attachments += stats.files_with_attachments
                
                # 保存结果
                if toc_items or sections:
                    # 计算相对路径以保持文件夹结构
                    relative_path = md_path.relative_to(file_input_dir)
                    output_folder_name = file_input_dir.name
                    toc_path, sections_path = save_results(
                        base_norm, toc_items, sections, out_dir, relative_path, output_folder_name
                    )
                    print(f"✅ {md_path.name} → 目录项={len(toc_items)}, 章节={len(sections)}")
                    print(f"📝 目录：{toc_path}")
                    print(f"📄 章节：{sections_path}")
                
                all_toc_items.extend(toc_items)
                all_sections.extend(sections)
                
            except Exception as e:
                print(f"❌ 处理失败：{md_path} -> {e}")

    # 输出统计
    print("\n=== 处理完成 ===")
    print(f"📊 统计信息：")
    print(f"   - 处理文件数：{total_stats.files_processed}")
    print(f"   - 成功复制文件数：{total_stats.files_copied}")
    print(f"   - 有标题文件数：{total_stats.files_with_titles}")
    print(f"   - 检测到附件文件数：{total_stats.files_with_attachments}")
    print(f"✅ 所有文件处理完成，结果保存在各文档文件夹中")

if __name__ == "__main__":
    main()

# 运行示例
'''
# 使用 nohup 在后台运行，输出重定向到日志文件
nohup python 2_extract_tree.py \
  /mnt/data/projects/tmp \
  --out-dir /mnt/data/wx/xiaofang/out_xiaofang_title_$(date +%m%d%H%M) \
  --debug-log-dir /mnt/data/wx/xiaofang/debug_tree_logs_$(date +%m%d%H%M) \
  --toc-window 80 \
  --toc-stride 60 \
  --max-retries 3 \
  --timeout 120 \
  --max-workers 100 \
  > extract_tree_$(date +%m%d%H%M).log 2>&1 &

# 查看运行状态
tail -f extract_tree_$(date +%m%d%H%M).log

# 查看进程
ps aux | grep extract_tree
'''
