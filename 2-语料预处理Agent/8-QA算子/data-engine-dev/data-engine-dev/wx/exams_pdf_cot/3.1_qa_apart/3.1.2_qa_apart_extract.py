#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import concurrent.futures as cf

# ========= Ark 配置 =========
API_KEY = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

try:
    from volcenginesdkarkruntime import Ark
except Exception:
    raise SystemExit("❌ 请先安装 Ark SDK: pip install 'volcengine-python-sdk[ark]'")

# ========= 参数常量 =========
DEFAULT_WINDOW_LINES = 80
DEFAULT_STRIDE_LINES = 40
MAX_CHARS_PER_PROMPT = 9000
DEFAULT_ARK_TIMEOUT = 120

# ========= System/User Prompt =========
QUESTION_SYSTEM_PROMPT = """\
你是一个精确的试题抽取助手。
我将给你一段从考试真题转成 Markdown 的文本片段。
你的任务：只抽取四种类型的完整试题：
- single（单选题）
- multiple（多选题）
- judge（判断题）
- fill（填空题）

规则：
1) 每道题必须包含：
   - "question_number"：题目编号（数字，如1、2、3等），这是必须的，用于匹配答案
   - "type"：取值只能是 ["single","multiple","judge","fill"]
   - "question"：题干的完整文字，包括所有选项（保留题干中的图片链接或公式，选项按原顺序保留在题目文本中）。
2) 只输出完整的试题：必须有完整的问题。
3) 如果题目在语义上被截断或不完整，则不要输出，如果题目只是缺少后半个括号这种非语义的小问题，则需要补全。
4) 严格输出 JSON 格式，UTF-8 编码，不要包含 Markdown、代码块围栏或多余文字。
5) 不要编造内容。答案、解析或知识点缺失时用空字符串表示。
6) 如果片段中没有符合条件的试题，返回空数组 []。
7) 必须提取题号（question_number），这是关键字段，用于后续的答案匹配。
8)题目里如果有表格，则需要解析每个<td>单元格中的内容。

输出格式：
JSON 数组，数组中每个元素是一个试题对象。
"""

QUESTION_CHUNK_PROMPT_TEMPLATE = """\
下面是全文的第 {start_line} 行到第 {end_line} 行的 Markdown 内容。

--- 开始片段 ---
{chunk_text}
--- 结束片段 ---

请记住：
- 只保留 ["single","multiple","judge","fill"] 这四种题型。
- 必须提取题号（question_number），这是关键字段，用于后续的答案匹配。
- 只返回严格 JSON 数组，不要输出任何额外文字。
"""

ANSWER_SYSTEM_PROMPT = """\
你是一个精确的答案提取助手。
我将给你一段从考试答案文件转成 Markdown 的文本片段。
你的任务：只提取答案信息。

规则：
1) 每道答案必须包含：
   - "question_number"：题目编号（数字，如1、2、3等），这是必须的
   - "answer"：简洁的答案文字；选择题用字母（如 "A" 或 "ACD"）；判断题用"对/错"或"True/False"；填空题用实际填空内容
2) 只输出完整的答案：必须有题号和答案。
3) 如果答案被截断或不完整，则不要输出。
4) 严格输出 JSON 格式，UTF-8 编码，不要包含 Markdown、代码块围栏或多余文字。
5) 不要编造内容。答案缺失时用空字符串表示。
6) 如果片段中没有符合条件的答案，返回空数组 []。
7）特别注意表格格式：如果遇到HTML表格，请解析每个<td>单元格中的内容。
    - 表格格式示例：<td>1.C(1)←</td> 应提取为 {"question_number": 1, "answer": "C"}
    - 处理表格时，忽略括号内的分数和箭头符号，只提取题号和答案字母。
8）如果遇到1-5：ABCDE，则提取为{"question_number": 1, "answer": "A"}，以此类推。
9）如果遇到某一题号有两个答案重复只保留一个。

输出格式：
JSON 数组，数组中每个元素是一个答案对象。
"""

ANSWER_CHUNK_PROMPT_TEMPLATE = """\
下面是答案文件的第 {start_line} 行到第 {end_line} 行的内容。

--- 开始片段 ---
{chunk_text}
--- 结束片段 ---

请记住：
- 只提取答案信息，不需要题目内容。
- 必须提取题号（question_number），这是关键字段。
- 必须提取答案（answer），这是关键字段。
- 只返回严格 JSON 数组，不要输出任何额外文字。

"""


@dataclass
class Question:
    """题目数据结构"""
    question_number: int
    question_type: str
    question_text: str
    answer: str = ""
    explanation: str = ""
    knowledge_points: str = ""
    source_file: str = ""
    source_window: Tuple[int, int] = (0, 0)


@dataclass
class ExamData:
    """考试数据结构"""
    exam_name: str
    total_questions: int
    questions: List[Question]
    missing_answers: List[int]  # 缺少答案的题号
    extra_answers: List[int]    # 多余答案的题号


class QAExtractor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def get_exam_directories(self) -> List[Path]:
        if not self.base_path.exists():
            print(f"路径不存在: {self.base_path}")
            return []
            
        directories = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                directories.append(item)
                
        return sorted(directories)

    def save_raw_chunks(self, raw_records: List[Dict[str, Any]], path: Path):
        """保存原始抽取记录"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def make_windows(self, lines: List[str], window: int, stride: int) -> List[Tuple[int, int, str]]:
        n = len(lines)
        out = []
        i = 0
        while i < n:
            s = i
            e = min(i + window - 1, n - 1)
            out.append((s+1, e+1, "\n".join(lines[s:e+1])))
            if e == n - 1:
                break
            i += stride
        return out

    def compress_chunk_text(self, s: str, max_chars: int = MAX_CHARS_PER_PROMPT) -> str:
        lines = s.replace("\r", "").splitlines()
        out = []
        blank = 0
        for ln in lines:
            ln = " ".join(ln.split())
            if not ln:
                blank += 1
                if blank > 1:
                    continue
            else:
                blank = 0
            out.append(ln)
        s2 = "\n".join(out).strip()
        return s2[:max_chars] if len(s2) > max_chars else s2

    def call_ark(self, prompt: str, system_prompt: str, temperature: float = 0.0, top_p: float = 0.9,
                 timeout: int = DEFAULT_ARK_TIMEOUT) -> str:
        time.sleep(random.uniform(0.05, 0.20))  # 轻微抖动
        client = Ark(api_key=API_KEY, timeout=timeout)
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)

    def write_text(self, path: Path, content: str):
        """写入文本文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
    
    def force_json_load_with_sanitize(self, s: str) -> Tuple[Any, bool]:
        s0 = s.strip()
        
        # 检查是否为空或无效
        if not s0 or s0 == "" or s0 == "[]" or s0 == "{}":
            # print(f"    输入为空或无效JSON，返回空数组")
            return [], True
        
        # 尝试直接解析
        try:
            if s0.startswith('[') and s0.endswith(']'):
                return json.loads(s0), False
        except json.JSONDecodeError as e:
            # 如果是"Extra data"错误，尝试提取所有完整的JSON数组并合并
            if "Extra data" in str(e):
                try:
                    all_arrays = []
                    current_pos = 0
                    
                    while current_pos < len(s0):
                        # 找到下一个 [ 的位置
                        start_pos = s0.find('[', current_pos)
                        if start_pos == -1:
                            break
                        
                        # 找到对应的 ] 的位置
                        bracket_count = 0
                        end_pos = -1
                        for i in range(start_pos, len(s0)):
                            if s0[i] == '[':
                                bracket_count += 1
                            elif s0[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = i
                                    break
                        
                        if end_pos != -1:
                            # 提取完整的JSON数组
                            json_array = s0[start_pos:end_pos + 1]
                            try:
                                # 尝试解析这个JSON数组
                                parsed_array = json.loads(json_array)
                                if isinstance(parsed_array, list):
                                    all_arrays.extend(parsed_array)
                            except Exception:
                                # 如果解析失败，跳过这个数组
                                pass
                            
                            current_pos = end_pos + 1
                        else:
                            # 没有找到匹配的 ]，跳过
                            current_pos = start_pos + 1
                    
                    # 如果成功提取了多个数组，返回合并结果
                    if all_arrays:
                        return all_arrays, True
                except Exception:
                    pass
        except Exception:
            pass

        # 截取首个 [ 到最后一个 ] 的核心
        first = s0.find('[')
        last = s0.rfind(']')
        core = s0 if (first == -1 or last == -1 or first >= last) else s0[first:last+1]
        
        # 处理可能存在的多个JSON数组：提取所有完整的JSON数组并合并
        if core.count('[') > 1:
            all_arrays = []
            current_pos = 0
            
            while current_pos < len(core):
                # 找到下一个 [ 的位置
                start_pos = core.find('[', current_pos)
                if start_pos == -1:
                    break
                
                # 找到对应的 ] 的位置
                bracket_count = 0
                end_pos = -1
                for i in range(start_pos, len(core)):
                    if core[i] == '[':
                        bracket_count += 1
                    elif core[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i
                            break
                
                if end_pos != -1:
                    # 提取完整的JSON数组
                    json_array = core[start_pos:end_pos + 1]
                    try:
                        # 尝试解析这个JSON数组
                        parsed_array = json.loads(json_array)
                        if isinstance(parsed_array, list):
                            all_arrays.extend(parsed_array)
                    except Exception:
                        # 如果解析失败，跳过这个数组
                        pass
                    
                    current_pos = end_pos + 1
                else:
                    # 没有找到匹配的 ]，跳过
                    current_pos = start_pos + 1
            
            # 如果成功提取了多个数组，返回合并结果
            if all_arrays:
                return all_arrays, True
        
        # 去除代码围栏
        fences = ("```json", "```JSON", "```", "~~~json", "~~~JSON", "~~~")
        for f in fences:
            if core.startswith(f):
                core = core[len(f):].strip()
            if core.endswith(f):
                core = core[:-len(f)].strip()

        # 尝试解析清洗后的JSON
        try:
            return json.loads(core), True
        except json.JSONDecodeError as e:
            error_msg = str(e)
            
            # 处理空值或无效值问题
            if "Expecting value" in error_msg:
                print(f"    ⚠️ JSON为空或无效，跳过此窗口")
                return [], True
            
            # 处理转义字符问题
            if "Invalid \\escape" in error_msg or "Invalid \\uXXXX escape" in error_msg:
                try:
                    # 尝试修复常见的转义字符问题
                    fixed_core = self._fix_json_escapes(core)
                    return json.loads(fixed_core), True
                except Exception:
                    pass
                
                # 如果修复失败，尝试更激进的方法：移除所有有问题的转义
                try:
                    aggressive_core = self._aggressive_json_fix(core)
                    return json.loads(aggressive_core), True
                except Exception:
                    pass
            
            # 处理格式问题
            if ("Expecting ',' delimiter" in error_msg or 
                "Expecting property name" in error_msg or 
                "Expecting ':' delimiter" in error_msg):
                try:
                    # 首先清理HTML内容
                    cleaned_core = self._clean_html_content(core)
                    # 尝试修复JSON格式问题
                    format_fixed_core = self._fix_json_format(cleaned_core)
                    return json.loads(format_fixed_core), True
                except Exception:
                    pass
                
                # 如果格式修复失败，尝试智能修复
                try:
                    smart_fixed_core = self._smart_json_fix(cleaned_core)
                    return json.loads(smart_fixed_core), True
                except Exception:
                    pass
                
                # 如果智能修复失败，尝试转义修复
                try:
                    escape_fixed_core = self._fix_json_escapes(smart_fixed_core)
                    return json.loads(escape_fixed_core), True
                except Exception:
                    pass
            
            # 最后的备用方案：尝试提取有效的JSON片段
            try:
                # 首先清理HTML内容
                cleaned_core = self._clean_html_content(core)
                # 尝试找到完整的JSON对象
                import re
                # 查找所有可能的JSON对象
                json_objects = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', cleaned_core)
                if json_objects:
                    # 尝试解析第一个完整的对象
                    for obj in json_objects:
                        try:
                            parsed_obj = json.loads(obj)
                            if isinstance(parsed_obj, dict) and "question_number" in parsed_obj:
                                # 找到有效的答案对象，包装成数组返回
                                return [parsed_obj], True
                        except:
                            continue
            except Exception:
                pass
            
            # 处理"Extra data"错误：尝试提取所有完整的JSON数组并合并
            if "Extra data" in error_msg:
                try:
                    all_arrays = []
                    current_pos = 0
                    
                    while current_pos < len(core):
                        # 找到下一个 [ 的位置
                        start_pos = core.find('[', current_pos)
                        if start_pos == -1:
                            break
                        
                        # 找到对应的 ] 的位置
                        bracket_count = 0
                        end_pos = -1
                        for i in range(start_pos, len(core)):
                            if core[i] == '[':
                                bracket_count += 1
                            elif core[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = i
                                    break
                        
                        if end_pos != -1:
                            # 提取完整的JSON数组
                            json_array = core[start_pos:end_pos + 1]
                            try:
                                # 尝试解析这个JSON数组
                                parsed_array = json.loads(json_array)
                                if isinstance(parsed_array, list):
                                    all_arrays.extend(parsed_array)
                            except Exception:
                                # 如果解析失败，跳过这个数组
                                pass
                            
                            current_pos = end_pos + 1
                        else:
                            # 没有找到匹配的 ]，跳过
                            current_pos = start_pos + 1
                    
                    # 如果成功提取了多个数组，返回合并结果
                    if all_arrays:
                        return all_arrays, True
                except Exception:
                    pass
            
            # 如果还是失败，返回空列表
            print(f"    JSON解析失败: {e}")
            if hasattr(e, 'pos'):
                print(f"    错误位置: 第{e.lineno}行，第{e.colno}列，字符{e.pos}")
                # 显示错误位置附近的文本
                start = max(0, e.pos - 50)
                end = min(len(s0), e.pos + 50)
                print(f"    错误附近文本: ...{s0[start:end]}...")
            else:
                print(f"    原始文本: {s0[:200]}...")
            return [], True
        except Exception as e:
            # 其他异常
            print(f"    JSON解析失败: {e}")
            print(f"    原始文本: {s0[:200]}...")
            return [], True
    
    def _fix_json_escapes(self, json_str: str) -> str:
        """修复JSON中的非法反斜杠：对非合法转义前的反斜杠进行二次转义"""
        _HEX = set("0123456789abcdefABCDEF")
        
        out = []
        i = 0
        n = len(json_str)
        while i < n:
            ch = json_str[i]
            if ch != '\\':
                out.append(ch)
                i += 1
                continue
            # ch is backslash
            if i + 1 >= n:
                out.append('\\\\')
                i += 1
                continue
            nxt = json_str[i+1]
            if nxt in '"\\/bfnrt':
                out.append('\\' + nxt)
                i += 2
                continue
            if nxt == 'u':
                # \uXXXX - 检查是否有完整的4位十六进制
                if i + 5 < n:
                    hex_part = json_str[i+2:i+6]
                    if len(hex_part) == 4 and all(c in _HEX for c in hex_part):
                        out.append(json_str[i:i+6])
                        i += 6
                        continue
                    else:
                        # 不完整的Unicode转义，转义反斜杠
                        out.append('\\\\u')
                        i += 2
                        continue
                else:
                    # 字符串末尾的不完整Unicode转义
                    out.append('\\\\u')
                    i += 2
                    continue
            # 非法转义 -> 双反斜杠
            out.append('\\\\' + nxt)
            i += 2
        return ''.join(out)
    
    def _aggressive_json_fix(self, json_str: str) -> str:
        """激进的JSON修复方法：移除所有有问题的转义字符"""
        import re
        
        # 移除所有不完整的Unicode转义
        json_str = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', '', json_str)
        
        # 移除所有未转义的反斜杠（保留转义字符）
        # 先保护转义字符
        protected = {}
        protected_count = 0
        
        def protect_escaped(match):
            nonlocal protected_count
            protected_count += 1
            key = f"__PROTECTED_{protected_count}__"
            protected[key] = match.group(0)
            return key
        
        # 保护转义字符
        json_str = re.sub(r'\\["\\/bfnrt]', protect_escaped, json_str)
        
        # 保护完整的Unicode转义
        json_str = re.sub(r'\\u[0-9a-fA-F]{4}', protect_escaped, json_str)
        
        # 移除所有剩余的反斜杠
        json_str = json_str.replace('\\', '')
        
        # 恢复保护的字符
        for key, value in protected.items():
            json_str = json_str.replace(key, value)
        
        return json_str
    
    def _fix_json_format(self, json_str: str) -> str:
        """修复JSON格式问题"""
        import re
        
        # 修复常见的格式问题
        
        # 1. 修复缺少冒号的问题：在键名后添加冒号
        json_str = re.sub(r'"([^"]+)"\s+([^"\s,{}[\]]+)', r'"\1":\2', json_str)
        
        # 2. 修复缺少逗号的问题：在 } 和 { 之间添加逗号
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # 3. 修复缺少逗号的问题：在 } 和 [ 之间添加逗号
        json_str = re.sub(r'}\s*\[', '},[', json_str)
        
        # 4. 修复缺少逗号的问题：在 ] 和 { 之间添加逗号
        json_str = re.sub(r'\]\s*{', '},{', json_str)
        
        # 5. 修复缺少逗号的问题：在 ] 和 [ 之间添加逗号
        json_str = re.sub(r'\]\s*\[', '],[', json_str)
        
        # 6. 修复多余的逗号：在 } 或 ] 前移除多余的逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # 7. 修复字符串末尾的逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # 8. 修复缺少引号的问题：确保键名有引号
        json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # 9. 修复字符串值缺少引号的问题（简单情况）
        # 注意：这个比较危险，只在特定情况下使用
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', json_str)
        
        # 10. 修复常见的键值对格式问题
        # 处理 "key" value 格式（缺少冒号）
        json_str = re.sub(r'"([^"]+)"\s+([^"\s,{}[\]]+)(?=\s*[,}])', r'"\1":\2', json_str)
        
        # 11. 修复数字值缺少引号的问题（如果应该是字符串）
        # 处理 "question_number" 1 格式
        json_str = re.sub(r'"question_number"\s+(\d+)(?=\s*[,}])', r'"question_number":"\1"', json_str)
        
        return json_str
    
    def _smart_json_fix(self, json_str: str) -> str:
        """智能JSON修复：尝试多种修复策略"""
        import re
        
        # 策略0：处理HTML内容
        # 移除或转义HTML标签
        json_str = re.sub(r'<[^>]+>', '', json_str)  # 移除HTML标签
        json_str = re.sub(r'&[a-zA-Z]+;', '', json_str)  # 移除HTML实体
        
        # 策略1：尝试修复常见的键值对格式
        patterns_to_fix = [
            # "key" value -> "key": value
            (r'"([^"]+)"\s+([^"\s,{}[\]]+)(?=\s*[,}])', r'"\1":\2'),
            # "key" "value" -> "key": "value"
            (r'"([^"]+)"\s+"([^"]+)"', r'"\1":"\2"'),
            # key: value -> "key": value
            (r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^"\s,{}[\]]+)', r'"\1":\2'),
            # key: "value" -> "key": "value"
            (r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"([^"]*)"', r'"\1":"\2"'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            json_str = re.sub(pattern, replacement, json_str)
        
        # 策略2：修复数组格式
        # 确保数组元素之间有逗号
        json_str = re.sub(r'}\s*{', '},{', json_str)
        json_str = re.sub(r'\]\s*\[', '],[', json_str)
        
        # 策略3：修复多余的逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # 策略4：修复字符串值
        # 将数字值转换为字符串（如果键名是question_number）
        json_str = re.sub(r'"question_number"\s*:\s*(\d+)', r'"question_number":"\1"', json_str)
        
        # 策略5：清理多余的空白字符
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = json_str.strip()
        
        return json_str
    
    def _clean_html_content(self, json_str: str) -> str:
        """清理JSON中的HTML内容"""
        import re
        
        # 移除HTML标签
        json_str = re.sub(r'<[^>]+>', '', json_str)
        
        # 移除HTML实体
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' ',
        }
        
        for entity, replacement in html_entities.items():
            json_str = json_str.replace(entity, replacement)
        
        # 移除其他HTML实体
        json_str = re.sub(r'&[a-zA-Z]+;', '', json_str)
        json_str = re.sub(r'&#\d+;', '', json_str)
        
        # 清理多余的空白字符
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = json_str.strip()
        
        return json_str
    
    def send_chunk_adaptive(self, s_line: int, e_line: int, chunk_text: str,
                           max_retries: int, retry_backoff: float, timeout: int,
                           debug_dir: Optional[Path] = None, base_norm: str = "",
                           prompt_template: str = QUESTION_CHUNK_PROMPT_TEMPLATE,
                           system_prompt: str = QUESTION_SYSTEM_PROMPT,
                           task_type: str = "questions") -> Tuple[str, list, dict]:
        compressed = self.compress_chunk_text(chunk_text, MAX_CHARS_PER_PROMPT)
        prompt = prompt_template.format(
            start_line=s_line, end_line=e_line, chunk_text=compressed
        )
        meta = {
            "prompt_len": len(prompt), "response_len": 0,
            "parse_error": "", "sanitized_used": False,
            "compressed_chunk": compressed
        }

        # 调试目录 - 根据任务类型创建子目录
        wdir = None
        if debug_dir:
            wdir = debug_dir / base_norm / task_type / f"{s_line}-{e_line}"
            self.write_text(wdir / "prompt.txt", prompt)  # 送给模型的完整用户消息（含片段）

        # 直接重试
        attempt = 0
        last_err = None
        while attempt <= max_retries:
            try:
                raw = self.call_ark(prompt, system_prompt=system_prompt, timeout=timeout)
                if debug_dir:
                    self.write_text(wdir / "response.txt", raw)
                meta["response_len"] = len(raw)
                obj, sanitized_used = self.force_json_load_with_sanitize(raw)
                meta["sanitized_used"] = sanitized_used
                if debug_dir and sanitized_used:
                    # 保存清洗后的 JSON 文本
                    self.write_text(wdir / "sanitized.json", json.dumps(obj, ensure_ascii=False, indent=2))
                return raw, obj, meta
            except Exception as e:
                last_err = str(e)
                meta["parse_error"] = last_err
                if debug_dir:
                    self.write_text(wdir / "parse_error.txt", last_err)
                if attempt == max_retries:
                    break
                time.sleep((retry_backoff ** attempt) + random.uniform(0.1, 0.5))
                attempt += 1

        return f"<<FAILED {s_line}-{e_line}>> {last_err}", [], meta

    def extract_all_exams(self, window: int = DEFAULT_WINDOW_LINES, stride: int = DEFAULT_STRIDE_LINES,
                         max_retries: int = 3, timeout: int = DEFAULT_ARK_TIMEOUT,
                         max_workers: int = 1,
                         debug_dir: Optional[Path] = None,
                         raw_dir: Optional[Path] = None,
                         output_dir: Optional[Path] = None) -> List[ExamData]:
        exam_dirs = self.get_exam_directories()
        
        if not exam_dirs:
            print("未找到考试目录")
            return []
        
        all_exams_data = []
        
        # 添加总的进度条
        total_pbar = tqdm(total=len(exam_dirs), desc="总体进度", ncols=100, position=0)
        

        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_exam_directory,
                    exam_dir=exam_dir,
                    window=window,
                    stride=stride,
                    max_retries=max_retries,
                    timeout=timeout,
                    debug_dir=debug_dir,
                    raw_dir=raw_dir
                ): exam_dir
                for exam_dir in exam_dirs
            }
            
            for fut in cf.as_completed(futures):
                exam_dir = futures[fut]
                try:
                    exam_data = fut.result()
                    all_exams_data.append(exam_data)
                    
                    # 自动保存每个考试的结果
                    if output_dir:
                        self.save_single_exam(exam_data, output_dir)
                        # 同时保存累积结果
                        self.save_to_json(all_exams_data, str(output_dir / "qa_extracted_incremental.json"))
                        print(f"    💾 自动保存完成: {len(all_exams_data)} 个考试")
                        
                except Exception as e:
                    print(f"❌ 处理 {exam_dir.name} 时出错: {e}")
                finally:
                    total_pbar.update(1)
        
        total_pbar.close()
        return all_exams_data
    
    def process_exam_directory(self, exam_dir: Path, window: int = DEFAULT_WINDOW_LINES,
                              stride: int = DEFAULT_STRIDE_LINES, max_retries: int = 3,
                              timeout: int = DEFAULT_ARK_TIMEOUT,
                              debug_dir: Optional[Path] = None,
                              raw_dir: Optional[Path] = None) -> ExamData:
        exam_name = exam_dir.name
        questions_file = exam_dir / 'questions.md'
        answers_file = exam_dir / 'answers.md'
        
        
        # 提取题目
        questions, questions_raw_records = self.extract_questions_from_file(
            questions_file, window=window, stride=stride,
            max_retries=max_retries, timeout=timeout, debug_dir=debug_dir,
            exam_name=exam_name
        )
        
        # 保存题目原始记录
        if raw_dir:
            questions_raw_path = raw_dir / exam_name / "questions_raw_chunks.jsonl"
            self.save_raw_chunks(questions_raw_records, questions_raw_path)
            
        answers, answers_raw_records = self.extract_answers_from_file(
            answers_file, window=window, stride=stride,
            max_retries=max_retries, timeout=timeout, debug_dir=debug_dir,
            exam_name=exam_name
        )
        
        # 调试信息
        if debug_dir:
            answers_debug_dir = debug_dir / exam_name / "answers"
        
        # 保存答案原始记录
        if raw_dir:
            answers_raw_path = raw_dir / exam_name / "answers_raw_chunks.jsonl"
            self.save_raw_chunks(answers_raw_records, answers_raw_path)
        
        # 验证匹配情况
        questions, missing_answers, extra_answers = self.validate_qa_matching(
            questions, answers, exam_name=exam_name, raw_dir=raw_dir
        )
        
        return ExamData(
            exam_name=exam_name,
            total_questions=len(questions),
            questions=questions,
            missing_answers=missing_answers,
            extra_answers=extra_answers
        )
    
    def extract_questions_from_file(self, file_path: Path, window: int = DEFAULT_WINDOW_LINES,
                                   stride: int = DEFAULT_STRIDE_LINES, max_retries: int = 3,
                                   timeout: int = DEFAULT_ARK_TIMEOUT,
                                   debug_dir: Optional[Path] = None,
                                   exam_name: str = "") -> Tuple[List[Question], List[Dict[str, Any]]]:
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return [], []
        
        text = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()
        windows = self.make_windows(lines, window=window, stride=stride)
        
        all_questions = []
        seen_questions = set()
        raw_records = []
        
        pbar = tqdm(total=len(windows), desc=f"提取题目 {exam_name} {file_path.name}", ncols=100, position=1)
        
        for s_line, e_line, chunk in windows:
            raw_out, data, meta = self.send_chunk_adaptive(
                s_line=s_line, e_line=e_line, chunk_text=chunk,
                max_retries=max_retries, retry_backoff=1.8, timeout=timeout,
                debug_dir=debug_dir, base_norm=exam_name,
                task_type="questions"
            )
            
            # 保存原始记录
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
            
            for obj in data:
                question_text = str(obj.get("question") or "").strip()
                qtype = str(obj.get("type") or "").strip()
                question_number = int(obj.get("question_number") or 0)
                
                # 去重（使用题号+题目内容）
                question_hash = hash(f"{question_number}_{question_text}")
                if question_hash in seen_questions:
                    continue
                seen_questions.add(question_hash)
                
                question = Question(
                    question_number=question_number,
                    question_type=qtype,
                    question_text=re.sub(r'\s+', ' ', question_text.strip()),
                    answer="",
                    explanation="",
                    knowledge_points="",
                    source_file=file_path.stem,
                    source_window=(s_line, e_line)
                )
                all_questions.append(question)
            
            pbar.update(1)
        pbar.close()
        
        # 按题号排序
        all_questions.sort(key=lambda x: x.question_number)
        
        return all_questions, raw_records
    
    def extract_answers_from_file(self, answers_file: Path, window: int = DEFAULT_WINDOW_LINES,
                                 stride: int = DEFAULT_STRIDE_LINES, max_retries: int = 3,
                                 timeout: int = DEFAULT_ARK_TIMEOUT,
                                 debug_dir: Optional[Path] = None,
                                 exam_name: str = "") -> Tuple[List[Question], List[Dict[str, Any]]]:
        if not answers_file.exists():
            print(f"答案文件不存在: {answers_file}")
            return [], []
        
        text = answers_file.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()
        windows = self.make_windows(lines, window=window, stride=stride)
        
        all_answers = []
        seen_answers = set()
        raw_records = []
        
        pbar = tqdm(total=len(windows), desc=f"提取答案 {exam_name} {answers_file.name}", ncols=100, position=1)
        
        for s_line, e_line, chunk in windows:
            raw_out, data, meta = self.send_chunk_adaptive(
                s_line=s_line, e_line=e_line, chunk_text=chunk,
                max_retries=max_retries, retry_backoff=1.8, timeout=timeout,
                debug_dir=debug_dir, base_norm=exam_name,
                prompt_template=ANSWER_CHUNK_PROMPT_TEMPLATE,
                system_prompt=ANSWER_SYSTEM_PROMPT,
                task_type="answers"
            )
            
            # 保存原始记录
            raw_records.append({
                "window": f"{s_line}-{e_line}",
                "chunk_text": chunk,
                "compressed_chunk": meta["compressed_chunk"],
                "prompt_length": meta["prompt_len"],
                "model_output": raw_out,
                "response_length": meta["response_len"],
                "sanitized_used": meta["sanitized_used"],
                "parse_error": meta["parse_error"],
                "parsed_answers": data
            })
            
            if not isinstance(data, list):
                data = []
            
            # 添加调试信息
            if len(data) > 0:
                print(f"    窗口 {s_line}-{e_line}: 解析到 {len(data)} 个答案对象")
                if debug_dir:
                    # 保存模型返回的原始数据用于调试
                    debug_file = debug_dir / exam_name / "answers" / f"{s_line}-{e_line}_model_response.json"
                    debug_file.parent.mkdir(parents=True, exist_ok=True)
                    with debug_file.open("w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                print(f"    窗口 {s_line}-{e_line}: 未解析到答案对象")
            
            for obj in data:
                try:
                    # 安全地获取题号
                    question_number_raw = obj.get("question_number")
                    if question_number_raw is None:
                        print(f"    警告: 跳过缺少题号的答案对象: {obj}")
                        continue
                    
                    try:
                        question_number = int(question_number_raw)
                    except (ValueError, TypeError):
                        print(f"    警告: 跳过无效题号的答案对象: {obj}")
                        continue
                    
                    if question_number <= 0:
                        print(f"    警告: 跳过题号小于等于0的答案对象: {obj}")
                        continue
                    
                    answer = str(obj.get("answer") or "").strip()
                    explanation = str(obj.get("explanation") or "").strip()
                    knowledge_points = str(obj.get("knowledge_points") or "").strip()
                    
                    # 去重
                    answer_hash = hash(f"{question_number}_{answer}")
                    if answer_hash in seen_answers:
                        continue
                    seen_answers.add(answer_hash)

                    answer_obj = Question(
                        question_number=question_number,
                        question_type="",
                        question_text="",
                        answer=answer,
                        explanation=explanation,
                        knowledge_points=knowledge_points,
                        source_file=answers_file.stem,
                        source_window=(s_line, e_line)
                    )
                    all_answers.append(answer_obj)
                except Exception as e:
                    print(f"    警告: 处理答案对象时出错: {e}, 对象: {obj}")
                    continue
            
            pbar.update(1)
        pbar.close()
        # 按题号排序
        all_answers.sort(key=lambda x: x.question_number)        
        
        return all_answers, raw_records
    
    def validate_qa_matching(self, questions: List[Question], answers: List[Question], 
                           exam_name: str = "", raw_dir: Optional[Path] = None) -> Tuple[List[Question], List[int], List[int]]:
        # 处理重复的题目和答案
        questions_dict = {}
        answers_dict = {}
        
        # 处理重复题目：随机选择一个保留
        for question in questions:
            if question.question_number in questions_dict:
                if random.random() < 0.5:
                    continue  # 保留现有的
                else:
                    # 替换现有的题目
                    pass  # 直接覆盖
            questions_dict[question.question_number] = question
        
        # 处理重复答案：随机选择一个保留
        for answer in answers:
            if answer.question_number in answers_dict:
                if random.random() < 0.5:
                    continue  # 保留现有的
                else:
                    # 替换现有的答案
                    pass  # 直接覆盖
            answers_dict[answer.question_number] = answer
        
        # 获取去重后的题号和答案号
        question_numbers = set(questions_dict.keys())
        answer_numbers = set(answers_dict.keys())
        
        missing_answers = list(question_numbers - answer_numbers)
        extra_answers = list(answer_numbers - question_numbers)
        
        # 为题目添加答案
        final_questions = []
        for question_number, question in questions_dict.items():
            if question_number in answers_dict:
                answer_obj = answers_dict[question_number]
                question.answer = answer_obj.answer
                question.explanation = answer_obj.explanation
                question.knowledge_points = answer_obj.knowledge_points
            else:
                question.answer = ""
                question.explanation = ""
                question.knowledge_points = ""
            final_questions.append(question)
        
        # 按题号排序
        final_questions.sort(key=lambda x: x.question_number)
        
        # 保存对比原始输入和统计结果
        if raw_dir:
            validation_stats = {
                "exam_name": exam_name,
                "original_questions_count": len(questions),
                "original_answers_count": len(answers),
                "deduplicated_questions_count": len(questions_dict),
                "deduplicated_answers_count": len(answers_dict),
                "question_number_range": {
                    "min": min(question_numbers) if question_numbers else 0,
                    "max": max(question_numbers) if question_numbers else 0
                },
                "answer_number_range": {
                    "min": min(answer_numbers) if answer_numbers else 0,
                    "max": max(answer_numbers) if answer_numbers else 0
                },
                "matched_questions_count": len(question_numbers & answer_numbers),
                "missing_answers": missing_answers,
                "extra_answers": extra_answers,
                "duplicate_questions_removed": len(questions) - len(questions_dict),
                "duplicate_answers_removed": len(answers) - len(answers_dict),
                "duplicate_questions_details": self._find_duplicates(questions),
                "duplicate_answers_details": self._find_duplicates(answers)
            }
            
            validation_path = raw_dir / exam_name / "validation_stats.json"
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            with validation_path.open("w", encoding="utf-8") as f:
                json.dump(validation_stats, f, ensure_ascii=False, indent=2)
        
        # 打印匹配统计
        print(f"    原始题目数: {len(questions)} -> 去重后: {len(questions_dict)}")
        print(f"    原始答案数: {len(answers)} -> 去重后: {len(answers_dict)}")
        print(f"    题目题号范围: {min(question_numbers) if question_numbers else 0} - {max(question_numbers) if question_numbers else 0}")
        print(f"    答案题号范围: {min(answer_numbers) if answer_numbers else 0} - {max(answer_numbers) if answer_numbers else 0}")
        print(f"    匹配的题目数: {len(question_numbers & answer_numbers)}")
        print(f"    重复题目移除: {len(questions) - len(questions_dict)}")
        print(f"    重复答案移除: {len(answers) - len(answers_dict)}")
        
        return final_questions, missing_answers, extra_answers
    
    def _find_duplicates(self, items: List[Question]) -> Dict[int, List[Dict]]:
        """查找重复项并返回详细信息"""
        duplicates = {}
        seen = {}
        
        for item in items:
            if item.question_number in seen:
                if item.question_number not in duplicates:
                    duplicates[item.question_number] = [seen[item.question_number]]
                duplicates[item.question_number].append({
                    "question_number": item.question_number,
                    "question_text": item.question_text[:100] + "..." if len(item.question_text) > 100 else item.question_text,
                    "answer": item.answer,
                    "source_window": item.source_window,
                    "source_file": item.source_file
                })
            else:
                seen[item.question_number] = {
                    "question_number": item.question_number,
                    "question_text": item.question_text[:100] + "..." if len(item.question_text) > 100 else item.question_text,
                    "answer": item.answer,
                    "source_window": item.source_window,
                    "source_file": item.source_file
                }
        
        return duplicates
    
    def save_to_json(self, data: List[ExamData], output_file: str):
        serializable_data = []
        for exam in data:
            exam_dict = {
                'exam_name': exam.exam_name,
                'total_questions': exam.total_questions,
                'missing_answers': exam.missing_answers,
                'extra_answers': exam.extra_answers,
                'questions': []
            }
            
            for question in exam.questions:
                question_dict = {
                    'question_number': question.question_number,
                    'question_type': question.question_type,
                    'question_text': question.question_text,
                    'answer': question.answer,
                    'explanation': question.explanation,
                    'knowledge_points': question.knowledge_points,
                    'source_file': question.source_file,
                    'source_window': question.source_window
                }
                exam_dict['questions'].append(question_dict)
                
            serializable_data.append(exam_dict)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到: {output_file}")
    
    def save_to_jsonl(self, data: List[ExamData], output_file: str):
        """保存为JSONL格式，每行一个题目"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for exam in data:
                for question in exam.questions:
                    # 创建新的数据结构，按照你的要求重新组织
                    new_question = {
                        'qid': question.question_number,
                        'question': question.question_text,
                        'answer': question.answer,
                        'explanation': question.explanation,
                        'knowledge_points': question.knowledge_points,
                        'source_file': exam.exam_name,  # 使用exam_name替换source_file
                        'source_window': question.source_window
                    }
                    # 写入JSONL格式（每行一个JSON对象）
                    f.write(json.dumps(new_question, ensure_ascii=False) + '\n')
        print(f"数据已保存到: {output_file}")
    
    def save_single_exam(self, exam_data: ExamData, output_dir: Path):
        """保存单个考试的结果"""
        exam_file = output_dir / f"{exam_data.exam_name}_qa.json"
        exam_dict = {
            'exam_name': exam_data.exam_name,
            'total_questions': exam_data.total_questions,
            'missing_answers': exam_data.missing_answers,
            'extra_answers': exam_data.extra_answers,
            'questions': []
        }
        
        for question in exam_data.questions:
            question_dict = {
                'question_number': question.question_number,
                'question_type': question.question_type,
                'question_text': question.question_text,
                'answer': question.answer,
                'explanation': question.explanation,
                'knowledge_points': question.knowledge_points,
                'source_file': question.source_file,
                'source_window': question.source_window
            }
            exam_dict['questions'].append(question_dict)
        
        with open(exam_file, 'w', encoding='utf-8') as f:
            json.dump(exam_dict, f, ensure_ascii=False, indent=2)
        print(f"    💾 单个考试保存完成: {exam_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用AI模型提取考试题目和答案")
    parser.add_argument("--base-path", type=str, help="考试真题目录的基础路径")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_LINES,
                       help="滑动窗口大小（行数）")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE_LINES,
                       help="滑动窗口步长（行数）")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="AI调用最大重试次数")
    parser.add_argument("--timeout", type=int, default=DEFAULT_ARK_TIMEOUT,
                       help="AI调用超时时间（秒）")
    parser.add_argument("--max-workers", type=int, default=256,
                       help="并发处理数")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="输出目录")
    parser.add_argument("--save-raw-dir", type=str, default="",
                       help="保存原文片段对照的目录（每文件 raw_chunks.jsonl）")
    parser.add_argument("--debug-log-dir", type=str, default="",
                       help="保存每个窗口的 prompt/response/parse_error/sanitized.json")

    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建调试和原始记录目录
    debug_dir = Path(args.debug_log_dir) if args.debug_log_dir else None
    raw_dir = Path(args.save_raw_dir) if args.save_raw_dir else None
    
    # 创建提取器
    extractor = QAExtractor(args.base_path)
    
    # 提取所有考试数据
    all_exams_data = extractor.extract_all_exams(
        window=args.window,
        stride=args.stride,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_workers=args.max_workers,
        debug_dir=debug_dir,
        raw_dir=raw_dir,
        output_dir=output_dir
    )
    
    if not all_exams_data:
        print("未提取到任何数据")
        return
    
    # 保存结果
    output_file = output_dir / "qa_extracted.jsonl"
    extractor.save_to_jsonl(all_exams_data, str(output_file))


if __name__ == "__main__":
    main()




'''python 3.2_qa_apart.py \
  --base-path /home/wangxi/workspace/gongye/yijizaojia/qna_split \
  --window 80 \
  --stride 60 \
  --max-retries 3 \
  --timeout 120 \
  --max-workers 256 \
  --output-dir yijizaojia/qa_apart_output_0819 \
  --save-raw-dir yijizaojia/raw_logs_0819 \
  --debug-log-dir yijizaojia/debug_logs_0819'''