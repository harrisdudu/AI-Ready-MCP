import re
from collections import OrderedDict

def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def remove_image(text: str) -> str:
    """删除 markdown 中的图片链接"""
    return re.sub(r'!\[.*?]\([^)]*\)', '', text)

def remove_toc_lines_basic(markdown_text):
    """
    删除包含多个点的目录行（基础版本）
    """
    # 匹配以点或空格开头，包含多个点，以数字结尾的行
    pattern = r'^[\s.]*\.{3,}[\s\d]*\d+\s*$'
    
    lines = markdown_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        if not re.match(pattern, line.strip()):
            if "......" not in line:
                filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)



def remove_table(text: str) -> str:
    """删除 markdown 中的 HTML 表格"""
    # 匹配 HTML 表格标签的正则表达式
    html_table_pattern = r'<table>.*?</table>'
    
    # 使用 DOTALL 标志让 . 匹配包括换行符在内的所有字符
    return re.sub(html_table_pattern, '', text, flags=re.DOTALL)

def extract_items(text: str, item_type: str) -> list:
    if item_type == "Q":
        pattern = r'<Q>(.*?)</Q>'
    elif item_type == "A":
        pattern = r'<A>(.*?)</A>'
    elif item_type == "COT":
        pattern = r'<COT>(.*?)</COT>'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches

def extract_tags_and_content(text: str) -> list:
    """提取文本中 <Q></Q>、<COT></COT> 和 <A></A> 标签对及其内容"""
    # 匹配三种标签对及其内容（包括标签）
    pattern = r'(<(Q|COT|A)>(.*?)</\2>)'
    result = re.findall(pattern, text, flags=re.DOTALL)
    res = {}
    for item in result:
        (_, t, content) = item
        res[t] = content
    return res

def extract_ref_num(text:str) -> list:
    """提取文本中{ref_i}中的数字内容"""
    pattern = r'\{ref_(\d+)\}'
    result = re.findall(pattern, text)
    result = [int(s) for s in result]
    return result


def trim_content(text: str) -> str:
    # 按句子结束符号分句（包括换行）
    sentences = re.split(r'([。！？；;\n])', text)
    sentences = ["".join(sentences[i:i+2]).strip() 
                 for i in range(0, len(sentences), 2) if "".join(sentences[i:i+2]).strip()]
    
    if not sentences:
        return text  # 如果无法分句，就原样返回

    n = len(sentences)
    start = int(n * 0.1)   # 前 10%
    end = int(n * 0.9)     # 后 10%

    return "".join(sentences[start:end])

def create_interval_mapping(full_text):
    """
    创建区间到页码的映射（更高效）
    返回:
    intervals: 列表，每个元素为 (start, end, page)
    """
    intervals = []
    
    # 查找所有页码标签
    page_tags = list(re.finditer(r'<special_page_num_tag>(\d+)</special_page_num_tag>', full_text))
    
    if not page_tags:
        return [(0, len(full_text), 1)]
    
    # 第一个区间：文档开始到第一个页码标签
    intervals.append((0, page_tags[0].start(), 1))
    
    # 处理每个页码标签对应的区间
    for i, match in enumerate(page_tags):
        page_num = int(match.group(1))
        start_pos = match.end()
        
        if i < len(page_tags) - 1:
            end_pos = page_tags[i + 1].start()
        else:
            end_pos = len(full_text)
        
        intervals.append((start_pos, end_pos, page_num))
    
    return intervals

def find_pages_by_range(intervals, start, end):
    """
    使用二分查找确定位置范围所覆盖的所有页码
    
    返回:
    pages: 有序的页码列表
    """
    pages = OrderedDict()  # 使用有序字典来保持页码顺序并去重
    
    # 找到起始位置所在的区间
    low, high = 0, len(intervals) - 1
    start_index = -1
    
    while low <= high:
        mid = (low + high) // 2
        s, e, p = intervals[mid]
        
        if s <= start < e:
            start_index = mid
            break
        elif start < s:
            high = mid - 1
        else:
            low = mid + 1
    
    # 如果找不到起始区间，使用第一个或最后一个区间
    if start_index == -1:
        if start < intervals[0][0]:
            start_index = 0
        else:
            start_index = len(intervals) - 1
    
    # 从起始区间开始，遍历直到覆盖结束位置
    for i in range(start_index, len(intervals)):
        s, e, p = intervals[i]
        
        # 如果当前区间与片段有重叠，添加页码
        if not (e <= start or s >= end):
            pages[p] = True  # 使用字典键来去重
        
        # 如果已经超过结束位置，停止遍历
        if s >= end:
            break
    
    return list(pages.keys())

def remove_special_page_tags_with_info(content: str):
    """
    去除内容中的所有 special_page_num_tag 并将其替换为换行符，
    同时返回所有被移除的页码
    
    参数:
    content: 包含特殊标签的文本内容
    
    返回:
    (处理后的文本内容, 页码列表)
    """
    # 定义正则表达式模式
    pattern = r'<special_page_num_tag>(\d+)</special_page_num_tag>'
    
    # 替换所有标签为换行符
    result = re.sub(pattern, '\n', content)
    
    return result


