import os
import sys
import json
import re

from PIL import Image
Image.MAX_IMAGE_PIXELS = 600000000
from dots_ocr.utils.image_utils import PILimage_to_base64


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.
    
    Args:
        text (str): The string to check.
        
    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False
    
    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r'\$\$.*?\$\$',           # Block-level math formula $$...$$
        r'\$[^$\n]+?\$',          # Inline math formula $...$
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environment \begin{...}...\end{...}
        r'\\[a-zA-Z]+\{.*?\}',    # LaTeX command \command{...}
        r'\\[a-zA-Z]+',           # Simple LaTeX command \command
        r'\\\[.*?\\\]',           # Display math formula \[...\]
        r'\\\(.*?\\\)',           # Inline math formula \(...\)
    ]
    
    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.
    
    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r'\\documentclass\{[^}]+\}',  # \documentclass{...}
        r'\\usepackage\{[^}]+\}',    # \usepackage{...}
        r'\\usepackage\[[^\]]*\]\{[^}]+\}',  # \usepackage[options]{...}
        r'\\begin\{document\}',       # \begin{document}
        r'\\end\{document\}',         # \end{document}
    ]
    
    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text
    

def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.
    
    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if it's already enclosed in $$
    if text.startswith('$$') and text.endswith('$$'):
        text_new = text[2:-2].strip()
        if not '$' in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith('\\[') and text.endswith('\\]'):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"
        
    # Check if it's enclosed in \[ \]
    if len(re.findall(r'.*\\\[.*\\\].*', text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r'\$([^$]+)\$'
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text  

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):  
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if 'usepackage' in text:
        text = clean_latex_preamble(text)

    if text[0] == '`' and text[-1] == '`':
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text 


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.
    
    Args:
        text: The original text.
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple consecutive whitespace characters with a single space
    if text[:2] == '`$' and text[-2:] == '$`':
        text = text[1:-1]
    
    return text


def layoutjson2md(image: Image.Image, cells: list, text_key: str = 'text', no_page_hf: bool = False) -> str:
    """
    Converts a layout JSON format to Markdown.
    
    In the layout JSON, formulas are LaTeX, tables are HTML, and text is Markdown.
    
    Args:
        image: A PIL Image object.
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_header_footer: If True, skips page headers and footers.
        
    Returns:
        str: The text in Markdown format.
    """
    text_items = []

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]
        text = cell.get(text_key, "")
        
        if no_page_hf and cell['category'] in ['Page-header', 'Page-footer']:
            continue
        
        if cell['category'] == 'Picture':
            image_crop = image.crop((x1, y1, x2, y2))
            image_base64 = PILimage_to_base64(image_crop)
            text_items.append(f"![]({image_base64})")
        elif cell['category'] == 'Formula':
            text_items.append(get_formula_in_markdown(text))
        else:            
            text = clean_text(text)
            text_items.append(f"{text}")

    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def fix_streamlit_formulas(md: str) -> str:
    """
    Fixes the format of formulas in Markdown to ensure they display correctly in Streamlit.
    It adds a newline after the opening $$ and before the closing $$ if they don't already exist.
    
    Args:
        md_text (str): The Markdown text to fix.
        
    Returns:
        str: The fixed Markdown text.
    """
    
    # This inner function will be used by re.sub to perform the replacement
    def replace_formula(match):
        content = match.group(1)
        # If the content already has surrounding newlines, don't add more.
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        return f'$$\n{content}\n$$'
    
    # Use regex to find all $$....$$ patterns and replace them using the helper function.
    return re.sub(r'\$\$(.*?)\$\$', replace_formula, md, flags=re.DOTALL)

# 在 dots_ocr/utils/format_transformer.py 中
def layoutjson2md_full_robust(image: Image.Image, cells: list, text_key: str = 'text', no_page_hf: bool = False) -> str:
    """
    转换 layout JSON 到 Markdown。
    - 安全处理无效bbox (修复 right < left 问题)。
    - 将坐标裁剪到图像边界内。
    - 检查并跳过零面积的裁剪区域。
    - 如果类别是 'Text'，移除换行符。
    - 如果类别是 'Section-header'，正确处理空格，保留Markdown语法。
    """
    if not isinstance(cells, list):
        print(f"警告：cells 数据不是一个列表，而是 {type(cells)} 类型。无法处理。")
        return ""
        
    img_width, img_height = image.size
    text_items = []
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue

        category = cell.get('category', 'Text')
        text = cell.get(text_key, "")
        
        if no_page_hf and category in ['Page-header', 'Page-footer']:
            continue
        
        if category == 'Picture':
            try:
                # --- 坐标清洗与验证层 ---
                coords = [int(coord) for coord in cell['bbox']]
                x1, y1, x2, y2 = coords

                # 1. 修正坐标顺序
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1

                # 2. 裁剪坐标到图像边界内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                # 3. 检查裁剪区域是否有效
                if x1 >= x2 or y1 >= y2:
                    print(f"警告：跳过一个图片单元格，因为其有效裁剪区域面积为零。原始Bbox: {cell.get('bbox')}")
                    continue
                # --- 清洗结束 ---

                image_crop = image.crop((x1, y1, x2, y2))
                image_base64 = PILimage_to_base64(image_crop)
                text_items.append(f"![]({image_base64})")
                
            except (KeyError, TypeError, ValueError) as e:
                print(f"警告：因 bbox 严重无效，跳过一个图片单元格。错误: {e}. Cell: {cell}")
                continue
        elif category == 'Formula':
            text_items.append(get_formula_in_markdown(text))
        else:
            if category == 'Section-header':
                # 步骤 1: 直接移除所有换行符，实现文本合并
                processed_text = text.replace('\n', '').replace('\r', '')
                # 步骤 2: 使用 split 和 join 来规范化剩余的空格（例如多个空格变一个）
                # 这样可以保留 "# 3" 中的空格，同时清理多余的空格。
                text = ' '.join(processed_text.split())

            elif category == 'Text':
                # 同样，对于普通文本，也直接移除换行符进行合并
                # 这更符合中文等无空格分词语言的习惯。
                text = text.replace('\n', '').replace('\r', '')
            
            # clean_text 会处理首尾空格和一些特殊情况
            text = clean_text(text)
            text_items.append(f"{text}")

    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def layoutjson2md_simple_extract(cells: list, text_key: str = 'text', no_page_hf: bool = False) -> str:
    # 它只按顺序提取文本，别的什么都不做。
    if not isinstance(cells, list): return ""
    md_lines = []
    for cell in cells:
        if not isinstance(cell, dict): continue
        if no_page_hf and cell.get('category', '') in ['Page-header', 'Page-footer']: continue
        text = cell.get(text_key, '').strip()
        if text: md_lines.append(text)
    return "\n\n".join(md_lines)
