from PIL import Image
Image.MAX_IMAGE_PIXELS = 600000000
from typing import Dict, List, Tuple

import fitz
from io import BytesIO
import json
import re
from json_repair import loads as json_repair_loads

from dots_ocr.utils.image_utils import smart_resize
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.output_cleaner import OutputCleaner


# Define a color map (using RGBA format)
dict_layout_type_to_color = {
    "Text": (0, 128, 0, 256),  # Green, translucent
    "Picture": (255, 0, 255, 256),  # Magenta, translucent
    "Caption": (255, 165, 0, 256),  # Orange, translucent
    "Section-header": (0, 255, 255, 256),  # Cyan, translucent
    "Footnote": (0, 128, 0, 256),  # Green, translucent
    "Formula": (128, 128, 128, 256),  # Gray, translucent
    "Table": (255, 192, 203, 256),  # Pink, translucent
    "Title": (255, 0, 0, 256),  # Red, translucent
    "List-item": (0, 0, 255, 256),  # Blue, translucent
    "Page-header": (0, 128, 0, 256),  # Green, translucent
    "Page-footer":  (128, 0, 128, 256),  # Purple, translucent
    "Other": (165, 42, 42, 256),  # Brown, translucent
    "Unknown": (0, 0, 0, 0),
}


def draw_layout_on_image(image, cells, resized_height=None, resized_width=None, fill_bbox=True, draw_bbox=True):
    """
    Draw transparent boxes on an image.
    
    Args:
        image: The source PIL Image.
        cells: A list of cells containing bounding box information.
        resized_height: The resized height.
        resized_width: The resized width.
        fill_bbox: Whether to fill the bounding box.
        draw_bbox: Whether to draw the bounding box.
        
    Returns:
        PIL.Image: The image with drawings.
    """
    # origin_image = Image.open(image_path)
    original_width, original_height = image.size
        
    # Create a new PDF document
    doc = fitz.open()
    
    # Get image information
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    # pix = fitz.Pixmap(image_path)
    pix = fitz.Pixmap(img_bytes)
    
    # Create a page
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(
        fitz.Rect(0, 0, pix.width, pix.height), 
        # filename=image_path
        pixmap=pix
        )

    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        layout_type = cell['category']
        order = i
        
        top_left = (bbox[0], bbox[1])
        down_right = (bbox[2], bbox[3])
        if resized_height and resized_width:
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            top_left = (int(bbox[0] / scale_x), int(bbox[1] / scale_y))
            down_right = (int(bbox[2] / scale_x), int(bbox[3] / scale_y))
            
        color = dict_layout_type_to_color.get(layout_type, (0, 128, 0, 256))
        color = [col/255 for col in color[:3]]

        x0, y0, x1, y1 = top_left[0], top_left[1], down_right[0], down_right[1]
        rect_coords = fitz.Rect(x0, y0, x1, y1)
        if draw_bbox:
            if fill_bbox:
                page.draw_rect(
                    rect_coords,
                    color=None,
                    fill=color,
                    fill_opacity=0.3,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
            else:
                page.draw_rect(
                    rect_coords,
                    color=color,
                    fill=None,
                    fill_opacity=1,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
        order_cate = f"{order}_{layout_type}"
        page.insert_text(
            (x1, y0 + 20), order_cate, fontsize=20, color=color
        )  # Insert the index in the top left corner of the rectangle

    # Convert to a Pixmap (maintaining original dimensions)
    mat = fitz.Matrix(1.0, 1.0)
    pix = page.get_pixmap(matrix=mat)

    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def pre_process_bboxes(
    origin_image,
    bboxes,
    input_width,
    input_height,
    factor: int = 28,
    min_pixels: int = 3136, 
    max_pixels: int = 11289600
):
    assert isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list)
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    original_width, original_height = origin_image.size

    input_height, input_width = smart_resize(input_height, input_width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    bboxes_out = []
    for bbox in bboxes:
        bbox_resized = [
            int(float(bbox[0]) / scale_x), 
            int(float(bbox[1]) / scale_y),
            int(float(bbox[2]) / scale_x), 
            int(float(bbox[3]) / scale_y)
        ]
        bboxes_out.append(bbox_resized)
    
    return bboxes_out

def post_process_cells(
    origin_image: Image.Image, 
    cells: List[Dict], 
    input_width,  # server input width, also has smart_resize in server
    input_height,
    factor: int = 28,
    min_pixels: int = 3136, 
    max_pixels: int = 11289600
) -> List[Dict]:
    """
    Post-processes cell bounding boxes, converting coordinates from the resized dimensions back to the original dimensions.
    
    Args:
        origin_image: The original PIL Image.
        cells: A list of cells containing bounding box information.
        input_width: The width of the input image sent to the server.
        input_height: The height of the input image sent to the server.
        factor: Resizing factor.
        min_pixels: Minimum number of pixels.
        max_pixels: Maximum number of pixels.
        
    Returns:
        A list of post-processed cells.
    """
    assert isinstance(cells, list) and len(cells) > 0 and isinstance(cells[0], dict)
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    original_width, original_height = origin_image.size

    input_height, input_width = smart_resize(input_height, input_width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    scale_x = input_width / original_width
    scale_y = input_height / original_height
    
    cells_out = []
    for cell in cells:
        bbox = cell['bbox']
        bbox_resized = [
            int(float(bbox[0]) / scale_x), 
            int(float(bbox[1]) / scale_y),
            int(float(bbox[2]) / scale_x), 
            int(float(bbox[3]) / scale_y)
        ]
        cell_copy = cell.copy()
        cell_copy['bbox'] = bbox_resized
        cells_out.append(cell_copy)
    
    return cells_out

def is_legal_bbox(cells):
    for cell in cells:
        bbox = cell['bbox']
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False
    return True


_REPEAT_TRIGGER = 10
_REPEAT_LIMIT = 5
_MAX_SUBSTRING_UNITS = 24
_MAX_SEQUENCE_TOKENS = 4


def _collapse_repeated_substrings(value: str, trigger: int, limit: int) -> str:
    if not value or trigger <= limit:
        return value
    max_unit = min(_MAX_SUBSTRING_UNITS, max(0, len(value) // trigger))
    if max_unit <= 0:
        return value
    for unit_len in range(1, max_unit + 1):
        pattern = re.compile(r'((?:.{{{}}}))\1{{{},}}'.format(unit_len, trigger - 1), re.DOTALL)
        while True:
            match = pattern.search(value)
            if not match:
                break
            unit = match.group(1)
            value = value[:match.start()] + unit * limit + value[match.end():]
    return value


def _tokenize_with_prefix(text: str) -> Tuple[List[Dict[str, str]], str]:
    entries: List[Dict[str, str]] = []
    pos = 0
    for match in re.finditer(r'\S+', text):
        leading_ws = text[pos:match.start()]
        token = match.group(0)
        entries.append({'leading_ws': leading_ws, 'token': token})
        pos = match.end()
    trailing_ws = text[pos:]
    return entries, trailing_ws


def _limit_repeated_token_sequences(entries: List[Dict[str, str]], trigger: int, limit: int) -> None:
    if trigger <= limit or len(entries) < trigger:
        return
    max_seq = min(_MAX_SEQUENCE_TOKENS, len(entries))
    for seq_len in range(max_seq, 0, -1):
        i = 0
        while i + seq_len <= len(entries):
            seq = tuple(entry['token'] for entry in entries[i:i + seq_len])
            run_len = 1
            j = i + seq_len
            while j + seq_len <= len(entries):
                next_seq = tuple(entry['token'] for entry in entries[j:j + seq_len])
                if next_seq != seq:
                    break
                run_len += 1
                j += seq_len
            if run_len >= trigger:
                keep_sequences = min(limit, run_len)
                remove_start = i + keep_sequences * seq_len
                remove_end = i + run_len * seq_len
                del entries[remove_start:remove_end]
                continue
            i += 1


def sanitize_repeated_tokens_in_text(text: str, trigger: int = _REPEAT_TRIGGER, limit: int = _REPEAT_LIMIT) -> str:
    if not text:
        return text
    cleaned = _collapse_repeated_substrings(text, trigger, limit)
    entries, trailing_ws = _tokenize_with_prefix(cleaned)
    if not entries:
        return cleaned
    for entry in entries:
        entry['token'] = _collapse_repeated_substrings(entry['token'], trigger, limit)
    if len(entries) >= trigger:
        _limit_repeated_token_sequences(entries, trigger, limit)
    return ''.join(entry['leading_ws'] + entry['token'] for entry in entries) + trailing_ws


def sanitize_cells_repeated_tokens(cells: List[Dict], trigger: int = _REPEAT_TRIGGER, limit: int = _REPEAT_LIMIT) -> List[Dict]:
    if not cells or trigger <= limit:
        return cells
    for cell in cells:
        text = cell.get('text')
        if isinstance(text, str) and text:
            new_text = sanitize_repeated_tokens_in_text(text, trigger, limit)
            if new_text != text:
                cell['text'] = new_text
    return cells

# 增加json_reapir的修复
def post_process_output(response, prompt_mode, origin_image, input_image, min_pixels=None, max_pixels=None):
    if prompt_mode in ["prompt_ocr", "prompt_table_html", "prompt_table_latex", "prompt_formula_latex"]:
        return response, False # 注意：为了保持返回签名一致，这里也返回一个元组

    cells = response
    try:
        # 使用 json_repair.loads 直接进行解析，它更强大
        cells = json_repair_loads(cells)
        
        # 确保修复后的结果是列表，否则视为失败
        if not isinstance(cells, list):
            raise TypeError(f"Repaired JSON is not a list, but {type(cells)}")

        cells = post_process_cells(
            origin_image, 
            cells,
            input_image.width,
            input_image.height,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        sanitize_cells_repeated_tokens(cells)
        return cells, False
    except Exception as e:
        # 如果连 json_repair 都失败了，说明输出格式问题很严重
        print(f"CRITICAL: JSON repair failed. Error: {e}, when using {prompt_mode}")
        
        # 触发旧的降级清理逻辑
        cleaner = OutputCleaner()
        response_clean = cleaner.clean_model_output(response) # 使用原始 response
        if isinstance(response_clean, list):
            response_clean = "\n\n".join([cell['text'] for cell in response_clean if 'text' in cell])
        return response_clean, True
