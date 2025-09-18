import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
import win32com.client as win32
import tempfile
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ppt_to_pdf_with_notes(ppt_path, pdf_path, font_path=None, notes_font_size=24, delete_temp=True):
    """
    将PowerPoint演示文稿转换为包含备注的PDF文件

    参数:
    ppt_path (str): PowerPoint文件路径
    pdf_path (str): 输出PDF文件路径
    font_path (str, optional): 中文字体路径，默认为微软雅黑
    notes_font_size (int, optional): 备注文字大小，默认为24
    delete_temp (bool, optional): 是否删除临时文件，默认为True

    返回:
    bool: 转换是否成功
    """
    try:
        # 验证输入文件存在
        if not os.path.exists(ppt_path):
            logger.error(f"PowerPoint文件不存在: {ppt_path}")
            return False

        # 确保输出目录存在
        output_dir = os.path.dirname(pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        logger.info(f"创建临时目录: {temp_dir}")

        # 打开PowerPoint
        logger.info("启动PowerPoint应用程序...")
        ppt_app = win32.Dispatch("PowerPoint.Application")
        ppt_app.Visible = True  # 必须为True，否则某些COM操作会失败

        # 打开演示文稿
        try:
            logger.info(f"打开演示文稿: {ppt_path}")
            presentation = ppt_app.Presentations.Open(ppt_path, WithWindow=True)
        except Exception as e:
            logger.error(f"打开演示文稿失败: {str(e)}")
            ppt_app.Quit()
            return False

        temp_images = []

        try:
            total_slides = len(presentation.Slides)
            logger.info(f"发现 {total_slides} 张幻灯片")

            for i, slide in enumerate(presentation.Slides, start=1):
                logger.info(f"处理幻灯片 {i}/{total_slides}")
                # 导出幻灯片为图片
                slide_img_path = os.path.join(temp_dir, f"slide_{i}.png")
                slide.Export(slide_img_path, "PNG")

                # 打开图片
                slide_img = Image.open(slide_img_path).convert("RGB")

                # 获取备注
                notes_text = ""
                try:
                    if slide.NotesPage.Shapes.Placeholders.Count >= 2:
                        notes_shape = slide.NotesPage.Shapes.Placeholders(2)
                        if notes_shape.HasTextFrame:
                            notes_text = notes_shape.TextFrame.TextRange.Text.strip()
                            logger.debug(f"幻灯片 {i} 备注: {notes_text[:30]}...")
                except Exception as e:
                    logger.warning(f"提取幻灯片 {i} 备注失败: {str(e)}")

                # 在幻灯片下方添加备注
                if notes_text:
                    width, height = slide_img.size
                    notes_height = height // 4
                    new_img = Image.new("RGB", (width, height + notes_height), (255, 255, 255))
                    new_img.paste(slide_img, (0, 0))

                    draw = ImageDraw.Draw(new_img)

                    # 使用中文字体
                    if font_path is None:
                        # 尝试使用系统中的微软雅黑字体
                        font_candidates = [
                            "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
                            "/System/Library/Fonts/PingFang.ttc",  # macOS
                            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Linux
                        ]
                        for candidate in font_candidates:
                            if os.path.exists(candidate):
                                font_path = candidate
                                break
                        if font_path is None:
                            logger.warning("未找到中文字体，可能导致乱码")
                            font_path = ImageFont.load_default()

                    try:
                        font = ImageFont.truetype(font_path, notes_font_size)
                    except Exception as e:
                        logger.warning(f"加载字体失败: {str(e)}，使用默认字体")
                        font = ImageFont.load_default()

                    # 自动换行
                    lines = []
                    for paragraph in notes_text.splitlines():
                        paragraph = paragraph.strip()
                        if not paragraph:
                            continue
                        current_line = ""
                        for char in paragraph:
                            test_line = current_line + char
                            w = font.getlength(test_line)
                            if w > width - 20:
                                lines.append(current_line)
                                current_line = char
                            else:
                                current_line = test_line
                        if current_line:
                            lines.append(current_line)

                    # 写备注文字
                    y = height + 10
                    for line in lines:
                        draw.text((10, y), line, fill=(0, 0, 0), font=font)
                        y += notes_font_size + 4

                    # 保存带备注的临时图片
                    new_slide_img_path = os.path.join(temp_dir, f"slide_{i}_notes.png")
                    new_img.save(new_slide_img_path)
                    temp_images.append(new_slide_img_path)
                else:
                    temp_images.append(slide_img_path)

            # 生成PDF
            logger.info(f"开始生成PDF文件: {pdf_path}")
            c = canvas.Canvas(pdf_path)
            for img_path in temp_images:
                img = Image.open(img_path)
                width, height = img.size
                c.setPageSize((width, height))
                c.drawImage(img_path, 0, 0, width=width, height=height)
                c.showPage()
            c.save()
            logger.info(f"PDF生成完成: {pdf_path}")

        finally:
            # 关闭演示文稿和PowerPoint
            presentation.Close()
            ppt_app.Quit()
            logger.info("已关闭PowerPoint应用程序")

            # 清理临时文件
            if delete_temp and os.path.exists(temp_dir):
                logger.info(f"清理临时目录: {temp_dir}")
                shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        logger.error(f"转换过程中发生错误: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # 示例用法
    input_ppt = r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座.pptx"
    output_pdf = r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座_with_notes.pdf"
    
    # 调用函数
    success = ppt_to_pdf_with_notes(input_ppt, output_pdf)
    
    if success:
        logger.info("转换成功！")
    else:
        logger.error("转换失败！")