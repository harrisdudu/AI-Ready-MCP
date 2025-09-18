import os
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
import win32com.client as win32

def ppt_to_pdf_with_notes(ppt_path, pdf_path, font_path=None, notes_font_size=24):
    # 打开 PowerPoint
    ppt_app = win32.Dispatch("PowerPoint.Application")
    ppt_app.Visible = True  # 必须 True，否则 COM 有些操作会失败

    # 打开演示文稿
    presentation = ppt_app.Presentations.Open(ppt_path, WithWindow=True)

    temp_images = []

    for i, slide in enumerate(presentation.Slides, start=1):
        # 导出幻灯片为图片
        slide_img_path = os.path.join(os.path.dirname(pdf_path), f"slide_{i}.png")
        slide.Export(slide_img_path, "PNG")

        # 打开图片
        slide_img = Image.open(slide_img_path).convert("RGB")

        # 获取备注
        notes_text = ""
        if slide.NotesPage.Shapes.Placeholders.Count >= 2:
            notes_shape = slide.NotesPage.Shapes.Placeholders(2)
            if notes_shape.HasTextFrame:
                notes_text = notes_shape.TextFrame.TextRange.Text.strip()

        # 在幻灯片下方添加备注
        if notes_text:
            width, height = slide_img.size
            notes_height = height // 4
            new_img = Image.new("RGB", (width, height + notes_height), (255, 255, 255))
            new_img.paste(slide_img, (0, 0))

            draw = ImageDraw.Draw(new_img)

            # 使用中文字体
            if font_path is None:
                font_path = "C:\\Windows\\Fonts\\msyh.ttc"  # 微软雅黑
            font = ImageFont.truetype(font_path, notes_font_size)

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
            new_slide_img_path = os.path.join(os.path.dirname(pdf_path), f"slide_{i}_notes.png")
            new_img.save(new_slide_img_path)
            temp_images.append(new_slide_img_path)
        else:
            temp_images.append(slide_img_path)

    presentation.Close()
    ppt_app.Quit()

    # 生成 PDF
    c = canvas.Canvas(pdf_path)
    for img_path in temp_images:
        img = Image.open(img_path)
        width, height = img.size
        c.setPageSize((width, height))
        c.drawImage(img_path, 0, 0, width=width, height=height)
        c.showPage()
    c.save()

    print("PDF生成完成:", pdf_path)

ppt_to_pdf_with_notes(r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座.pptx", r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座.pptx.pdf")