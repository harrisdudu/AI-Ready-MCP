# PPT转PDF带备注工具

这个工具可以将PowerPoint演示文稿转换为包含备注的PDF文件，解决了原代码中可能存在的一些问题，并添加了新功能。

## 功能特点

- 提取幻灯片备注并将其添加到PDF中
- 支持中文字体显示
- 自动换行处理备注文本
- 提供单文件转换和批量转换功能
- 完善的错误处理和日志记录
- 临时文件管理（可选择保留或删除）
- 可自定义备注字体大小

## 文件说明

1. `improved_ppt2pdf.py`: 主脚本，包含`ppt_to_pdf_with_notes`函数
2. `test_ppt2pdf.py`: 测试脚本，演示如何使用主函数
3. `README.md`: 本说明文件

## 安装依赖

运行前需要安装以下Python库：

```bash
pip install pillow reportlab pywin32
```

## 函数说明

### `ppt_to_pdf_with_notes`函数

```python
ppt_to_pdf_with_notes(ppt_path, pdf_path, font_path=None, notes_font_size=24, delete_temp=True)
```

**参数:**
- `ppt_path` (str): PowerPoint文件路径
- `pdf_path` (str): 输出PDF文件路径
- `font_path` (str, optional): 中文字体路径，默认为系统中的微软雅黑
- `notes_font_size` (int, optional): 备注文字大小，默认为24
- `delete_temp` (bool, optional): 是否删除临时文件，默认为True

**返回:**
- `bool`: 转换是否成功

### `batch_convert_ppts_to_pdfs`函数

```python
batch_convert_ppts_to_pdfs(input_dir, output_dir, **kwargs)
```

**参数:**
- `input_dir` (str): 包含PowerPoint文件的目录
- `output_dir` (str): 输出PDF文件的目录
- `**kwargs`: 传递给`ppt_to_pdf_with_notes`的其他参数

**返回:**
- `dict`: 包含每个文件转换结果的字典

## 使用示例

### 单个文件转换

```python
from improved_ppt2pdf import ppt_to_pdf_with_notes

input_ppt = r"C:\path\to\your\presentation.pptx"
output_pdf = r"C:\path\to\output.pdf"

success = ppt_to_pdf_with_notes(
    input_ppt,
    output_pdf,
    notes_font_size=20,
    delete_temp=True
)

if success:
    print("转换成功！")
else:
    print("转换失败！")
```

### 批量转换

```python
from test_ppt2pdf import batch_convert_ppts_to_pdfs

input_dir = r"C:\path\to\ppts"
output_dir = r"C:\path\to\pdfs"

batch_results = batch_convert_ppts_to_pdfs(
    input_dir,
    output_dir,
    notes_font_size=18,
    delete_temp=True
)
```

## 注意事项

1. 运行此工具需要安装Microsoft PowerPoint
2. 确保PowerPoint文件路径和输出PDF路径使用绝对路径
3. 如果出现中文字体显示问题，可以手动指定中文字体路径
4. 对于大型演示文稿，转换过程可能需要一些时间
5. 临时文件默认会被删除，如果需要调试可以设置`delete_temp=False`

## 故障排除

1. **PowerPoint无法打开文件**: 检查文件路径是否正确，文件是否被其他程序占用
2. **中文字体显示乱码**: 手动指定中文字体路径，确保系统中已安装该字体
3. **转换过程卡住**: 尝试以管理员身份运行脚本，或检查PowerPoint是否正常工作
4. **临时文件未清理**: 手动删除临时目录，通常位于系统的临时文件夹中

## 改进记录

1. 添加了完善的错误处理和日志记录
2. 改进了中文字体支持
3. 实现了自动换行功能
4. 增加了批量转换功能
5. 优化了临时文件管理
6. 修复了备注提取逻辑
7. 提供了更灵活的参数配置