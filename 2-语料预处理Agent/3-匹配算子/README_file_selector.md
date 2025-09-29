# 文件筛选复制工具使用说明

## 工具介绍 📋

这个工具专门设计用于从完整的文件路径列表中筛选并复制文件到指定的目标目录。特别适合处理像`scanPDFs.txt`这样包含大量完整文件路径的清单文件。

## 功能特点 ✨

- 📂 从完整文件路径列表中批量复制文件
- 🎯 支持根据文件扩展名进行筛选（如只复制PDF文件）
- 🗂️ 可选保留原始目录结构或扁平化复制
- ⚡ 分批处理大文件列表，避免内存问题
- 📊 提供详细的复制进度和统计信息
- ⚙️ 支持通过命令行参数或配置文件设置选项

## 安装依赖 🚀

确保您已安装Python 3.6或更高版本，然后安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 快速开始 🏃

### 使用方式1：通过命令行参数

```bash
python file_selector_copier.py --file-list scanPDFs.txt --target-dir d:\target_folder
```

### 使用方式2：通过配置文件

编辑`pdf_config2.json`文件，设置默认参数：

```json
{
    "default_pdf_path": "path/to/your/default/document.pdf",
    "file_selector_file_list": "scanPDFs.txt",
    "file_selector_target_dir": "E:\\规资-726数据（1-4批+采购+人工）\\数字签名",
    "file_selector_preserve_structure": false,
    "file_selector_extensions": ["pdf"],
    "file_selector_chunk_size": 100
}
```

然后直接运行：

```bash
python file_selector_copier.py
```

## 命令行参数详解 📝

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--file-list` | 包含文件路径的列表文件（如scanPDFs.txt） | 配置文件中的值 |
| `--target-dir` | 目标目录路径 | 配置文件中的值 |
| `--preserve-structure` | 是否保留原始目录结构 | 配置文件中的值（默认false） |
| `--extensions` | 要筛选的文件扩展名（不包含点号） | 配置文件中的值（默认["pdf"]） |
| `--chunk-size` | 每批处理的文件数量 | 100 |
| `--create-example` | 创建示例文件列表 | 无 |

## 文件列表格式要求 📄

文件列表（如`scanPDFs.txt`）应遵循以下格式：

- 每行一个完整的文件路径
- 支持Windows路径格式（如`E:\规资-726数据（1-4批+采购+人工）\0620文件\10964-10964文书类电子档案检测一般要求.pdf`）
- 以`#`开头的行将被视为注释并忽略

示例：

```
# 这是一个注释行
E:\文件夹1\file1.pdf
E:\文件夹2\file2.pdf
D:\文件夹3\file3.pdf
```

## 配置文件详解 ⚙️

`pdf_config2.json`文件支持以下配置项：

- `file_selector_file_list`: 默认的文件路径列表文件
- `file_selector_target_dir`: 默认的目标目录路径
- `file_selector_preserve_structure`: 是否默认保留目录结构
- `file_selector_extensions`: 默认筛选的文件扩展名列表
- `file_selector_chunk_size`: 默认的批处理大小

## 高级功能 💡

### 1. 保留目录结构

如果您希望在复制文件时保留原始的目录结构，可以使用`--preserve-structure`参数：

```bash
python file_selector_copier.py --file-list scanPDFs.txt --target-dir d:\target_folder --preserve-structure
```

这将在目标目录下重建与源文件相对应的目录结构。

### 2. 筛选特定扩展名的文件

如果您只想复制特定类型的文件，可以使用`--extensions`参数：

```bash
python file_selector_copier.py --file-list scanPDFs.txt --target-dir d:\target_folder --extensions pdf docx xlsx
```

### 3. 处理大量文件

对于包含大量文件路径的列表，您可以调整`--chunk-size`参数来优化性能：

```bash
python file_selector_copier.py --file-list scanPDFs.txt --target-dir d:\target_folder --chunk-size 200
```

## 常见问题解答 ❓

**问题1：为什么有些文件显示"跳过(文件不存在)"？**

这意味着工具无法在指定的路径上找到文件。请检查文件路径是否正确，以及您是否有权限访问这些文件。

**问题2：复制速度很慢，如何优化？**

- 增加`--chunk-size`参数的值
- 确保源文件和目标目录在同一物理驱动器上
- 关闭`--preserve-structure`选项，如果不需要保留目录结构

**问题3：文件名冲突怎么办？**

工具会自动在冲突的文件名后添加数字后缀（如`file_1.pdf`、`file_2.pdf`）来避免覆盖现有文件。

## 示例输出 📊

```
=== 文件筛选复制任务信息 ===
文件路径列表: scanPDFs.txt
目标目录: d:\target_folder
文件数量: 1917
保留目录结构: False
筛选扩展名: ['pdf']
批处理大小: 100

进度: 10/1917 已复制，速度: 2.50 文件/秒
进度: 20/1917 已复制，速度: 2.48 文件/秒
...
批次 1 完成: 100 个文件，耗时: 40.25 秒，速度: 2.49 文件/秒
批次 2 完成: 100 个文件，耗时: 39.87 秒，速度: 2.51 文件/秒
...
复制任务完成，总耗时: 765.32 秒

=== 文件筛选复制结果摘要 ===
总文件数: 1917
成功复制: 1895
复制失败: 12
跳过(文件不存在): 10

文件已成功复制到: d:\target_folder
```

## 注意事项 ⚠️

- 请确保您有足够的磁盘空间用于复制文件
- 复制大量文件可能需要较长时间，请耐心等待
- 对于网络驱动器上的文件，复制速度可能会受到网络带宽的限制
- 处理包含特殊字符的文件路径时，请确保使用正确的编码

## 版本信息 📌

- 版本: 1.0.0
- 最后更新: 2023-10-25

## 反馈与支持 💬

如有任何问题或建议，请联系工具维护人员。