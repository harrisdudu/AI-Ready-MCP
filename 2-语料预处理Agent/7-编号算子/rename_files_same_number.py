import os
import re

# 定义目标文件夹路径
folder_path = r"g:\总规五年期档案数据-0930\2a725c4698024181ba9bc4cbfe35ed53"

# 指定此文件夹使用的固定编号
folder_number = 50000

# 获取文件夹中的所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 遍历所有文件并重命名
for file_name in files:
    # 构建完整的文件路径
    old_file_path = os.path.join(folder_path, file_name)
    
    # 检查文件名是否已经包含编号前缀，如果有则移除
    # 正则表达式匹配：数字+下划线开头的部分
    match = re.match(r'^(\d+)_(.+)', file_name)
    if match:
        # 如果文件名已经有编号前缀，则使用原始文件名部分
        original_name_part = match.group(2)
    else:
        # 如果文件名没有编号前缀，则使用整个文件名
        original_name_part = file_name
    
    # 构建新文件名，格式为：固定编号_原始文件名
    new_file_name = f"{folder_number}_{original_name_part}"
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 执行重命名操作
    os.rename(old_file_path, new_file_path)
    
    # 输出重命名信息
    print(f"已重命名: {file_name} -> {new_file_name}")

# 输出完成信息
print(f"\n文件夹 {os.path.basename(folder_path)} 中的文件已全部修改为相同编号 {folder_number}。")
print(f"共处理 {len(files)} 个文件。")