import os

# 定义目标文件夹路径
folder_path = r"g:\总规五年期档案数据-0930\2a725c4698024181ba9bc4cbfe35ed53"

# 初始化计数器，从50000开始
counter = 50000

# 获取文件夹中的所有文件
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 遍历所有文件并重命名
for file_name in files:
    # 构建完整的文件路径
    old_file_path = os.path.join(folder_path, file_name)
    
    # 获取文件名和扩展名
    name_without_ext, ext = os.path.splitext(file_name)
    
    # 构建新文件名
    new_file_name = f"{counter}_{name_without_ext}{ext}"
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 执行重命名操作
    os.rename(old_file_path, new_file_path)
    
    # 输出重命名信息
    print(f"已重命名: {file_name} -> {new_file_name}")
    
    # 增加计数器
    counter += 1

# 输出完成信息
print(f"所有文件重命名完成！共处理 {len(files)} 个文件。")