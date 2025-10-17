import os
import re

"""
批量为文件夹中的文件进行顺序编号
每个文件夹使用一个唯一的编号，从50000开始
文件夹内的所有文件使用相同的编号前缀
"""

def get_next_number(start_number, processed_folders):
    """
    获取下一个可用的编号
    检查已经处理过的文件夹编号，确保不重复
    """
    current_number = start_number
    while current_number in processed_folders:
        current_number += 1
    return current_number

def get_processed_folder_numbers(root_dir):
    """
    获取已经处理过的文件夹编号
    通过检查文件夹内文件的编号前缀来确定
    """
    processed_numbers = set()
    # 遍历所有文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # 检查文件夹内的文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # 检查文件名是否符合编号格式 (数字_文件名.扩展名)
                match = re.match(r'(\d+)_', filename)
                if match:
                    number = int(match.group(1))
                    processed_numbers.add(number)
                    break  # 找到一个编号就可以确定该文件夹的编号了
    return processed_numbers

def batch_rename_files(root_dir, start_number=50000):
    """
    批量重命名文件夹中的文件
    每个文件夹使用一个唯一的编号
    """
    # 获取已经处理过的文件夹编号
    processed_folders = get_processed_folder_numbers(root_dir)
    print(f"已处理的文件夹编号: {processed_folders}")
    
    # 获取所有子文件夹
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    print(f"找到 {len(folders)} 个文件夹")
    
    # 跟踪处理的文件夹和文件数量
    processed_folder_count = 0
    total_file_count = 0
    
    # 为每个文件夹分配编号并重命名文件
    for folder_name in folders:
        folder_path = os.path.join(root_dir, folder_name)
        
        # 检查文件夹是否为空
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not files:
            continue
        
        # 检查文件夹内是否已有编号文件
        has_numbered_files = False
        for filename in files:
            if re.match(r'\d+_', filename):
                has_numbered_files = True
                break
        
        # 如果已有编号文件，则跳过该文件夹
        if has_numbered_files:
            print(f"跳过已编号的文件夹: {folder_name}")
            continue
        
        # 获取下一个可用的编号
        current_number = get_next_number(start_number, processed_folders)
        processed_folders.add(current_number)
        
        # 重命名文件夹内的所有文件
        file_count = 0
        for filename in files:
            old_path = os.path.join(folder_path, filename)
            new_filename = f"{current_number}_{filename}"
            new_path = os.path.join(folder_path, new_filename)
            
            try:
                os.rename(old_path, new_path)
                file_count += 1
                total_file_count += 1
            except Exception as e:
                print(f"重命名文件 {filename} 失败: {e}")
        
        if file_count > 0:
            processed_folder_count += 1
            print(f"文件夹 {folder_name} 处理完成，编号: {current_number}，重命名了 {file_count} 个文件")
    
    print(f"\n批量处理完成！")
    print(f"成功处理了 {processed_folder_count} 个文件夹")
    print(f"总共重命名了 {total_file_count} 个文件")

if __name__ == "__main__":
    root_directory = "g:\\总规五年期档案数据-0930"
    print(f"开始批量重命名文件...")
    print(f"根目录: {root_directory}")
    batch_rename_files(root_directory, start_number=50000)