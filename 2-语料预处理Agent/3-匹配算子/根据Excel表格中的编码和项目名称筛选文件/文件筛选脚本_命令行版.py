#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件筛选脚本 - 命令行版本
功能：根据Excel表格中的编码和项目名称筛选文件
"""

import os
import shutil
import pandas as pd
import argparse
from pathlib import Path

def read_excel_data(excel_path):
    """读取Excel表格数据"""
    try:
        # 尝试读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 检查必要的列是否存在
        required_columns = []
        for col in ['编码', '项目名称', '编号', '名称']:
            if col in df.columns:
                required_columns.append(col)
        
        if len(required_columns) < 2:
            raise ValueError("Excel表格中未找到合适的编码和名称列")
        
        # 获取编码列和名称列
        code_col = required_columns[0]
        name_col = required_columns[1] if len(required_columns) > 1 else required_columns[0]
        
        # 创建匹配列表
        match_list = []
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            
            if code and code != 'nan':
                match_list.append({
                    'code': code,
                    'name': name if name and name != 'nan' else ''
                })
        
        return match_list
        
    except Exception as e:
        raise Exception(f"读取Excel文件失败: {str(e)}")

def find_matching_files(source_dir, match_list):
    """在源目录中查找匹配的文件"""
    matching_files = []
    
    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            filename = os.path.splitext(file)[0]  # 去掉扩展名的文件名
            
            # 检查文件名是否包含任何编码或项目名称
            for item in match_list:
                if item['code'] in filename or (item['name'] and item['name'] in filename):
                    matching_files.append({
                        'source_path': file_path,
                        'filename': file,
                        'matched_code': item['code'],
                        'matched_name': item['name']
                    })
                    break  # 找到一个匹配就停止检查当前文件
    
    return matching_files

def copy_files(matching_files, target_dir):
    """复制匹配的文件到目标目录"""
    copied_files = []
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    for file_info in matching_files:
        source_path = file_info['source_path']
        filename = file_info['filename']
        target_path = os.path.join(target_dir, filename)
        
        try:
            # 如果目标文件已存在，添加序号避免覆盖
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(target_path):
                target_path = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            copied_files.append({
                'source': source_path,
                'target': target_path,
                'filename': filename
            })
            print(f"✓ 已复制: {filename}")
            
        except Exception as e:
            print(f"✗ 复制文件失败 {filename}: {str(e)}")
    
    return copied_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文件筛选工具 - 根据Excel表格筛选文件')
    parser.add_argument('--source', '-s', required=True, help='目标目录路径')
    parser.add_argument('--target', '-t', required=True, help='新文件夹路径')
    parser.add_argument('--excel', '-e', required=True, help='Excel表格路径')
    
    args = parser.parse_args()
    
    print("=== 文件筛选工具 ===")
    print(f"目标目录: {args.source}")
    print(f"新文件夹: {args.target}")
    print(f"Excel表格: {args.excel}")
    print("-" * 50)
    
    try:
        # 验证路径
        if not os.path.exists(args.source):
            raise Exception(f"目标目录不存在: {args.source}")
        
        if not os.path.exists(args.excel):
            raise Exception(f"Excel文件不存在: {args.excel}")
        
        # 读取Excel数据
        print("正在读取Excel表格...")
        match_list = read_excel_data(args.excel)
        print(f"找到 {len(match_list)} 个匹配项")
        
        # 查找匹配的文件
        print("正在搜索匹配的文件...")
        matching_files = find_matching_files(args.source, match_list)
        print(f"找到 {len(matching_files)} 个匹配文件")
        
        if not matching_files:
            print("未找到匹配的文件")
            return
        
        # 复制文件
        print("正在复制文件...")
        copied_files = copy_files(matching_files, args.target)
        
        print("-" * 50)
        print(f"✓ 筛选完成！")
        print(f"✓ 成功复制 {len(copied_files)} 个文件到: {args.target}")
        
    except Exception as e:
        print(f"✗ 错误: {str(e)}")

if __name__ == "__main__":
    main()