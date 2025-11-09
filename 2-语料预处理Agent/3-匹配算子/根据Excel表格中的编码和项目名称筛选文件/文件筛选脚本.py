#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件筛选脚本
功能：根据Excel表格中的编码和项目名称筛选文件
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys

# 设置编码
if sys.version_info < (3, 0):
    reload(sys)
    sys.setdefaultencoding('utf-8')

class FileFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件筛选工具")
        self.root.geometry("600x400")
        
        # 变量
        self.source_dir = tk.StringVar()
        self.target_dir = tk.StringVar()
        self.excel_file = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="文件筛选工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 源目录选择
        ttk.Label(main_frame, text="目标目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.source_dir, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_source_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # 目标目录选择
        ttk.Label(main_frame, text="新文件夹路径:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.target_dir, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_target_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # Excel文件选择
        ttk.Label(main_frame, text="Excel表格:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.excel_file, width=50).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_excel_file).grid(row=3, column=2, padx=5, pady=5)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="开始筛选", command=self.start_filtering).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="清空", command=self.clear_all).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=10)
    
    def browse_source_dir(self):
        """选择源目录"""
        directory = filedialog.askdirectory(title="选择目标目录")
        if directory:
            self.source_dir.set(directory)
    
    def browse_target_dir(self):
        """选择目标目录"""
        directory = filedialog.askdirectory(title="选择新文件夹路径")
        if directory:
            self.target_dir.set(directory)
    
    def browse_excel_file(self):
        """选择Excel文件"""
        file_path = filedialog.askopenfilename(
            title="选择Excel表格",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if file_path:
            self.excel_file.set(file_path)
    
    def read_excel_data(self, excel_path):
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
    
    def find_matching_files(self, source_dir, match_list):
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
    
    def copy_files(self, matching_files, target_dir):
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
                
            except Exception as e:
                print(f"复制文件失败 {filename}: {str(e)}")
        
        return copied_files
    
    def start_filtering(self):
        """开始筛选过程"""
        # 验证输入
        if not self.source_dir.get():
            messagebox.showerror("错误", "请选择目标目录")
            return
        
        if not self.target_dir.get():
            messagebox.showerror("错误", "请选择新文件夹路径")
            return
        
        if not self.excel_file.get():
            messagebox.showerror("错误", "请选择Excel表格")
            return
        
        # 开始进度条
        self.progress.start()
        self.status_label.config(text="正在读取Excel表格...")
        self.root.update()
        
        try:
            # 读取Excel数据
            match_list = self.read_excel_data(self.excel_file.get())
            self.status_label.config(text=f"找到 {len(match_list)} 个匹配项，正在搜索文件...")
            self.root.update()
            
            # 查找匹配的文件
            matching_files = self.find_matching_files(self.source_dir.get(), match_list)
            self.status_label.config(text=f"找到 {len(matching_files)} 个匹配文件，正在复制...")
            self.root.update()
            
            # 复制文件
            copied_files = self.copy_files(matching_files, self.target_dir.get())
            
            # 停止进度条
            self.progress.stop()
            
            # 显示结果
            messagebox.showinfo("完成", f"筛选完成！\n成功复制 {len(copied_files)} 个文件到:\n{self.target_dir.get()}")
            self.status_label.config(text=f"完成！复制了 {len(copied_files)} 个文件")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("错误", f"筛选过程中出现错误:\n{str(e)}")
            self.status_label.config(text="筛选失败")
    
    def clear_all(self):
        """清空所有输入"""
        self.source_dir.set("")
        self.target_dir.set("")
        self.excel_file.set("")
        self.status_label.config(text="")

def main():
    """主函数"""
    root = tk.Tk()
    app = FileFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()