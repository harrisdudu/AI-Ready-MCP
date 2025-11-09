#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel文件转换为scanPDFs.txt格式的文本列表转换器
功能：读取沪派江南精选知识库-265项.xlsx文件，提取文件名并转换为scanPDFs.txt格式
"""

import pandas as pd
import os

def convert_excel_to_txt():
    """
    将Excel文件转换为scanPDFs.txt格式的文本列表
    
    Returns:
        bool: 转换是否成功
    """
    try:
        # 文件路径
        excel_file = "沪派江南精选知识库-265项.xlsx"
        output_file = "scanPDFs.txt"
        
        # 检查Excel文件是否存在
        if not os.path.exists(excel_file):
            print(f"❌ Excel文件不存在: {excel_file}")
            return False
        
        # 读取Excel文件
        print("📊 正在读取Excel文件...")
        df = pd.read_excel(excel_file)
        
        # 显示数据基本信息
        print(f"📋 数据形状: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")
        
        # 查找文件名列
        filename_column = None
        for col in df.columns:
            if '文件名' in str(col) or '文件' in str(col) or 'name' in str(col).lower():
                filename_column = col
                break
        
        if filename_column is None:
            print("❌ 未找到文件名列，尝试使用第一列")
            filename_column = df.columns[0]
        
        print(f"📝 使用列: {filename_column}")
        
        # 提取文件名
        filenames = df[filename_column].dropna().astype(str).tolist()
        
        print(f"📄 找到 {len(filenames)} 个文件名")
        
        # 转换为scanPDFs.txt格式
        # 格式示例: "E:\规资-726数据（1-4批+采购+人工）\0620文件\文件名.pdf"
        pdf_paths = []
        for filename in filenames:
            # 清理文件名，移除特殊字符
            clean_filename = filename.strip()
            # 构建PDF路径格式
            pdf_path = f"E:\\规资-726数据（1-4批+采购+人工）\\0620文件\\{clean_filename}.pdf"
            pdf_paths.append(pdf_path)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for path in pdf_paths:
                f.write(path + '\n')
        
        print(f"✅ 转换完成！共生成 {len(pdf_paths)} 个PDF路径")
        print(f"📁 输出文件: {output_file}")
        
        # 显示前10个生成的路径作为示例
        print("\n📋 前10个生成的PDF路径示例:")
        for i, path in enumerate(pdf_paths[:10]):
            print(f"{i+1}. {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 Excel文件转换为scanPDFs.txt格式转换器")
    print("=" * 60)
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 工作目录: {script_dir}")
    
    # 执行转换
    success = convert_excel_to_txt()
    
    if success:
        print("\n🎉 转换成功完成！")
    else:
        print("\n💥 转换失败！")

if __name__ == "__main__":
    main()