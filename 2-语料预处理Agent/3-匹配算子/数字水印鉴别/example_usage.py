#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF文档签名识别工具使用示例

此脚本展示了如何在Python代码中直接使用pdf_signature_detector模块
来检测PDF文档中的签名内容。
"""
import os
import json
import argparse
from  import detect_pdf_signatures, print_signature_info



def process_single_document(pdf_path):
    """
    处理单个PDF文档，检测其中的签名
    
    参数:
        pdf_path: PDF文件路径
    """
    print(f"\n处理文档: {pdf_path}")
    
    # 调用签名检测函数
    result = detect_pdf_signatures(pdf_path)
    
    # 打印签名信息
    if result:
        print_signature_info(result)
        
        # 可以根据需要进一步处理签名信息
        if result['has_signatures']:
            print(f"\n文档 '{os.path.basename(pdf_path)}' 中找到 {result['signature_count']} 个签名")
            
            # 提取所有签名的字段名
            signature_names = [sig['field_name'] for sig in result['signatures']]
            print(f"签名字段名列表: {', '.join(signature_names)}")
    else:
        print("处理文档失败")


def batch_process_documents(directory_path):
    """
    批量处理目录中的所有PDF文档
    
    参数:
        directory_path: 包含PDF文件的目录路径
    """
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return
    
    # 获取目录中的所有PDF文件
    pdf_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory_path, file))
    
    if not pdf_files:
        print(f"目录 '{directory_path}' 中没有找到PDF文件")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件，开始批量处理...")
    
    # 统计结果
    total_docs = len(pdf_files)
    signed_docs = 0
    total_signatures = 0
    
    # 处理每个PDF文件
    for pdf_path in pdf_files:
        result = detect_pdf_signatures(pdf_path)
        
        if result and result['has_signatures']:
            signed_docs += 1
            total_signatures += result['signature_count']
            print(f"✓ {os.path.basename(pdf_path)} - 包含 {result['signature_count']} 个签名")
        else:
            print(f"✗ {os.path.basename(pdf_path)} - 未找到签名")
    
    # 输出统计信息
    print(f"\n=== 批量处理统计 ===")
    print(f"总文档数: {total_docs}")
    print(f"含签名文档数: {signed_docs}")
    print(f"签名总数: {total_signatures}")
    print(f"签名率: {signed_docs/total_docs*100:.1f}%")


def extract_signature_data(pdf_path):
    """
    提取PDF文档中的签名数据并返回结构化信息
    
    参数:
        pdf_path: PDF文件路径
    
    返回:
        dict: 结构化的签名数据
    """
    result = detect_pdf_signatures(pdf_path)
    
    if not result:
        return None
    
    # 构建结构化数据
    signature_data = {
        'document': {
            'name': os.path.basename(pdf_path),
            'path': pdf_path,
            'size': result['file_size'],
            'pages': result['page_count']
        },
        'has_signatures': result['has_signatures'],
        'signature_count': result['signature_count'],
        'signatures': []
    }
    
    # 添加每个签名的详细信息
    if result['has_signatures']:
        for sig in result['signatures']:
            signature_info = {
                'field_name': sig['field_name'],
                'type': sig['type']
            }
            
            # 添加可选信息
            if 'page' in sig:
                signature_info['page'] = sig['page']
            if sig['location']:
                signature_info['location'] = sig['location']
            if sig['reason']:
                signature_info['reason'] = sig['reason']
            if sig['contact_info']:
                signature_info['contact_info'] = sig['contact_info']
            
            signature_data['signatures'].append(signature_info)
    
    return signature_data


def load_config(config_path='pdf_config.json'):
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        dict: 配置信息
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
    return {}


def main():
    """
    主函数，展示工具的不同使用方式
    """
    print("PDF文档签名识别工具使用示例\n")
    
    # 加载配置文件
    config = load_config()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF文档签名识别工具使用示例')
    parser.add_argument('--pdf', help='要分析的PDF文件路径')
    parser.add_argument('--dir', help='包含PDF文件的目录路径')
    args = parser.parse_args()
    
    # 确定默认的PDF文件路径
    default_pdf = args.pdf or config.get('default_pdf_path', 'path/to/your/sample.pdf')
    default_dir = args.dir or config.get('default_pdf_dir', 'path/to/your/pdf/folder')
    
    # 示例1: 处理单个文档
    print("示例1: 处理单个文档")
    print("="*50)
    if os.path.exists(default_pdf):
        process_single_document(default_pdf)
    else:
        print(f"示例文档不存在: {default_pdf}")
        print("请将示例文档路径替换为实际存在的PDF文件路径，或在pdf_config.json中设置'default_pdf_path'")
    
    # 示例2: 批量处理文档
    print("\n示例2: 批量处理文档")
    print("="*50)
    if os.path.exists(default_dir):
        batch_process_documents(default_dir)
    else:
        print(f"示例目录不存在: {default_dir}")
        print("请将示例目录路径替换为实际存在的目录路径，或在pdf_config.json中设置'default_pdf_dir'")
    
    # 示例3: 提取结构化签名数据
    print("\n示例3: 提取结构化签名数据")
    print("="*50)
    if os.path.exists(default_pdf):
        data = extract_signature_data(default_pdf)
        if data:
            print("成功提取签名数据:")
            print(f"文档名称: {data['document']['name']}")
            print(f"签名数量: {data['signature_count']}")
    

if __name__ == '__main__':
    main()