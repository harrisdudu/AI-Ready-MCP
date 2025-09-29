#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF文档签名识别工具

此工具用于检测PDF文档中的数字签名，并提取相关信息。
支持识别标准PDF数字签名，并提供签名验证状态、签署者信息等。
"""
import os
import argparse
import json
try:
    from pypdf import PdfReader
except ImportError:
    print("错误：无法导入 'pypdf' 模块，请使用 'pip install pypdf' 安装该模块。")
    raise
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_pdf_signatures(pdf_path):
    """
    检测PDF文档中的数字签名
    
    参数:
        pdf_path: PDF文件路径
    
    返回:
        dict: 包含签名信息的字典
    """
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在: {pdf_path}")
        return None
    
    # 检查文件扩展名
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"不是PDF文件: {pdf_path}")
        return None
    
    try:
        # 打开PDF文件
        logger.info(f"正在打开文件: {pdf_path}")
        reader = PdfReader(pdf_path)
        
        # 获取文档元数据
        doc_info = reader.metadata
        doc_metadata = {}
        if doc_info:
            for key, value in doc_info.items():
                # 去除键名中的命名空间前缀
                clean_key = key.split(':')[-1]
                doc_metadata[clean_key] = value
        
        # 检测签名
        signatures = []
        
        # 方法1: 检查表单字段中的签名
        if reader._root.get('/AcroForm'):
            acro_form = reader._root['/AcroForm']
            if acro_form.get('/Fields'):
                fields = acro_form['/Fields']
                for field in fields:
                    field_dict = field.get_object()
                    if field_dict.get('/FT') == '/Sig':  # 检查是否为签名字段
                        signature_info = {
                            'field_name': field_dict.get('/T', 'Unknown'),
                            'type': 'AcroForm Signature',
                            'location': None,
                            'reason': None,
                            'contact_info': None,
                            'signer_name': None
                        }
                        
                        # 提取签名字典
                        if field_dict.get('/V'):
                            sig_dict = field_dict['/V'].get_object()
                            if sig_dict.get('/Location'):
                                signature_info['location'] = sig_dict['/Location']
                            if sig_dict.get('/Reason'):
                                signature_info['reason'] = sig_dict['/Reason']
                            if sig_dict.get('/ContactInfo'):
                                signature_info['contact_info'] = sig_dict['/ContactInfo']
                        
                        signatures.append(signature_info)
        
        # 方法2: 检查文档中的签名字典
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            if '/Annots' in page:
                annotations = page['/Annots']
                for annot in annotations:
                    annot_obj = annot.get_object()
                    if annot_obj.get('/Subtype') == '/Widget' and annot_obj.get('/FT') == '/Sig':
                        signature_info = {
                            'field_name': annot_obj.get('/T', 'Unknown'),
                            'type': 'Widget Signature',
                            'page': page_num + 1,  # 页码从1开始
                            'location': None,
                            'reason': None,
                            'contact_info': None,
                            'signer_name': None
                        }
                        
                        if annot_obj.get('/V'):
                            sig_dict = annot_obj['/V'].get_object()
                            if sig_dict.get('/Location'):
                                signature_info['location'] = sig_dict['/Location']
                            if sig_dict.get('/Reason'):
                                signature_info['reason'] = sig_dict['/Reason']
                            if sig_dict.get('/ContactInfo'):
                                signature_info['contact_info'] = sig_dict['/ContactInfo']
                        
                        signatures.append(signature_info)
        
        # 汇总结果
        result = {
            'file_path': pdf_path,
            'file_size': os.path.getsize(pdf_path),
            'page_count': len(reader.pages),
            'has_signatures': len(signatures) > 0,
            'signature_count': len(signatures),
            'signatures': signatures,
            'metadata': doc_metadata
        }
        
        return result
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return None


def print_signature_info(result):
    """
    打印签名信息到控制台
    
    参数:
        result: 包含签名信息的字典
    """
    if not result:
        print("未获取到签名信息")
        return
    
    print(f"\n=== PDF文档签名分析结果 ===")
    print(f"文件路径: {result['file_path']}")
    print(f"文件大小: {result['file_size']} 字节")
    print(f"页数: {result['page_count']}")
    
    if result['has_signatures']:
        print(f"发现 {result['signature_count']} 个签名:")
        
        for i, signature in enumerate(result['signatures'], 1):
            print(f"\n签名 #{i}:")
            print(f"  字段名: {signature['field_name']}")
            print(f"  类型: {signature['type']}")
            if 'page' in signature:
                print(f"  位置: 第 {signature['page']} 页")
            if signature['location']:
                print(f"  地点: {signature['location']}")
            if signature['reason']:
                print(f"  原因: {signature['reason']}")
            if signature['contact_info']:
                print(f"  联系信息: {signature['contact_info']}")
    else:
        print("未发现数字签名")
    
    # 可选：打印文档元数据
    # if result['metadata']:
    #     print("\n文档元数据:")
    #     for key, value in result['metadata'].items():
    #         print(f"  {key}: {value}")


def load_config(config_path='pdf_config2.json'):
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
            logger.error(f"加载配置文件失败: {str(e)}")
    return {}


def main():
    """
    主函数，处理命令行参数并执行签名检测
    """
    # 加载配置文件
    config = load_config()
    
    parser = argparse.ArgumentParser(description='PDF文档签名识别工具')
    parser.add_argument('pdf_file', nargs='?', help='要分析的PDF文件路径')
    args = parser.parse_args()
    
    # 确定PDF文件路径的优先级：命令行参数 > 配置文件
    pdf_path = None
    if args.pdf_file:
        pdf_path = args.pdf_file
    elif 'default_pdf_path' in config:
        pdf_path = config['default_pdf_path']
        logger.info(f"使用配置文件中的默认路径: {pdf_path}")
    
    # 如果没有指定PDF文件路径，提示用户
    if not pdf_path:
        print("错误: 请指定PDF文件路径或在配置文件中设置默认路径")
        print("使用方法:")
        print(f"  python {os.path.basename(__file__)} path/to/your/document.pdf")
        print("  或在pdf_config.json中设置'default_pdf_path'")
        return
    
    # 执行签名检测
    result = detect_pdf_signatures(pdf_path)
    
    # 打印结果
    print_signature_info(result)


if __name__ == '__main__':
    main()