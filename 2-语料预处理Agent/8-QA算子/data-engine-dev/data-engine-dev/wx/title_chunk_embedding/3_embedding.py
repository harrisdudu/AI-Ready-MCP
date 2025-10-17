#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理文件夹，生成embedding JSON文件
nohup python3 3_embedding.py --input_path /mnt/data/wx/xiaofang/out_xiaofang_title_09011402 --output_path /mnt/data/wx/xiaofang/embedding_0902_2 > embedding_0902_2.log 2>&1 &
"""

import os
import json
import argparse
import re
import requests
from pathlib import Path


def get_embedding(text: str, api_urls: list, model_name: str, url_index: int = 0) -> list:
    """获取单个文本的embedding，支持轮询多个API端点"""
    if not text.strip():
        return ""
    
    # 轮询尝试所有API端点
    for attempt in range(len(api_urls)):
        current_url_index = (url_index + attempt) % len(api_urls)
        api_url = api_urls[current_url_index]
        
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": text,
                    "model": model_name
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                print(f"API响应格式错误: {result}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"API调用失败 (端口 {api_url.split(':')[-1]}): {e}")
            continue
    
    print(f"所有API端点都调用失败")
    return ""

def split_markdown_paragraphs(content: str) -> list:
    """智能分割markdown内容为段落"""
    # 移除多余的空白字符
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # 按双换行符分割段落
    paragraphs = []
    raw_paragraphs = content.split('\n\n')
    
    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # 移除段落开头的markdown标记（如#、*、-等）
            cleaned_paragraph = re.sub(r'^[#*\-+>\s]+', '', paragraph)
            if cleaned_paragraph:
                paragraphs.append(cleaned_paragraph)
    
    return paragraphs

def process_folder(folder_path: Path, api_urls: list, model_name: str) -> list:
    """处理单个文件夹，返回包含embedding的JSONL数据列表"""
    folder_name = folder_path.name
    
    # 检查是否有sections.jsonl文件
    sections_file = folder_path / "sections.jsonl"
    md_files = list(folder_path.glob("*.md"))
    
    if sections_file.exists() and sections_file.is_file():
        # 规则1: 有sections.jsonl文件
        print(f"处理文件夹 {folder_name}: 使用sections.jsonl内容")
        
        try:
            sections_with_embeddings = []
            contents_to_embed = []
            section_data_list = []
            
            with open(sections_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            # 解析JSONL中的每一行
                            section_data = json.loads(line)
                            
                            # 提取content字段作为文本内容
                            if isinstance(section_data, dict) and 'content' in section_data:
                                content = section_data['content'].strip()
                                # 无论content是否为空，都保留这个条目
                                contents_to_embed.append(content)
                                section_data_list.append(section_data)
                                if content:
                                    print(f"  准备处理第{line_num}行: {section_data.get('section_title', '无标题')}")
                                else:
                                    print(f"  第{line_num}行content为空，embedding将为空字符串")
                            else:
                                print(f"  第{line_num}行格式不正确（缺少content字段），跳过")
                                
                        except json.JSONDecodeError as e:
                            print(f"  第{line_num}行JSON解析错误: {e}")
                            continue
            
            # 验证数据一致性
            if len(contents_to_embed) != len(section_data_list):
                print(f"  错误：contents_to_embed长度({len(contents_to_embed)})与section_data_list长度({len(section_data_list)})不匹配")
                return None
            
            # 生成embedding
            if contents_to_embed:
                print(f"  生成 {len(contents_to_embed)} 个embedding...")
                embeddings = []
                
                for i, content in enumerate(contents_to_embed):
                    if i % 10 == 0:  # 每10个显示一次进度
                        print(f"    处理第 {i+1}/{len(contents_to_embed)} 个文本...")
                    
                    # 轮询使用不同的API端点
                    url_index = i % len(api_urls)
                    embedding = get_embedding(content, api_urls, model_name, url_index)
                    embeddings.append(embedding)
                
                # 将embedding添加到对应的section_data中
                for i, (section_data, embedding) in enumerate(zip(section_data_list, embeddings)):
                    section_data['embedding'] = embedding
                    sections_with_embeddings.append(section_data)
                            
        except Exception as e:
            print(f"读取sections.jsonl文件时出错: {e}")
            return None
        
        return sections_with_embeddings
    
    elif len(md_files) >= 1:
        # 规则2: 处理所有md文件，按段落切分
        print(f"处理文件夹 {folder_name}: 处理 {len(md_files)} 个md文件")
        
        all_sections = []
        all_paragraphs = []
        paragraph_info = []
        
        for md_file in md_files:
            try:
                print(f"  处理文件: {md_file.name}")
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read().strip()
                
                # 使用智能段落分割
                paragraphs = split_markdown_paragraphs(md_content)
                
                for i, paragraph in enumerate(paragraphs):
                    # 无论段落是否为空，都保留这个条目
                    all_paragraphs.append(paragraph)
                    paragraph_info.append({
                        "title": md_file.stem,
                        "content": paragraph
                    })
                    if paragraph:
                        print(f"    准备处理段落 {i+1}: {len(paragraph)} 字符")
                    else:
                        print(f"    段落 {i+1} 为空，embedding将为空字符串")
                
            except Exception as e:
                print(f"读取文件 {md_file} 时出错: {e}")
                continue
        
        # 验证数据一致性
        if len(all_paragraphs) != len(paragraph_info):
            print(f"  错误：all_paragraphs长度({len(all_paragraphs)})与paragraph_info长度({len(paragraph_info)})不匹配")
            return None
        
        # 生成embedding
        if all_paragraphs:
            print(f"  生成 {len(all_paragraphs)} 个embedding...")
            embeddings = []
            
            for i, paragraph in enumerate(all_paragraphs):
                if i % 10 == 0:  # 每10个显示一次进度
                    print(f"    处理第 {i+1}/{len(all_paragraphs)} 个段落...")
                
                # 轮询使用不同的API端点
                url_index = i % len(api_urls)
                embedding = get_embedding(paragraph, api_urls, model_name, url_index)
                embeddings.append(embedding)
            
            # 创建标准格式的JSON对象
            for i, (info, embedding) in enumerate(zip(paragraph_info, embeddings)):
                section_data = {
                    "title": info["title"],
                    "content": info["content"],
                    "embedding": embedding
                }
                all_sections.append(section_data)
        
        return all_sections
    
    else:
        # 不符合规则的情况
        print(f"文件夹 {folder_name} 不符合处理规则，跳过")
        return None

def main():
    parser = argparse.ArgumentParser(description='批量处理文件夹生成embedding JSON文件')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='输入文件夹路径')
    parser.add_argument('--output_path', type=str, required=True, 
                       help='输出文件夹路径')
    parser.add_argument('--api_urls', type=str, nargs='+',
                       default=["http://localhost:7100/v1/embeddings", "http://localhost:7101/v1/embeddings", 
                               "http://localhost:7102/v1/embeddings", "http://localhost:7103/v1/embeddings",
                               "http://localhost:7104/v1/embeddings", "http://localhost:7105/v1/embeddings",
                               "http://localhost:7106/v1/embeddings", "http://localhost:7107/v1/embeddings"],
                       help='Embedding API地址列表，支持多个端点轮询')
    parser.add_argument('--model_name', type=str, 
                       default="qwen3-8b-embd",
                       help='模型名称')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 检查输入路径是否存在
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}")
        return
    
    # 测试API连接
    print(f"测试API连接: {len(args.api_urls)} 个端点")
    for i, url in enumerate(args.api_urls):
        print(f"  端点 {i+1}: {url}")
    print(f"使用模型: {args.model_name}")
    
    # 测试所有API端点
    working_urls = []
    for i, url in enumerate(args.api_urls):
        try:
            test_response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": "测试",
                    "model": args.model_name
                },
                timeout=10
            )
            test_response.raise_for_status()
            print(f"  端点 {i+1} 测试成功: {url}")
            working_urls.append(url)
        except requests.exceptions.RequestException as e:
            print(f"  端点 {i+1} 测试失败: {url} - {e}")
    
    if not working_urls:
        print("所有API端点都连接失败，请确保embedding服务正在运行")
        return
    
    print(f"成功连接 {len(working_urls)} 个API端点")
    args.api_urls = working_urls  # 只使用可用的端点

    # 递归处理所有子目录
    def process_directory_recursive(current_input_path: Path, current_output_path: Path):
        """递归处理目录，保持目录结构"""
        processed_count = 0
        
        # 检查当前目录是否有sections.jsonl文件
        sections_file = current_input_path / "sections.jsonl"
        md_files = list(current_input_path.glob("*.md"))
        
        # 如果当前目录有可处理的文件，则处理它
        if sections_file.exists() or md_files:
            print(f"\n处理目录: {current_input_path}")
            
            # 确保输出目录存在
            current_output_path.mkdir(parents=True, exist_ok=True)
            
            result = process_folder(current_input_path, args.api_urls, args.model_name)
            
            if result:
                # 保存JSONL文件，每行一个JSON
                output_file = current_output_path / "sections_with_embeddings.jsonl"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for item in result:
                            # 确保每个JSON对象包含title、content、embedding字段
                            json_line = json.dumps(item, ensure_ascii=False)
                            f.write(json_line + '\n')
                    print(f"已保存: {output_file} ({len(result)} 行)")
                    processed_count += 1
                except Exception as e:
                    print(f"保存文件 {output_file} 时出错: {e}")
        
        # 递归处理所有子目录
        for item in current_input_path.iterdir():
            if item.is_dir():
                # 计算相对路径，用于在输出目录中创建相同的结构
                relative_path = item.relative_to(input_path)
                sub_output_path = output_path / relative_path
                
                # 递归处理子目录
                sub_processed = process_directory_recursive(item, sub_output_path)
                processed_count += sub_processed
        
        return processed_count
    
    # 开始递归处理
    print(f"开始递归处理目录: {input_path}")
    total_processed = process_directory_recursive(input_path, output_path)
    
    print(f"\n处理完成! 共处理了 {total_processed} 个目录")

if __name__ == "__main__":
    main()

