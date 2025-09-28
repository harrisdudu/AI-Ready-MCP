#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件筛选复制工具

此工具用于从指定文件路径列表中筛选并复制文件到新的目标目录。
支持从完整文件路径列表中提取文件名，并保持目录结构或直接复制到目标目录。
"""
import os
import shutil
import argparse
import json
import logging
from pathlib import Path
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def load_file_paths(file_list_path):
    """
    从文件加载完整的文件路径列表
    
    参数:
        file_list_path: 包含文件路径的列表文件
    
    返回:
        list: 完整的文件路径列表
    """
    file_paths = []
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    file_paths.append(line)
        logger.info(f"已加载 {len(file_paths)} 个文件路径")
    except Exception as e:
        logger.error(f"加载文件路径列表失败: {str(e)}")
    return file_paths


def copy_files_from_paths(file_paths, target_dir, preserve_structure=False, chunk_size=100):
    """
    根据文件路径列表复制文件到目标目录
    
    参数:
        file_paths: 文件路径列表
        target_dir: 目标目录路径
        preserve_structure: 是否保留原始目录结构
        chunk_size: 每批处理的文件数量
    
    返回:
        tuple: (成功复制的文件数, 失败的文件数, 跳过的文件数)
    """
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    total_files = len(file_paths)
    logger.info(f"开始复制任务，共 {total_files} 个文件，目标目录: {target_dir}")
    start_time = time.time()
    
    # 分批处理文件以避免内存问题
    for i in range(0, total_files, chunk_size):
        chunk = file_paths[i:i+chunk_size]
        chunk_start_time = time.time()
        
        for file_path in chunk:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                skip_count += 1
                continue
            
            # 获取文件名
            file_name = os.path.basename(file_path)
            
            # 构建目标文件路径
            if preserve_structure:
                # 提取相对路径（从根目录的下一级开始）
                if ':' in file_path:  # Windows路径
                    drive, path = os.path.splitdrive(file_path)
                    path_parts = path.split(os.sep)
                    # 排除空字符串和根目录
                    valid_parts = [part for part in path_parts if part]
                    if len(valid_parts) > 1:
                        # 使用除了最后一个部分（文件名）之外的所有部分作为相对路径
                        relative_path = os.sep.join(valid_parts[:-1])
                        target_file_dir = os.path.join(target_dir, relative_path)
                        os.makedirs(target_file_dir, exist_ok=True)
                        target_file_path = os.path.join(target_file_dir, file_name)
                    else:
                        target_file_path = os.path.join(target_dir, file_name)
            else:
                target_file_path = os.path.join(target_dir, file_name)
            
            # 避免文件名冲突
            if os.path.exists(target_file_path):
                base_name, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(os.path.join(os.path.dirname(target_file_path), f"{base_name}_{counter}{ext}")):
                    counter += 1
                target_file_path = os.path.join(os.path.dirname(target_file_path), f"{base_name}_{counter}{ext}")
                logger.warning(f"文件名冲突，已重命名为: {os.path.basename(target_file_path)}")
            
            try:
                # 复制文件
                shutil.copy2(file_path, target_file_path)
                success_count += 1
                
                # 每成功复制10个文件打印一次进度
                if success_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    files_per_second = success_count / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"进度: {success_count}/{total_files} 已复制，速度: {files_per_second:.2f} 文件/秒")
            except Exception as e:
                logger.error(f"复制文件失败 '{file_path}': {str(e)}")
                fail_count += 1
        
        chunk_elapsed_time = time.time() - chunk_start_time
        chunk_files_per_second = len(chunk) / chunk_elapsed_time if chunk_elapsed_time > 0 else 0
        logger.info(f"批次 {i//chunk_size + 1} 完成: {len(chunk)} 个文件，耗时: {chunk_elapsed_time:.2f} 秒，速度: {chunk_files_per_second:.2f} 文件/秒")
    
    total_elapsed_time = time.time() - start_time
    logger.info(f"复制任务完成，总耗时: {total_elapsed_time:.2f} 秒")
    
    return success_count, fail_count, skip_count


def filter_files_by_extension(file_paths, extensions=None):
    """
    根据文件扩展名筛选文件
    
    参数:
        file_paths: 文件路径列表
        extensions: 要筛选的文件扩展名列表（不包含点号），默认为None（不过滤）
    
    返回:
        list: 筛选后的文件路径列表
    """
    if not extensions:
        return file_paths
    
    filtered_paths = []
    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        if ext.lower().lstrip('.') in [e.lower() for e in extensions]:
            filtered_paths.append(file_path)
    
    logger.info(f"根据扩展名筛选后，剩余 {len(filtered_paths)} 个文件")
    return filtered_paths


def create_example_file_list(output_path='file_list_example.txt'):
    """
    创建示例文件列表
    
    参数:
        output_path: 输出文件路径
    """
    example_content = """# 文件路径列表示例
# 这是一个示例文件路径列表，用于演示file_selector_copier.py工具
# 每行一个完整的文件路径
# 以#开头的行被视为注释

# 示例PDF文件路径
E:\示例目录\file1.pdf
E:\示例目录\file2.pdf
D:\另一个目录\file3.pdf
"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(example_content)
        logger.info(f"已创建示例文件列表: {output_path}")
    except Exception as e:
        logger.error(f"创建示例文件列表失败: {str(e)}")


def main():
    """
    主函数，处理命令行参数并执行文件筛选和复制
    """
    # 加载配置文件
    config = load_config()
    
    parser = argparse.ArgumentParser(description='文件筛选复制工具')
    parser.add_argument('--file-list', help='包含文件路径的列表文件（如scanPDFs.txt）')
    parser.add_argument('--target-dir', help='目标目录路径')
    parser.add_argument('--preserve-structure', action='store_true', help='是否保留原始目录结构')
    parser.add_argument('--extensions', nargs='+', help='要筛选的文件扩展名（不包含点号），如 pdf docx')
    parser.add_argument('--create-example', action='store_true', help='创建示例文件列表')
    parser.add_argument('--chunk-size', type=int, default=100, help='每批处理的文件数量')
    
    args = parser.parse_args()
    
    # 如果请求创建示例文件列表
    if args.create_example:
        create_example_file_list()
        return
    
    # 确定参数的优先级：命令行参数 > 配置文件
    file_list_path = args.file_list or config.get('file_selector_file_list')
    target_dir = args.target_dir or config.get('file_selector_target_dir')
    preserve_structure = args.preserve_structure or config.get('file_selector_preserve_structure', False)
    extensions = args.extensions or config.get('file_selector_extensions')
    chunk_size = args.chunk_size
    
    # 检查必要参数
    if not file_list_path:
        print("错误: 请指定包含文件路径的列表文件")
        print("使用方法:")
        print(f"  python {os.path.basename(__file__)} --file-list scanPDFs.txt --target-dir /path/to/target")
        print(f"  python {os.path.basename(__file__)} --create-example  # 创建示例文件列表")
        print("  或在pdf_config2.json中设置'file_selector_file_list'和'file_selector_target_dir'")
        return
    
    if not target_dir:
        print("错误: 请指定目标目录")
        return
    
    # 加载文件路径列表
    file_paths = load_file_paths(file_list_path)
    if not file_paths:
        print("错误: 文件路径列表为空")
        return
    
    # 根据扩展名筛选文件
    if extensions:
        file_paths = filter_files_by_extension(file_paths, extensions)
        if not file_paths:
            print(f"错误: 没有找到匹配扩展名 {extensions} 的文件")
            return
    
    print(f"\n=== 文件筛选复制任务信息 ===")
    print(f"文件路径列表: {file_list_path}")
    print(f"目标目录: {target_dir}")
    print(f"文件数量: {len(file_paths)}")
    print(f"保留目录结构: {preserve_structure}")
    if extensions:
        print(f"筛选扩展名: {extensions}")
    print(f"批处理大小: {chunk_size}")
    
    # 执行文件复制
    success_count, fail_count, skip_count = copy_files_from_paths(
        file_paths, target_dir, preserve_structure, chunk_size
    )
    
    # 打印结果摘要
    print(f"\n=== 文件筛选复制结果摘要 ===")
    print(f"总文件数: {len(file_paths)}")
    print(f"成功复制: {success_count}")
    print(f"复制失败: {fail_count}")
    print(f"跳过(文件不存在): {skip_count}")
    
    if success_count > 0:
        print(f"\n文件已成功复制到: {target_dir}")


if __name__ == '__main__':
    main()