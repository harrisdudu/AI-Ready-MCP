#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件匹配与复制工具

功能：
1. 从Excel文件中提取序号
2. 遍历指定文件夹及其子文件夹
3. 匹配文件名中的序号
4. 将匹配文件复制到目标文件夹
5. 实现基于文件名和大小的去重
6. 记录操作日志
"""

import os
import shutil
import re
import pandas as pd
import logging
from datetime import datetime
import traceback


def setup_logger():
    """
    配置日志记录器
    返回: 配置好的logger对象
    """
    # 创建日志文件名
    log_filename = f"file_matcher_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(os.getcwd(), log_filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_path}")
    return logger


def extract_serial_numbers(excel_path, logger):
    """
    从Excel文件提取序号列中的阿拉伯数字
    参数:
        excel_path: Excel文件路径
        logger: 日志记录器
    返回:
        set: 提取的序号集合
    """
    try:
        logger.info(f"正在读取Excel文件: {excel_path}")
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 获取第二列数据（索引为1）
        second_column = df.iloc[:, 1]
        
        # 提取阿拉伯数字（忽略列标题）
        serial_numbers = set()
        for value in second_column[1:]:  # 跳过第一行（标题行）
            # 检查值是否为数字
            if pd.notna(value):
                # 如果是数字，直接添加
                if isinstance(value, (int, float)) and value.is_integer():
                    serial_numbers.add(int(value))
                # 如果是字符串，尝试提取数字
                elif isinstance(value, str):
                    match = re.search(r'\d+', value)
                    if match:
                        serial_numbers.add(int(match.group()))
        
        logger.info(f"成功提取{len(serial_numbers)}个序号")
        return serial_numbers
    
    except Exception as e:
        logger.error(f"读取Excel文件失败: {str(e)}")
        logger.debug(traceback.format_exc())
        return set()


def find_matching_files(source_dir, serial_numbers, logger):
    """
    查找匹配序号的文件
    参数:
        source_dir: 源文件夹路径
        serial_numbers: 序号集合
        logger: 日志记录器
    返回:
        list: 匹配的文件路径列表
    """
    matching_files = []
    
    try:
        logger.info(f"开始遍历文件夹: {source_dir}")
        
        # 遍历文件夹及其子文件夹
        for root, _, files in os.walk(source_dir):
            for filename in files:
                # 使用正则表达式匹配文件名格式: "序号-名称" 或 "序号 - 名称"
                match = re.match(r'(\d+)\s*-\s*', filename)
                if match:
                    file_serial = int(match.group(1))
                    # 检查序号是否在提取的序号集合中
                    if file_serial in serial_numbers:
                        file_path = os.path.join(root, filename)
                        matching_files.append(file_path)
                        logger.info(f"找到匹配文件: {file_path}")
        
        logger.info(f"共找到{len(matching_files)}个匹配文件")
        return matching_files
    
    except Exception as e:
        logger.error(f"遍历文件夹失败: {str(e)}")
        logger.debug(traceback.format_exc())
        return []


def copy_files_with_deduplication(matching_files, target_dir, logger):
    """
    复制文件到目标文件夹并去重
    参数:
        matching_files: 匹配的文件路径列表
        target_dir: 目标文件夹路径
        logger: 日志记录器
    返回:
        tuple: (成功复制数, 去重文件数, 失败文件数)
    """
    success_count = 0
    duplicate_count = 0
    failed_count = 0
    
    # 用于记录已复制文件的信息，格式: {文件名: 文件大小}
    copied_files_info = {}
    
    try:
        # 确保目标文件夹存在
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info(f"创建目标文件夹: {target_dir}")
        
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            
            # 获取文件大小
            try:
                file_size = os.path.getsize(file_path)
            except Exception as e:
                logger.error(f"获取文件大小失败: {file_path}, 错误: {str(e)}")
                failed_count += 1
                continue
            
            # 检查是否重复
            if filename in copied_files_info:
                if copied_files_info[filename] == file_size:
                    logger.info(f"发现重复文件，跳过: {filename}")
                    duplicate_count += 1
                    continue
            
            # 复制文件
            try:
                shutil.copy2(file_path, target_path)
                copied_files_info[filename] = file_size
                success_count += 1
                logger.info(f"成功复制: {file_path} -> {target_path}")
            except Exception as e:
                logger.error(f"复制文件失败: {file_path}, 错误: {str(e)}")
                logger.debug(traceback.format_exc())
                failed_count += 1
    
    except Exception as e:
        logger.error(f"文件复制过程中出错: {str(e)}")
        logger.debug(traceback.format_exc())
    
    return success_count, duplicate_count, failed_count


def main():
    """
    主函数
    """
    # 配置日志
    logger = setup_logger()
    
    # 配置路径
    excel_path = r"G:\沪派江南精选知识库-265项.xlsx"
    source_dir = r"G:\1-原料仓库"
    target_dir = r"G:\265项"
    
    logger.info("===== 文件匹配与复制工具开始运行 =====")
    
    # 1. 从Excel提取序号
    serial_numbers = extract_serial_numbers(excel_path, logger)
    
    if not serial_numbers:
        logger.error("未能从Excel文件中提取有效序号，程序终止")
        return
    
    # 2. 查找匹配文件
    matching_files = find_matching_files(source_dir, serial_numbers, logger)
    
    if not matching_files:
        logger.warning("未找到匹配的文件")
    else:
        # 3. 复制文件并去重
        success_count, duplicate_count, failed_count = copy_files_with_deduplication(
            matching_files, target_dir, logger
        )
        
        # 4. 输出统计信息
        logger.info("===== 操作完成 =====")
        logger.info(f"成功复制文件数: {success_count}")
        logger.info(f"去重文件数: {duplicate_count}")
        logger.info(f"复制失败文件数: {failed_count}")
        logger.info(f"目标文件夹: {target_dir}")
    
    logger.info("===== 文件匹配与复制工具运行结束 =====")


if __name__ == "__main__":
    main()