#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""nohup python 2_extract_questions.py /home/wangxi/workspace/huawei/source_output /home/wangxi/workspace/huawei/source """

import os
import json
import re
import sys
import argparse
import shutil
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

# Ark 配置
API_KEY = "c702676e-a69f-4ff0-a672-718d0d4723ed"
MODEL_ID = "doubao-seed-1-6-250615"

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    raise SystemExit("❌ 需要安装: pip install 'volcengine-python-sdk[ark]'")

def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志配置"""
    log_file = os.path.join(output_dir, "extract_questions.log")
    
    # 创建logger
    logger = logging.getLogger('extract_questions')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def read_md_file(file_path: str) -> str:
    """读取markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return ""

def extract_question_number(dir_name: str) -> str:
    """提取题号，返回题号"""
    # 通用匹配：任何字符-数字 或 任何字符
    # 例如：二-1, A-1, 三, B等
    # 提取"-"前面的部分作为题号
    match = re.match(r'^(.+?)(?:-\d+)?$', dir_name)
    if match:
        return match.group(1)    
    return ""

def get_original_image_paths(question_number: str, question_type: str, source_path: str, subject: str) -> List[str]:
    """根据题号、类型和主题获取source文件夹中的图片路径"""
    image_paths = []
    
    # 递归搜索source路径下的所有目录
    for root, dirs, files in os.walk(source_path):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # 检查文件名是否以题号开头（支持题号-数字的格式）
                if file_name.startswith(question_number) or f"{question_number}-" in file_name:
                    # 检查当前目录名是否匹配题目类型
                    current_dir = os.path.basename(root)
                    if current_dir == question_type:
                        # 检查路径中是否包含对应的主题
                        if subject in root:
                            # 使用相对于source_path的路径，并添加source/前缀
                            rel_path = os.path.relpath(os.path.join(root, file_name), source_path)
                            image_paths.append(f"source/{rel_path}")
    
    # 按照文件名中的序号排序
    def extract_number_from_filename(path):
        """从路径中提取文件名，然后提取序号"""
        filename = os.path.basename(path)
        # 匹配题号-数字的格式，如A-5中的5
        match = re.search(rf'{re.escape(question_number)}-(\d+)', filename)
        if match:
            return int(match.group(1))
        # 如果没有匹配到题号-数字格式，返回0
        return 0
    
    # 按照序号排序
    image_paths.sort(key=extract_number_from_filename)
    
    return image_paths

def generate_new_image_name(subject: str, question_type: str, dir_name: str, original_name: str) -> str:
    """生成新的图片文件名"""
    # 生成新文件名：subject-题目类型-题号文件夹名-原文件名
    new_name = f"{subject}-{question_type}-{dir_name}-{original_name}"
    
    # 替换不合法的文件名字符
    new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)
    
    return new_name

def update_content_image_paths(content: str, md_file_path: str, output_dir: str) -> str:
    """更新内容中的图片路径，处理同级images文件夹中的图片，并将HTML img标签转换为Markdown格式"""
    updated_content = content
    
    # 首先将HTML img标签转换为Markdown格式
    # 匹配 <img src="filename.jpg"/> 格式
    img_pattern = r'<img\s+src="([^"]+)"[^>]*/?>'
    
    def replace_img_tag(match):
        src = match.group(1)
        # 如果src是相对路径，保持原样；如果是绝对路径，只取文件名
        if '/' in src:
            filename = os.path.basename(src)
        else:
            filename = src
        return f'![图片](images/{filename})'
    
    updated_content = re.sub(img_pattern, replace_img_tag, updated_content)
    
    # 获取md文件所在的目录
    md_dir = os.path.dirname(md_file_path)
    images_dir = os.path.join(md_dir, "images")
    
    # 检查是否存在images文件夹
    if not os.path.exists(images_dir):
        return updated_content
    
    # 获取路径信息（从md文件路径向上四级）
    path_parts = md_file_path.split(os.sep)
    if len(path_parts) >= 4:
        # 上四级文件夹名字：path_parts[-4] (subject)
        # 上三级文件夹名字：path_parts[-3] (题干/答案/解析)
        # 上二级文件夹名字：path_parts[-2] (A-1, B-2等)
        fourth_level_name = path_parts[-4]  # subject
        third_level_name = path_parts[-3]   # 题干/答案/解析
        second_level_name = path_parts[-2]  # A-1, B-2等
    else:
        fourth_level_name = "unknown"
        third_level_name = "unknown"
        second_level_name = "unknown"
    
    # 创建输出目录的images文件夹
    output_images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    
    # 处理images文件夹中的图片
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            continue
        
        # 生成新的文件名：subject-题干/答案/解析-A1-原文件名
        new_image_name = f"{fourth_level_name}-{third_level_name}-{second_level_name}-{image_file}"
        # 替换不合法的文件名字符
        new_image_name = re.sub(r'[<>:"/\\|?*]', '_', new_image_name)
        
        # 源图片路径和目标路径
        old_image_path = os.path.join(images_dir, image_file)
        new_image_path = os.path.join(output_images_dir, new_image_name)
        
        try:
            # 复制图片到输出目录
            if not os.path.exists(new_image_path):
                shutil.copy2(old_image_path, new_image_path)
            
            # 更新内容中的图片路径
            # 匹配 ![alt](images/filename) 格式
            old_path_pattern = f"images/{image_file}"
            new_path_pattern = f"images/{new_image_name}"
            
            # 使用正则表达式替换图片路径
            pattern = r'!\[([^\]]*)\]\((' + re.escape(old_path_pattern) + r')\)'
            replacement = r'![\1](' + new_path_pattern + r')'
            updated_content = re.sub(pattern, replacement, updated_content)
            
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")
    
    return updated_content

def get_content(base_path: str, question_type: str, dir_name: str, sub_num: int, image_mapping: Dict[str, str], output_dir: str) -> str:
    """获取题目内容（题干、答案或解析）并更新图片路径"""
    dir_path = os.path.join(base_path, question_type, dir_name)
    if not os.path.exists(dir_path):
        return ""
    
    # 查找md文件
    md_file = os.path.join(dir_path, f"{dir_name}.md")
    if os.path.exists(md_file):
        content = read_md_file(md_file)
        return update_content_image_paths(content, md_file, output_dir)
    
    return ""

def process_questions(root_path: str, output_dir: str, original_path: str, logger: logging.Logger, max_workers: int = 4) -> List[Dict]:
    """处理所有题目并返回JSON格式的数据"""
    questions = []
    question_id = 1  # 题目ID计数器
    
    # 遍历根目录下的所有竞赛目录
    for competition_dir in os.listdir(root_path):
        competition_path = os.path.join(root_path, competition_dir)
        if not os.path.isdir(competition_path):
            continue
        
        logger.info(f"处理竞赛: {competition_dir}")
        
        # 遍历竞赛目录下的题目类型目录（理论题/实验题等）
        for question_type_dir in os.listdir(competition_path):
            question_type_path = os.path.join(competition_path, question_type_dir)
            if not os.path.isdir(question_type_path):
                continue
            
            logger.info(f"  处理题目类型: {question_type_dir}")
            
            # 检查题目类型目录下是否直接有题干/答案/解析目录
            question_base = os.path.join(question_type_path, "题干")
            answer_base = os.path.join(question_type_path, "答案")
            analysis_base = os.path.join(question_type_path, "解析")
            
            if os.path.exists(question_base) and os.path.exists(answer_base):
                # 直接处理题干/答案/解析目录
                logger.info(f"    直接处理题干/答案/解析目录")
                new_questions = process_question_group_internal(
                    question_type_path, output_dir, competition_dir, 
                    question_id, logger
                )
                questions.extend(new_questions)
                question_id += len(new_questions)
            else:
                # 检查是否有子分类目录（如Part1、Part2等）
                for sub_category_dir in os.listdir(question_type_path):
                    sub_category_path = os.path.join(question_type_path, sub_category_dir)
                    if not os.path.isdir(sub_category_path):
                        continue
                    
                    logger.info(f"    处理子分类: {sub_category_dir}")
                    
                    # 检查子分类目录下是否有题干/答案/解析目录
                    sub_question_base = os.path.join(sub_category_path, "题干")
                    sub_answer_base = os.path.join(sub_category_path, "答案")
                    sub_analysis_base = os.path.join(sub_category_path, "解析")
                    
                    if os.path.exists(sub_question_base) or os.path.exists(sub_answer_base):
                        new_questions = process_question_group_internal(
                            sub_category_path, output_dir, competition_dir, 
                            question_id, logger
                        )
                        questions.extend(new_questions)
                        question_id += len(new_questions)
    
    return questions

def process_question_group_internal(base_path: str, output_dir: str, competition_dir: str, 
                         start_id: int, logger: logging.Logger) -> List[Dict]:
    """处理一个题目组（题干/答案/解析）的内部函数"""
    questions = []
    question_id = start_id
    
    # 检查是否有题干目录
    question_base = os.path.join(base_path, "题干")
    answer_base = os.path.join(base_path, "答案")
    analysis_base = os.path.join(base_path, "解析")
    
    # 获取所有题干目录
    question_dirs = []
    if os.path.exists(question_base):
        for item in os.listdir(question_base):
            item_path = os.path.join(question_base, item)
            if os.path.isdir(item_path):
                question_dirs.append(item)
    
    # 获取所有答案目录
    answer_dirs = []
    if os.path.exists(answer_base):
        for item in os.listdir(answer_base):
            item_path = os.path.join(answer_base, item)
            if os.path.isdir(item_path):
                answer_dirs.append(item)
    
    # 获取所有解析目录
    analysis_dirs = []
    if os.path.exists(analysis_base):
        for item in os.listdir(analysis_base):
            item_path = os.path.join(analysis_base, item)
            if os.path.isdir(item_path):
                analysis_dirs.append(item)
    
    logger.info(f"      找到题干目录: {question_dirs}")
    logger.info(f"      找到答案目录: {answer_dirs}")
    logger.info(f"      找到解析目录: {analysis_dirs}")
    
    # 合并所有目录，按题号分组
    all_groups = {}
    
    # 处理题干目录
    for dir_name in question_dirs:
        question_number = extract_question_number(dir_name)
        if question_number not in all_groups:
            all_groups[question_number] = {"题干": [], "答案": [], "解析": []}
        all_groups[question_number]["题干"].append(dir_name)
    
    # 处理答案目录
    for dir_name in answer_dirs:
        question_number = extract_question_number(dir_name)
        if question_number not in all_groups:
            all_groups[question_number] = {"题干": [], "答案": [], "解析": []}
        all_groups[question_number]["答案"].append(dir_name)
    
    # 处理解析目录
    for dir_name in analysis_dirs:
        question_number = extract_question_number(dir_name)
        if question_number not in all_groups:
            all_groups[question_number] = {"题干": [], "答案": [], "解析": []}
        all_groups[question_number]["解析"].append(dir_name)
    
    # 打印最终所有题号
    all_question_numbers = sorted(all_groups.keys())
    logger.info(f"      最终所有题号: {all_question_numbers}")
    
    # 处理每个题组
    for question_number in sorted(all_groups.keys()):
        if question_number == "":
            continue  # 跳过无效题号
            
        # 按序号排序目录名
        def extract_sub_number(dir_name):
            match = re.match(r'^.*?-(\d+)$', dir_name)
            return int(match.group(1)) if match else 0
        
        # 获取各部分目录，按序号排序
        question_dir_names = sorted(all_groups[question_number]["题干"], key=extract_sub_number)
        answer_dir_names = sorted(all_groups[question_number]["答案"], key=extract_sub_number)
        analysis_dir_names = sorted(all_groups[question_number]["解析"], key=extract_sub_number)
        
        logger.info(f"        处理题目 {question_number}")
        logger.info(f"        题干目录: {question_dir_names}")
        logger.info(f"        答案目录: {answer_dir_names}")
        logger.info(f"        解析目录: {analysis_dir_names}")
        
        # 合并题干 - 按序号顺序拼接
        question_content = ""
        for dir_name in question_dir_names:
            content = get_content(base_path, "题干", dir_name, 1, {}, output_dir)
            if content:
                if question_content:
                    question_content += "\n\n"
                question_content += content
        
        # 合并答案 - 按序号顺序拼接
        answer_content = ""
        for dir_name in answer_dir_names:
            content = get_content(base_path, "答案", dir_name, 1, {}, output_dir)
            if content:
                if answer_content:
                    answer_content += "\n\n"
                answer_content += content
        
        # 合并解析 - 按序号顺序拼接
        analysis_content = ""
        for dir_name in analysis_dir_names:
            content = get_content(base_path, "解析", dir_name, 1, {}, output_dir)
            if content:
                if analysis_content:
                    analysis_content += "\n\n"
                analysis_content += content
        
        # 从question、answer、cot字段中提取图片路径
        current_image_paths = []
        
        # 合并所有文本内容用于匹配
        all_content = f"{question_content} {answer_content} {analysis_content}"
        
        # 使用正则表达式匹配图片路径
        # 匹配 ![alt](path) 格式的markdown图片引用
        markdown_image_pattern = r'!\[.*?\]\((.*?)\)'
        markdown_matches = re.findall(markdown_image_pattern, all_content)
        
        # 收集所有匹配到的图片路径
        all_matched_paths = markdown_matches
        
        # 添加调试日志
        logger.info(f"        题目 {question_number} 匹配到的图片路径: {all_matched_paths}")
        
        # 从匹配的路径中提取图片路径
        for matched_path in all_matched_paths:
            if matched_path.startswith("images/"):
                current_image_paths.append(matched_path)
                logger.info(f"          添加图片路径: {matched_path}")
        
        # 按照文件名中的序号排序
        def extract_number_from_filename(path):
            """从路径中提取文件名，然后提取序号"""
            filename = os.path.basename(path)
            # 匹配题号-数字的格式，如A-5中的5
            match = re.search(rf'{re.escape(question_number)}-(\d+)', filename)
            if match:
                return int(match.group(1))
            # 如果没有匹配到题号-数字格式，返回0
            return 0
        
        # 按照序号排序
        current_image_paths.sort(key=extract_number_from_filename)
        
        # 获取原始图片路径（相对于source目录）
        source_dir = os.path.join(output_dir, "source")
        question_origin_images = get_original_image_paths(question_number, "题干", source_dir, competition_dir)
        answer_origin_images = get_original_image_paths(question_number, "答案", source_dir, competition_dir)
        cot_origin_images = get_original_image_paths(question_number, "解析", source_dir, competition_dir)
        
        # 构建题目对象
        question_obj = {
            "id": question_id,
            "subject": competition_dir,  # 使用竞赛名称作为主题
            "qtype": "",
            "question": question_content,
            "answer": answer_content,
            "cot": analysis_content,
            "keypoint": [],
            "image_paths": current_image_paths,
            "question_origin_images": question_origin_images,
            "answer_origin_images": answer_origin_images,
            "cot_origin_images": cot_origin_images
        }
        
        # 生成qtype、keypoints字段
        # question_obj = generate_fields(question_obj)
        
        questions.append(question_obj)
        logger.info(f"        处理完成: {question_number}题 (ID: {question_id})")
        question_id += 1
    
    return questions


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='提取题目内容并生成JSON文件，同时处理图片重命名')
    parser.add_argument('root_path', help='根目录路径，包含题目文件的目录')
    parser.add_argument('original_path', help='原始图片目录路径，包含CPHOS17等原始文件的目录')
    parser.add_argument('-o', '--output', help='输出目录名 (默认: CoT_root名字_日期)')
    parser.add_argument('-j', '--jobs', type=int, default=200, help='并发线程数 (默认: 200)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    root_path = args.root_path
    original_path = args.original_path
    max_workers = args.jobs
    
    if not os.path.exists(root_path):
        print(f"根目录不存在: {root_path}")
        sys.exit(1)
    
    if not os.path.exists(original_path):
        print(f"原始图片目录不存在: {original_path}")
        sys.exit(1)
    
    # 创建输出目录
    if args.output:
        output_dir = args.output
    else:
        # 生成默认输出目录名：CoT_root名字_日期时间
        root_name = os.path.basename(os.path.abspath(root_path))
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"CoT_{root_name}_{datetime_str}"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info(f"开始处理题目，根目录: {root_path}")
    logger.info(f"原始图片目录: {original_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"并发线程数: {max_workers}")
    
    # 复制原始文件夹到输出目录，重命名为source
    source_dir = os.path.join(output_dir, "source")
    if not os.path.exists(source_dir):
        try:
            shutil.copytree(original_path, source_dir)
            logger.info(f"复制原始文件夹到: {source_dir}")
        except Exception as e:
            logger.error(f"复制原始文件夹失败: {e}")
            sys.exit(1)
    
    questions = process_questions(root_path, output_dir, original_path, logger, max_workers)
    
    # 生成字段（并发处理）
    logger.info(f"\n开始生成字段，共 {len(questions)} 道题目...")
    
    def process_question_fields(question_data):
        index, question_obj = question_data
        logger.info(f"处理第 {index+1}/{len(questions)} 题:")
        try:
            question_obj = generate_fields(question_obj, logger=logger)
            logger.info(f"第 {index+1} 题处理完成")
            return index, question_obj
        except Exception as e:
            logger.error(f"第 {index+1} 题处理失败: {e}")
            return index, question_obj
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_question_fields, (i, question_obj)): i 
            for i, question_obj in enumerate(questions)
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            try:
                index, updated_question = future.result()
                questions[index] = updated_question
            except Exception as e:
                logger.error(f"处理题目时发生错误: {e}")
    
    # 保存为JSON文件
    output_file = os.path.join(output_dir, "questions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成！共处理 {len(questions)} 道题目")
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"图片已保存到: {os.path.join(output_dir, 'images')}")
    logger.info(f"日志文件保存到: {os.path.join(output_dir, 'extract_questions.log')}")
    
    # 显示前几道题的结构
    if questions:
        logger.info("\n示例题目结构:")
        logger.info(json.dumps(questions[0], ensure_ascii=False, indent=2))

def generate_qtype_and_keypoints_concurrent(question_data: tuple, logger: logging.Logger = None) -> tuple:
    """并发生成题目类型和关键知识点"""
    index, question, answer, analysis = question_data
    try:
        qtype, keypoints = generate_qtype_and_keypoints(question, answer, analysis, logger=logger)
        return index, qtype, keypoints
    except Exception as e:
        error_msg = f"生成qtype和keypoints失败 (题目{index+1}): {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return index, "综合题", []

def generate_qtype_and_keypoints(question: str, answer: str, analysis: str, retries: int = 3, logger: logging.Logger = None) -> tuple:
    """合并生成题目类型和关键知识点"""
    client = Ark(api_key=API_KEY)
    
    system_prompt = """你是一个专业的题目分析助手。根据给定的题目内容、答案和解析，你需要完成两个任务：

1. 判断题目类型：从以下选项中选择一个
   - 选择题 - 题目提供多个选项供选择
   - 填空题 - 需要填写答案的题目
   - 计算题 - 需要进行数学计算的题目
   - 证明题 - 需要证明某个结论的题目
   - 实验题 - 涉及实验操作或实验数据的题目
   - 分析题 - 需要分析现象或过程的题目
   - 设计题 - 需要设计实验方案或装置的题目
   - 综合题 - 涉及多个知识点的复杂题目

2. 提取关键知识点：提取出3-8个最重要的物理关键知识点，涵盖核心概念和术语、关键的方法或技术、重要的理论或定律、重要的计算方法和公式。

请严格按照以下JSON格式输出：
{
  "qtype": "题目类型",
  "keypoint": ["知识点1", "知识点2", "知识点3", ...]
}

关键知识点应该是该题所考察的物理理论、物理学科知识点，而非该题提的是什么问题。"""
    
    user_prompt = f"""请分析以下题目：

题目：{question}
答案：{answer}
解析：{analysis}

请按照要求的JSON格式输出题目类型和关键知识点。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.0,
                top_p=0.9,
            )
            
            content = resp.choices[0].message.content
            return parse_combined_response(content)
            
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.6 ** (attempt - 1))  # 指数退避
    
    error_msg = f"生成qtype和keypoints失败 (重试{retries}次): {last_err}"
    if logger:
        logger.error(error_msg)
    else:
        print(error_msg)
    return "综合题", []

def parse_combined_response(content: str) -> tuple:
    """解析合并响应的JSON内容"""
    if not content:
        return "综合题", []
    
    # 清理内容
    content = content.strip()
    
    # 移除markdown代码块标记
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    
    # 尝试直接解析
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            qtype = data.get("qtype", "综合题")
            keypoints = data.get("keypoint", [])
            if isinstance(keypoints, list):
                keypoints = [str(k).strip() for k in keypoints if k and str(k).strip()]
            return qtype, keypoints
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                if isinstance(data, dict):
                    qtype = data.get("qtype", "综合题")
                    keypoints = data.get("keypoint", [])
                    if isinstance(keypoints, list):
                        keypoints = [str(k).strip() for k in keypoints if k and str(k).strip()]
                    return qtype, keypoints
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return "综合题", []

def parse_keypoints_response(content: str) -> List[str]:
    """解析模型返回的keypoints JSON内容"""
    if not content:
        return []
    
    # 清理内容
    content = content.strip()
    
    # 移除markdown代码块标记
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    
    # 尝试直接解析
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "keypoint" in data:
            keypoints = data.get("keypoint", [])
            if isinstance(keypoints, list):
                return [str(k).strip() for k in keypoints if k and str(k).strip()]
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                if isinstance(data, dict) and "keypoint" in data:
                    keypoints = data.get("keypoint", [])
                    if isinstance(keypoints, list):
                        return [str(k).strip() for k in keypoints if k and str(k).strip()]
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return []

def generate_fields(question_obj: Dict, logger: logging.Logger = None) -> Dict:
    """生成qtype、keypoint字段"""
    question = question_obj.get("question", "")
    answer = question_obj.get("answer", "")
    analysis = question_obj.get("cot", "")
    
    # 检查是否需要生成qtype或keypoint
    needs_qtype = not question_obj.get("qtype")
    needs_keypoint = not question_obj.get("keypoint")
    
    # 如果两个都需要生成，使用合并函数
    if needs_qtype and needs_keypoint:
        qtype, keypoints = generate_qtype_and_keypoints(question, answer, analysis, logger=logger)
        question_obj["qtype"] = qtype
        question_obj["keypoint"] = keypoints
        if logger:
            logger.info(f"  生成qtype: {qtype}, keypoint: {len(keypoints)}个")
        else:
            print(f"  生成qtype: {qtype}, keypoint: {len(keypoints)}个")
    else:
        # 如果只需要生成其中一个，使用合并函数但只取需要的部分
        if needs_qtype or needs_keypoint:
            qtype, keypoints = generate_qtype_and_keypoints(question, answer, analysis, logger=logger)
            if needs_qtype:
                question_obj["qtype"] = qtype
                if logger:
                    logger.info(f"  生成qtype: {qtype}")
                else:
                    print(f"  生成qtype: {qtype}")
            if needs_keypoint:
                question_obj["keypoint"] = keypoints
                if logger:
                    logger.info(f"  生成keypoint: {len(keypoints)}个")
                else:
                    print(f"  生成keypoint: {len(keypoints)}个")
    
    return question_obj

if __name__ == "__main__":
    main()
