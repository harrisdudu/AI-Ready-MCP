# -*- coding: utf-8 -*-
"""
增强版文件夹与Excel匹配脚本：
1. 先通过文件夹名称与Excel项目编号进行严格匹配
2. 再通过文件编号进行精确匹配
3. 最后在文件夹严格匹配的基础上，增强文件名与规划名称的相似匹配
"""

import os
import re
import pandas as pd
import difflib
import logging
import jieba
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FolderMatcher:
    """文件夹与Excel匹配器类，封装所有匹配相关的功能"""
    
    def __init__(self, excel_path):
        """
        初始化匹配器
        :param excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.df = None
        self.folder_infos = []
        self.number_to_folder = {}
        self.name_to_folder = {}
        self.project_id_col = None
        self.planning_name_col = None
    
    def load_excel(self):
        """读取并解析Excel文件"""
        try:
            logger.info(f"正在读取Excel文件: {self.excel_path}")
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"成功读取Excel文件，共{len(self.df)}条记录")
            logger.info(f"Excel文件的列名: {list(self.df.columns)}")
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            return False
        
        # 检查是否有项目编号列
        project_id_columns = ['项目编号', '编号', '档案编号', '文件编号']
        for col in project_id_columns:
            if col in self.df.columns:
                self.project_id_col = col
                break
        
        if not self.project_id_col:
            logger.error("未找到项目编号列，请确保Excel文件中包含'项目编号'、'编号'、'档案编号'或'文件编号'列")
            return False
        
        logger.info(f"使用'{self.project_id_col}'列作为项目编号")
        
        # 尝试获取规划名称列
        planning_name_columns = ['规划名称', '项目名称', '名称', '档案名称']
        for col in planning_name_columns:
            if col in self.df.columns:
                self.planning_name_col = col
                break
        
        if not self.planning_name_col:
            logger.warning("未找到规划名称列，无法进行规划名称匹配")
        else:
            logger.info(f"使用'{self.planning_name_col}'列作为规划名称")
        
        # 在Excel中准备列
        required_columns = {
            '匹配文件编号': None,
            '匹配文件夹名称': None,
            '匹配文件名': None,
            '匹配相似度': None,
            '匹配状态': None,  # 标记匹配状态：文件夹名称匹配、文件编号匹配、完全匹配、未匹配
            '匹配类型': None  # 匹配类型说明
        }
        
        for col, default_value in required_columns.items():
            if col not in self.df.columns:
                self.df[col] = default_value
        
        return True
    
    def scan_folders(self):
        """扫描所有有效文件夹并提取信息"""
        self.folder_infos = []
        self.number_to_folder = {}
        self.name_to_folder = {}
        
        main_dir = os.getcwd()
        for item in os.listdir(main_dir):
            item_path = os.path.join(main_dir, item)
            if os.path.isdir(item_path) and self._is_valid_folder(item):
                folder_info = self._extract_folder_info(item_path)
                if folder_info:
                    self.folder_infos.append(folder_info)
                    # 添加到映射中
                    if folder_info['number']:
                        self.number_to_folder[folder_info['number']] = folder_info
                    self.name_to_folder[folder_info['folder_name']] = folder_info
        
        logger.info(f"共扫描到{len(self.folder_infos)}个有效的文件夹")
        
        # 打印一些文件夹信息示例
        if self.folder_infos:
            logger.info("文件夹信息示例:")
            for i, folder in enumerate(self.folder_infos[:3]):
                logger.info(f"  {i+1}. 编号: {folder['number']}, 文件夹名称: {folder['folder_name']}")
        
        return len(self.folder_infos) > 0
    
    def folder_name_matching(self):
        """阶段1: 文件夹名称与项目编号直接匹配"""
        name_matched_count = 0
        
        for idx, row in self.df.iterrows():
            # 获取项目编号
            project_id = row.get(self.project_id_col)
            
            if pd.isna(project_id):
                continue
            
            project_id_str = str(project_id)
            
            # 检查文件夹名称是否与项目编号完全匹配
            if project_id_str in self.name_to_folder:
                matched_folder = self.name_to_folder[project_id_str]
                self.df.at[idx, '匹配文件编号'] = matched_folder['number']
                self.df.at[idx, '匹配文件夹名称'] = matched_folder['folder_name']
                self.df.at[idx, '匹配状态'] = '文件夹名称匹配'
                self.df.at[idx, '匹配类型'] = '精确匹配 - 文件夹名称与项目编号完全一致'
                name_matched_count += 1
                continue
        
        logger.info(f"阶段1（文件夹名称与项目编号直接匹配）成功匹配{name_matched_count}条记录")
        return name_matched_count
    
    def file_number_matching(self):
        """阶段2: 文件编号精确匹配"""
        number_matched_count = 0
        
        for idx, row in self.df.iterrows():
            # 跳过已经匹配的记录
            if row['匹配状态'] is not None:
                continue
            
            # 获取项目编号
            project_id = row.get(self.project_id_col)
            
            if pd.isna(project_id):
                continue
            
            # 尝试从项目编号中提取数字部分进行匹配
            project_id_str = str(project_id)
            
            # 方法1: 尝试直接匹配文件夹编号
            for folder_number, folder_info in self.number_to_folder.items():
                # 检查文件夹编号是否在项目编号中出现
                if folder_number in project_id_str:
                    self.df.at[idx, '匹配文件编号'] = folder_number
                    self.df.at[idx, '匹配文件夹名称'] = folder_info['folder_name']
                    self.df.at[idx, '匹配状态'] = '文件编号匹配'
                    self.df.at[idx, '匹配类型'] = f'文件编号{folder_number}在项目编号中出现'
                    number_matched_count += 1
                    break
        
        logger.info(f"阶段2（文件编号精确匹配）成功匹配{number_matched_count}条记录")
        return number_matched_count
    
    def enhanced_filename_planning_matching(self):
        """阶段3: 增强的文件名与规划名称相似匹配"""
        exact_matched_count = 0
        
        if not self.planning_name_col:
            logger.warning("没有规划名称列，无法进行文件名与规划名称的匹配")
            return 0
        
        for idx, row in self.df.iterrows():
            # 只处理已文件夹严格匹配的记录
            if row['匹配状态'] not in ['文件夹名称匹配', '文件编号匹配']:
                continue
            
            # 获取规划名称
            planning_name = row.get(self.planning_name_col)
            if pd.isna(planning_name) or not isinstance(planning_name, str):
                continue
            
            # 获取匹配的文件夹编号
            matched_number = row['匹配文件编号']
            if pd.isna(matched_number):
                continue
            
            # 获取文件夹信息
            folder_info = self.number_to_folder.get(str(matched_number))
            if not folder_info:
                # 如果通过文件夹名称匹配，使用名称映射查找
                folder_info = self.name_to_folder.get(row['匹配文件夹名称'])
                if not folder_info:
                    continue
            
            # 在文件夹中查找与规划名称最匹配的文件
            best_file_name, file_similarity = self._find_best_matching_file_enhanced(folder_info, planning_name)
            
            # 设置一个合理的阈值，例如0.6
            if file_similarity >= 0.6:
                self.df.at[idx, '匹配文件名'] = best_file_name
                self.df.at[idx, '匹配相似度'] = file_similarity
                self.df.at[idx, '匹配状态'] = '完全匹配'
                self.df.at[idx, '匹配类型'] = f'文件夹严格匹配+文件名与规划名称相似匹配，相似度: {file_similarity:.2f}'
                exact_matched_count += 1
        
        logger.info(f"阶段3（文件名与规划名称相似匹配）成功匹配{exact_matched_count}条记录")
        return exact_matched_count
    
    def save_results(self, output_path):
        """保存匹配结果到Excel文件"""
        try:
            self.df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"已成功保存匹配后的Excel文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存Excel文件失败: {e}")
            return False
    
    def generate_report(self):
        """生成匹配情况报告"""
        # 计算总匹配数量
        total_name_matched = len(self.df[self.df['匹配状态'] == '文件夹名称匹配'])
        total_number_matched = len(self.df[self.df['匹配状态'] == '文件编号匹配'])
        total_exact_matched = len(self.df[self.df['匹配状态'] == '完全匹配'])
        total_any_matched = total_name_matched + total_number_matched + total_exact_matched
        
        # 输出匹配情况报告
        logger.info("\n匹配情况报告:")
        logger.info(f"- 总记录数量: {len(self.df)}")
        logger.info(f"- 文件夹名称匹配数量: {total_name_matched}")
        logger.info(f"- 文件编号匹配数量: {total_number_matched}")
        logger.info(f"- 完全匹配数量: {total_exact_matched}")
        logger.info(f"- 总匹配率: {total_any_matched/len(self.df)*100:.2f}%")
        
        # 输出完全匹配的例子
        exact_matched = self.df[self.df['匹配状态'] == '完全匹配']
        if not exact_matched.empty:
            logger.info(f"\n完全匹配的例子({len(exact_matched)}个):")
            for i, (_, row) in enumerate(exact_matched.head(5).iterrows()):
                logger.info(f"  {i+1}. 项目编号: {row[self.project_id_col]}")
                logger.info(f"     规划名称: {row[self.planning_name_col]}")
                logger.info(f"     匹配文件夹: {row['匹配文件夹名称']}")
                logger.info(f"     匹配文件名: {row['匹配文件名']}")
                logger.info(f"     匹配相似度: {row['匹配相似度']:.2f}")
                logger.info(f"     匹配类型: {row['匹配类型']}")
        
        # 统计未匹配的文件夹数量
        matched_numbers = set()
        matched_names = set()
        
        for _, row in self.df[self.df['匹配文件编号'].notna()].iterrows():
            if not pd.isna(row['匹配文件编号']):
                matched_numbers.add(str(row['匹配文件编号']))
            if not pd.isna(row['匹配文件夹名称']):
                matched_names.add(str(row['匹配文件夹名称']))
        
        unmatched_folders = []
        for folder in self.folder_infos:
            if (folder['number'] and folder['number'] not in matched_numbers) and (folder['folder_name'] not in matched_names):
                unmatched_folders.append(folder)
        
        logger.info(f"\n未匹配的文件夹数量: {len(unmatched_folders)}")
        
        # 如果有未匹配的文件夹，显示一些例子
        if unmatched_folders and len(unmatched_folders) <= 5:
            logger.info("未匹配的文件夹:")
            for folder in unmatched_folders:
                logger.info(f"  - 编号: {folder['number']}, 文件夹名称: {folder['folder_name']}")
        elif unmatched_folders:
            logger.info(f"前5个未匹配的文件夹:")
            for folder in unmatched_folders[:5]:
                logger.info(f"  - 编号: {folder['number']}, 文件夹名称: {folder['folder_name']}")
            logger.info(f"  ... 还有{len(unmatched_folders)-5}个未显示")
    
    def _extract_folder_info(self, folder_path):
        """
        提取文件夹中的文件编号和详细信息
        :param folder_path: 文件夹路径
        :return: 包含文件夹信息的字典
        """
        folder_name = os.path.basename(folder_path)
        files = os.listdir(folder_path)
        
        if not files:
            return None
        
        # 获取编号（假设所有文件都有相同的编号）
        first_file = files[0]
        match = re.match(r'(\d+)_', first_file)
        if match:
            number = match.group(1)
        else:
            number = None
        
        # 提取所有可能的关键词和文本
        all_text = []
        
        # 从文件夹名称提取
        all_text.append(folder_name)
        
        # 从所有文件名提取
        for file in files:
            # 去除编号前缀
            name_without_number = re.sub(r'^\d+_', '', file)
            # 去除文件扩展名
            name_without_ext = os.path.splitext(name_without_number)[0]
            all_text.append(name_without_ext)
        
        # 合并所有文本
        combined_text = ' '.join(all_text)
        
        return {
            'folder_name': folder_name,
            'number': number,
            'all_text': combined_text,
            'file_names': files,
            'folder_path': folder_path
        }
    
    def _normalize_text(self, text):
        """
        标准化文本，移除特殊字符，转换为小写
        :param text: 原始文本
        :return: 标准化后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 移除非中文字符和数字
        normalized = re.sub(r'[^\u4e00-\u9fa5\d]', '', text)
        # 转换为小写
        normalized = normalized.lower()
        return normalized
    
    def _calculate_similarity(self, text1, text2, is_strict=False):
        """
        计算两个文本的相似度
        :param text1: 第一个文本
        :param text2: 第二个文本
        :param is_strict: 是否使用严格匹配模式
        :return: 相似度分数（0-1）
        """
        if not text1 or not text2 or not isinstance(text1, str) or not isinstance(text2, str):
            return 0
        
        # 标准化文本
        normalized_text1 = self._normalize_text(text1)
        normalized_text2 = self._normalize_text(text2)
        
        if not normalized_text1 or not normalized_text2:
            return 0
        
        # 方法1: 精确包含关系
        if normalized_text1 in normalized_text2 or normalized_text2 in normalized_text1:
            return 1.0  # 非常高的相似度
        
        # 方法2: difflib相似度
        seq_matcher = difflib.SequenceMatcher(None, normalized_text1, normalized_text2)
        difflib_ratio = seq_matcher.ratio()
        
        # 方法3: 关键词匹配 - 使用jieba分词并计算词频
        try:
            # 分词
            words1 = jieba.lcut(normalized_text1)
            words2 = jieba.lcut(normalized_text2)
            
            # 计算词频
            counter1 = Counter(words1)
            counter2 = Counter(words2)
            
            # 计算共同关键词
            common_words = set(counter1.keys()) & set(counter2.keys())
            
            if common_words:
                # 计算关键词匹配得分
                keyword_score = len(common_words) / max(len(counter1), len(counter2))
                # 综合得分 = difflib相似度 * 0.6 + 关键词匹配得分 * 0.4
                combined_score = difflib_ratio * 0.6 + keyword_score * 0.4
                return combined_score
        except Exception:
            # 如果分词失败，回退到difflib相似度
            pass
        
        return difflib_ratio
    
    def _find_best_matching_file_enhanced(self, folder_info, planning_name):
        """
        在文件夹中查找与规划名称最匹配的文件（增强版）
        :param folder_info: 文件夹信息
        :param planning_name: 规划名称
        :return: 最佳匹配的文件名和相似度
        """
        best_file_name = None
        best_file_similarity = 0
        
        # 提取规划名称中的关键词
        planning_keywords = set(re.findall(r'[\u4e00-\u9fa5]+', planning_name))
        
        for file in folder_info['file_names']:
            # 使用增强版相似度计算
            file_similarity = self._calculate_similarity(planning_name, file)
            
            # 额外的关键词匹配加分
            file_keywords = set(re.findall(r'[\u4e00-\u9fa5]+', file))
            common_keywords = planning_keywords & file_keywords
            if common_keywords:
                # 为每个共同关键词增加额外的相似度分数
                keyword_boost = len(common_keywords) * 0.1
                file_similarity = min(1.0, file_similarity + keyword_boost)
            
            if file_similarity > best_file_similarity:
                best_file_similarity = file_similarity
                best_file_name = file
        
        return best_file_name, best_file_similarity
    
    def _is_valid_folder(self, folder_name):
        """
        检查文件夹名称是否为有效的项目文件夹
        :param folder_name: 文件夹名称
        :return: 是否为有效文件夹
        """
        # 避免处理临时文件或系统文件夹
        if folder_name.startswith('.') or folder_name.startswith('~$'):
            return False
        # 避免处理脚本文件所在的文件夹
        if folder_name == '第一次匹配加工':
            return False
        return True


def main():
    """主函数"""
    # Excel文件路径
    excel_path = os.path.join(os.getcwd(), '5年历史档案（总规详规）清单.xlsx')
    output_path = os.path.join(os.getcwd(), '5年历史档案（总规详规）清单_增强相似度匹配.xlsx')
    
    # 创建匹配器实例
    matcher = FolderMatcher(excel_path)
    
    # 加载Excel文件
    if not matcher.load_excel():
        return
    
    # 扫描文件夹
    if not matcher.scan_folders():
        logger.error("未找到有效文件夹，无法进行匹配")
        return
    
    # 执行三阶段匹配
    matcher.folder_name_matching()  # 阶段1: 文件夹名称与项目编号直接匹配
    matcher.file_number_matching()  # 阶段2: 文件编号精确匹配
    matcher.enhanced_filename_planning_matching()  # 阶段3: 增强的文件名与规划名称相似匹配
    
    # 保存结果
    if not matcher.save_results(output_path):
        return
    
    # 生成报告
    matcher.generate_report()


if __name__ == "__main__":
    main()