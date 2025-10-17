# -*- coding: utf-8 -*-
"""
严格文件夹关联匹配脚本：
1. 严格按照文件夹名称和Excel项目编号进行精确匹配
2. 对于匹配到的文件夹，将其中的所有文件都关联到Excel中
3. 保持简单直接的匹配逻辑
"""

import os
import pandas as pd
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrictFolderAssociator:
    """严格文件夹关联器类，实现文件夹名称与项目编号的精确匹配及文件关联"""
    
    def __init__(self, excel_path):
        """
        初始化关联器
        :param excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.df = None
        self.folders = []
        self.name_to_folder = {}
        self.project_id_col = None
    
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
        
        # 在Excel中准备列
        required_columns = {
            '匹配文件夹名称': None,
            '匹配文件数量': None,
            '匹配文件列表': None,
            '匹配状态': None  # 标记匹配状态：已匹配、未匹配
        }
        
        for col, default_value in required_columns.items():
            if col not in self.df.columns:
                self.df[col] = default_value
        
        return True
    
    def scan_folders(self):
        """扫描所有有效文件夹"""
        self.folders = []
        self.name_to_folder = {}
        
        main_dir = os.getcwd()
        for item in os.listdir(main_dir):
            item_path = os.path.join(main_dir, item)
            if os.path.isdir(item_path) and self._is_valid_folder(item):
                # 获取文件夹中的所有文件
                files = os.listdir(item_path)
                self.folders.append({
                    'name': item,
                    'path': item_path,
                    'files': files
                })
                self.name_to_folder[item] = {
                    'path': item_path,
                    'files': files
                }
        
        logger.info(f"共扫描到{len(self.folders)}个有效的文件夹")
        
        # 打印一些文件夹信息示例
        if self.folders:
            logger.info("文件夹信息示例:")
            for i, folder in enumerate(self.folders[:3]):
                logger.info(f"  {i+1}. 文件夹名称: {folder['name']}, 文件数量: {len(folder['files'])}")
        
        return len(self.folders) > 0
    
    def associate_folders(self):
        """执行文件夹与项目编号的严格匹配和文件关联"""
        matched_count = 0
        
        for idx, row in self.df.iterrows():
            # 获取项目编号
            project_id = row.get(self.project_id_col)
            
            if pd.isna(project_id):
                continue
            
            project_id_str = str(project_id)
            
            # 严格按照文件夹名称与项目编号进行精确匹配
            if project_id_str in self.name_to_folder:
                folder_info = self.name_to_folder[project_id_str]
                
                # 关联文件夹信息
                self.df.at[idx, '匹配文件夹名称'] = project_id_str
                self.df.at[idx, '匹配文件数量'] = len(folder_info['files'])
                
                # 将所有文件列表关联到Excel，使用换行符实现并列显示
                # 由于Excel单元格有字符限制，我们可以保存文件名列表或文件路径列表
                file_list = '\n'.join(folder_info['files'])  # 使用换行符替代分号
                if len(file_list) > 32767:  # Excel单元格字符限制
                    # 如果文件名列表太长，只保存前20个文件名
                    file_list = '\n'.join(folder_info['files'][:20]) + f"\n... 等共{len(folder_info['files'])}个文件"
                
                self.df.at[idx, '匹配文件列表'] = file_list
                self.df.at[idx, '匹配状态'] = '已匹配'
                matched_count += 1
            else:
                self.df.at[idx, '匹配状态'] = '未匹配'
        
        logger.info(f"成功匹配并关联了{matched_count}个文件夹")
        return matched_count
    
    def save_results(self, output_path):
        """保存关联结果到Excel文件"""
        try:
            self.df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"已成功保存严格匹配关联后的Excel文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存Excel文件失败: {e}")
            return False
    
    def generate_report(self):
        """生成关联情况报告"""
        # 计算匹配数量
        total_matched = len(self.df[self.df['匹配状态'] == '已匹配'])
        total_unmatched = len(self.df[self.df['匹配状态'] == '未匹配'])
        total_records = len(self.df)
        
        # 输出匹配情况报告
        logger.info("\n匹配情况报告:")
        logger.info(f"- 总记录数量: {total_records}")
        logger.info(f"- 已匹配数量: {total_matched}")
        logger.info(f"- 未匹配数量: {total_unmatched}")
        logger.info(f"- 匹配率: {total_matched/total_records*100:.2f}%")
        
        # 输出已匹配的例子
        matched_records = self.df[self.df['匹配状态'] == '已匹配']
        if not matched_records.empty:
            logger.info(f"\n已匹配的例子({len(matched_records)}个):")
            for i, (_, row) in enumerate(matched_records.head(3).iterrows()):
                logger.info(f"  {i+1}. 项目编号: {row[self.project_id_col]}")
                logger.info(f"     匹配文件夹: {row['匹配文件夹名称']}")
                logger.info(f"     文件数量: {row['匹配文件数量']}")
                
                # 显示部分文件名列表
                file_list = row['匹配文件列表']
                if isinstance(file_list, str):
                    # 只显示前3个文件名
                    files_preview = file_list.split('; ')[:3]
                    logger.info(f"     文件示例: {', '.join(files_preview)}{'...' if len(files_preview) < row['匹配文件数量'] else ''}")
        
        # 统计未匹配的文件夹数量
        matched_folder_names = set(self.df[self.df['匹配文件夹名称'].notna()]['匹配文件夹名称'])
        unmatched_folders = [folder for folder in self.folders if folder['name'] not in matched_folder_names]
        
        logger.info(f"\n未匹配到Excel记录的文件夹数量: {len(unmatched_folders)}")
        
        # 如果有未匹配的文件夹，显示一些例子
        if unmatched_folders and len(unmatched_folders) <= 5:
            logger.info("未匹配的文件夹:")
            for folder in unmatched_folders:
                logger.info(f"  - 文件夹名称: {folder['name']}, 文件数量: {len(folder['files'])}")
        elif unmatched_folders:
            logger.info(f"前5个未匹配的文件夹:")
            for folder in unmatched_folders[:5]:
                logger.info(f"  - 文件夹名称: {folder['name']}, 文件数量: {len(folder['files'])}")
            logger.info(f"  ... 还有{len(unmatched_folders)-5}个未显示")
    
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
        # 避免处理明显不是项目文件夹的目录
        if folder_name.endswith('.py') or folder_name.endswith('.md') or folder_name.endswith('.xlsx'):
            return False
        return True


def main():
    """主函数"""
    # Excel文件路径
    excel_path = os.path.join(os.getcwd(), '5年历史档案（总规详规）清单.xlsx')
    output_path = os.path.join(os.getcwd(), '5年历史档案（总规详规）清单_严格文件夹关联.xlsx')
    
    # 创建关联器实例
    associator = StrictFolderAssociator(excel_path)
    
    # 加载Excel文件
    if not associator.load_excel():
        return
    
    # 扫描文件夹
    if not associator.scan_folders():
        logger.error("未找到有效文件夹，无法进行匹配")
        return
    
    # 执行文件夹关联
    associator.associate_folders()
    
    # 保存结果
    if not associator.save_results(output_path):
        return
    
    # 生成报告
    associator.generate_report()


if __name__ == "__main__":
    main()