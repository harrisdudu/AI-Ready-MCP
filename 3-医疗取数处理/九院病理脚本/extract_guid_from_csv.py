#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取CSV文件中花括号内的内容
功能：从指定CSV文件中提取所有花括号{}内的GUID内容
"""
import csv
import re
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVGUIDExtractor:
    """CSV文件中GUID提取器"""
    def __init__(self, input_file, output_file='extracted_guids.txt'):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.guid_pattern = re.compile(r'\{([0-9A-Fa-f-]+)\}')  # 精确匹配花括号内的GUID格式内容

    def extract_guids(self):
        """从CSV文件中提取所有GUID并保存到输出文件"""
        if not self.input_file.exists():
            logger.error(f"❌ 输入文件不存在: {self.input_file}")
            return False

        guids = set()  # 使用集合避免重复GUID

        try:
            with open(self.input_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                row_count = 0

                for row in reader:
                    row_count += 1
                    # 检查行中的每个单元格
                    for cell in row:
                        # 查找所有匹配的GUID
                        matches = self.guid_pattern.findall(cell)
                        for match in matches:
                            # 确保只保留GUID部分，移除可能的花括号
                            clean_guid = match.strip('{}')
                            guids.add(clean_guid)
                            logger.debug(f"找到GUID: {match} (行: {row_count})")

            logger.info(f"✅ 成功提取 {len(guids)} 个唯一GUID")

            # 保存提取的GUID
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for guid in sorted(guids):
                    f.write(f"{guid}\n")

            logger.info(f"📄 GUID已保存到: {self.output_file}")
            return True

        except Exception as e:
            logger.error(f"❌ 处理文件时出错: {str(e)}")
            return False

if __name__ == "__main__":
    # CSV文件路径
    input_csv = '合并文件_处理结果.csv'
    output_txt = 'extracted_guids.txt'

    extractor = CSVGUIDExtractor(input_csv, output_txt)
    extractor.extract_guids()