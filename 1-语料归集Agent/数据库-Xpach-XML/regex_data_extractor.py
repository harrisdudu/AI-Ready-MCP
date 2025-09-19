import re
import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

class RegexDataExtractor:
    """正则表达式数据提取器，用于从文本中提取结构化数据"""

    def __init__(self):
        self.patterns = {}

    def add_pattern(self, name: str, pattern: str) -> None:
        """添加提取模式

        Args:
            name: 提取字段名称
            pattern: 正则表达式模式
        """
        self.patterns[name] = pattern

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取数据

        Args:
            text: 输入文本

        Returns:
            提取的数据字典
        """
        result = {}
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # 如果只有一个匹配项且是元组，取第一个元素
                if len(matches) == 1 and isinstance(matches[0], tuple):
                    result[name] = matches[0][0].strip()
                # 如果是多个匹配项
                else:
                    result[name] = [m.strip() if isinstance(m, tuple) else m for m in matches]
            else:
                result[name] = None
        return result

    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """从文件中提取数据

        Args:
            file_path: 文件路径

        Returns:
            提取的数据字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.extract_from_text(text)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return {}

    def save_results(self, results: Dict[str, Any], output_file: str) -> bool:
        """保存提取结果到文件

        Args:
            results: 提取结果
            output_file: 输出文件路径

        Returns:
            是否保存成功
        """
        try:
            # 根据文件扩展名确定保存格式
            ext = os.path.splitext(output_file)[1].lower()
            if ext == '.json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
            elif ext == '.csv':
                # 转换为DataFrame并保存
                df = pd.DataFrame([results])
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                # 默认保存为文本文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    for k, v in results.items():
                        f.write(f"{k}: {v}\n")
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False

    def batch_extract(self, input_dir: str, output_dir: str) -> None:
        """批量提取目录中所有文件的数据

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path) and file_path.endswith(('.txt', '.md', '.csv')):
                print(f"处理文件: {filename}")
                results = self.extract_from_file(file_path)
                output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_extracted.json")
                self.save_results(results, output_file)
                print(f"结果已保存到: {output_file}")

# 医疗数据提取示例
if __name__ == '__main__':
    # 创建提取器实例
    extractor = RegexDataExtractor()

    # 添加医疗数据常用提取模式
    extractor.add_pattern('patient_name', r'患者姓名[:：]\s*([\u4e00-\u9fa5]+)')
    extractor.add_pattern('gender', r'性别[:：]\s*([男女])')
    extractor.add_pattern('age', r'年龄[:：]\s*([0-9]+)')
    extractor.add_pattern('visit_number', r'就诊号[:：]\s*([A-Za-z0-9]+)')
    extractor.add_pattern('diagnosis', r'诊断[:：]\s*([\u4e00-\u9fa5、,；;]+)')
    extractor.add_pattern('treatment', r'治疗[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
    extractor.add_pattern('medicine', r'药物[:：]\s*([\u4e00-\u9fa5]+(?:片|胶囊|注射液|颗粒))')
    extractor.add_pattern('date', r'(\d{4}-\d{2}-\d{2})')

    # 示例文本
    sample_text = """
    患者信息：
    患者姓名：张三
    性别：男
    年龄：45岁
    就诊号：B000831020
    诊断：高血压、糖尿病
    治疗：口服药物治疗
    药物：降压片、降糖胶囊
    就诊日期：2024-05-15
    """

    # 从文本提取数据
    results = extractor.extract_from_text(sample_text)
    print("提取结果:")
    print(json.dumps(results, ensure_ascii=False, indent=4))

    # 保存结果到JSON文件
    extractor.save_results(results, 'patient_data_extracted.json')
    print("结果已保存到 patient_data_extracted.json")

    # 如需批量处理文件，可使用以下代码
    # extractor.batch_extract('input_files', 'output_results')