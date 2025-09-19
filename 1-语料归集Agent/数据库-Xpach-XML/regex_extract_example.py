import re
import json
from regex_data_extractor import RegexDataExtractor

# 创建一个自定义的医疗数据提取器类，继承自RegexDataExtractor
class MedicalDataExtractor(RegexDataExtractor):
    """医疗数据提取器，扩展自通用正则表达式提取器"""

    def __init__(self):
        super().__init__()
        # 添加医疗领域常用的提取模式
        self._add_medical_patterns()

    def _add_medical_patterns(self):
        """添加医疗领域专用的正则表达式模式"""
        # 患者基本信息
        self.add_pattern('patient_name', r'患者姓名[:：]\s*([\u4e00-\u9fa5]+)')
        self.add_pattern('gender', r'性别[:：]\s*([男女])')
        self.add_pattern('age', r'年龄[:：]\s*([0-9]+)')
        self.add_pattern('birth_date', r'出生日期[:：]\s*(\d{4}-\d{2}-\d{2})')
        self.add_pattern('id_card', r'身份证号[:：]\s*(\d{17}[\dXx])')
        self.add_pattern('visit_number', r'就诊号[:：]\s*([A-Za-z0-9]+)')
        self.add_pattern('department', r'科室[:：]\s*([\u4e00-\u9fa5]+)')
        self.add_pattern('doctor', r'医生[:：]\s*([\u4e00-\u9fa5]+)')

        # 诊断相关
        self.add_pattern('main_diagnosis', r'主要诊断[:：]\s*([\u4e00-\u9fa5、,；;]+)')
        self.add_pattern('secondary_diagnosis', r'次要诊断[:：]\s*([\u4e00-\u9fa5、,；;]+)')
        self.add_pattern('diagnosis_basis', r'诊断依据[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('differential_diagnosis', r'鉴别诊断[:：]\s*([\u4e00-\u9fa5，。,;；]+)')

        # 症状和病史
        self.add_pattern('chief_complaint', r'主诉[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('present_illness', r'现病史[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('past_history', r'既往史[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('personal_history', r'个人史[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('family_history', r'家族史[:：]\s*([\u4e00-\u9fa5，。,;；]+)')

        # 体格检查
        self.add_pattern('physical_exam', r'体格检查[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('blood_pressure', r'血压[:：]\s*([0-9]+/[0-9]+)\s*mmHg')
        self.add_pattern('heart_rate', r'心率[:：]\s*([0-9]+)\s*次/分')
        self.add_pattern('temperature', r'体温[:：]\s*([0-9.]+)\s*°C')

        # 实验室检查
        self.add_pattern('blood_test', r'血常规[:：]\s*([\u4e00-\u9fa5，。,;；0-9.]+)')
        self.add_pattern('biochemistry', r'生化检查[:：]\s*([\u4e00-\u9fa5，。,;；0-9.]+)')
        self.add_pattern('imaging', r'(CT|MRI|超声|X线)[:：]\s*([\u4e00-\u9fa5，。,;；]+)')

        # 治疗相关
        self.add_pattern('treatment_plan', r'治疗计划[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
        self.add_pattern('medication', r'药物治疗[:：]\s*([\u4e00-\u9fa5]+(?:片|胶囊|注射液|颗粒))\s*(\d+[mg|g|ml]+)\s*(\d+次/日)')
        self.add_pattern('surgery', r'手术[:：]\s*([\u4e00-\u9fa5，。,;；]+)')

        # 日期相关
        self.add_pattern('visit_date', r'就诊日期[:：]\s*(\d{4}-\d{2}-\d{2})')
        self.add_pattern('admission_date', r'入院日期[:：]\s*(\d{4}-\d{2}-\d{2})')
        self.add_pattern('discharge_date', r'出院日期[:：]\s*(\d{4}-\d{2}-\d{2})')

    def extract_medical_data(self, text: str) -> dict:
        """提取医疗数据并进行整理

        Args:
            text: 医疗文本

        Returns:
            整理后的医疗数据字典
        """
        # 基础提取
        raw_data = self.extract_from_text(text)

        # 数据清洗和整理
        cleaned_data = {
            'patient_info': {
                'name': raw_data.get('patient_name'),
                'gender': raw_data.get('gender'),
                'age': raw_data.get('age'),
                'birth_date': raw_data.get('birth_date'),
                'id_card': raw_data.get('id_card'),
                'visit_number': raw_data.get('visit_number'),
                'department': raw_data.get('department'),
                'doctor': raw_data.get('doctor'),
            },
            'diagnosis': {
                'main_diagnosis': raw_data.get('main_diagnosis'),
                'secondary_diagnosis': raw_data.get('secondary_diagnosis'),
                'diagnosis_basis': raw_data.get('diagnosis_basis'),
                'differential_diagnosis': raw_data.get('differential_diagnosis'),
            },
            'medical_history': {
                'chief_complaint': raw_data.get('chief_complaint'),
                'present_illness': raw_data.get('present_illness'),
                'past_history': raw_data.get('past_history'),
                'personal_history': raw_data.get('personal_history'),
                'family_history': raw_data.get('family_history'),
            },
            'physical_exam': {
                'general': raw_data.get('physical_exam'),
                'blood_pressure': raw_data.get('blood_pressure'),
                'heart_rate': raw_data.get('heart_rate'),
                'temperature': raw_data.get('temperature'),
            },
            'tests': {
                'blood_test': raw_data.get('blood_test'),
                'biochemistry': raw_data.get('biochemistry'),
                'imaging': raw_data.get('imaging'),
            },
            'treatment': {
                'plan': raw_data.get('treatment_plan'),
                'medication': raw_data.get('medication'),
                'surgery': raw_data.get('surgery'),
            },
            'dates': {
                'visit_date': raw_data.get('visit_date'),
                'admission_date': raw_data.get('admission_date'),
                'discharge_date': raw_data.get('discharge_date'),
            }
        }

        return cleaned_data

# 示例使用
if __name__ == '__main__':
    # 创建医疗数据提取器实例
    medical_extractor = MedicalDataExtractor()

    # 示例医疗文本
    sample_medical_text = """
    患者信息：
    患者姓名：李四
    性别：女
    年龄：65
    出生日期：1959-03-12
    身份证号：110101195903121234
    就诊号：B000831021
    科室：内科
    医生：王医生

    主诉：反复头晕头痛3年，加重1周。
    现病史：患者3年前无明显诱因出现头晕头痛，呈持续性胀痛，无恶心呕吐，无视物旋转，未予重视。1周前上述症状加重，伴乏力、纳差。
    既往史：高血压病史5年，最高血压160/100mmHg，规律服用降压药。糖尿病病史2年，血糖控制可。
    个人史：无吸烟饮酒史。
    家族史：父亲患有高血压。

    体格检查：T 36.5°C，P 78次/分，R 18次/分，BP 150/95mmHg。神志清楚，精神可，全身皮肤黏膜无黄染及出血点，浅表淋巴结未触及肿大。心肺腹未见明显异常。神经系统检查未见阳性体征。

    辅助检查：血常规：白细胞计数5.6×10^9/L，红细胞计数4.5×10^12/L，血红蛋白130g/L，血小板计数220×10^9/L。
    生化检查：血糖5.8mmol/L，总胆固醇4.8mmol/L，甘油三酯1.6mmol/L，肌酐78μmol/L。
    头颅CT：未见明显异常。

    诊断：
    主要诊断：高血压2级（很高危）
    次要诊断：2型糖尿病
    诊断依据：患者有高血压病史5年，最高血压160/100mmHg，规律服用降压药，目前血压150/95mmHg；有糖尿病病史2年，血糖控制可。
    鉴别诊断：需与继发性高血压、脑血管疾病等相鉴别。

    治疗计划：
    1. 调整降压药物：氨氯地平片5mg 1次/日，缬沙坦胶囊80mg 1次/日。
    2. 控制血糖：继续服用二甲双胍片0.5g 2次/日。
    3. 生活方式干预：低盐低脂饮食，适量运动，戒烟限酒。
    4. 定期监测血压、血糖。

    就诊日期：2024-05-20
    """

    # 提取医疗数据
    medical_data = medical_extractor.extract_medical_data(sample_medical_text)

    # 打印提取结果
    print("提取的医疗数据:")
    print(json.dumps(medical_data, ensure_ascii=False, indent=4))

    # 保存结果到文件
    with open('extracted_medical_data.json', 'w', encoding='utf-8') as f:
        json.dump(medical_data, f, ensure_ascii=False, indent=4)

    print("\n结果已保存到 extracted_medical_data.json 文件。")

    # 展示如何添加自定义模式
    print("\n添加自定义提取模式示例:")
    medical_extractor.add_pattern('allergic_history', r'过敏史[:：]\s*([\u4e00-\u9fa5，。,;；]+)')
    sample_with_allergy = sample_medical_text + "\n过敏史：对青霉素过敏。"
    allergy_data = medical_extractor.extract_from_text(sample_with_allergy)
    print(f"过敏史: {allergy_data.get('allergic_history')}")