'''
cd langextract
pip install -e .
cd ..
python langextract_test.py
'''


import langextract.langextract as lx
import textwrap
from langextract.factory import ModelConfig

# 1. Define the prompt and extraction rules for knowledge graph construction
prompt = textwrap.dedent("""\
    从文本中抽取知识图谱三元组（实体-关系-实体）。
    要求：
    1. 识别文本中的各种实体，不限定具体类型，根据文本内容灵活判断
    2. 识别实体之间的各种关系，不限定关系类型，根据语义自然提取
    3. 使用精确的文本片段，不要改写或重叠实体
    4. 为每个实体和关系提供有意义的属性以增加上下文
    5. 确保三元组的完整性和准确性
    6. 尽可能全面地提取文本中的知识关系
    
    抽取原则：
    - 实体可以是任何有意义的对象：人物、地点、组织、概念、事件、产品、技术、方法等
    - 关系可以是任何语义关联：属于、位于、参与、创建、使用、影响、包含、相似、依赖、导致、拥有、管理、研究、开发、采用、部署等
    - 根据文本的具体内容和语境灵活判断实体类型和关系类型
    - 优先提取明确、重要的知识关系""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="张三在北京大学计算机系工作，他参与了人工智能项目，该项目使用了深度学习技术。",
        extractions=[
            lx.data.Extraction(
                extraction_class="人物",
                extraction_text="张三",
                attributes={"context": "工作于北京大学", "role": "员工"}
            ),
            lx.data.Extraction(
                extraction_class="机构", 
                extraction_text="北京大学计算机系",
                attributes={"context": "教育机构", "field": "计算机科学"}
            ),
            lx.data.Extraction(
                extraction_class="项目",
                extraction_text="人工智能项目",
                attributes={"context": "技术研发", "domain": "AI"}
            ),
            lx.data.Extraction(
                extraction_class="技术",
                extraction_text="深度学习技术",
                attributes={"context": "AI技术", "category": "机器学习"}
            ),
            lx.data.Extraction(
                extraction_class="工作于",
                extraction_text="张三在北京大学计算机系工作",
                attributes={"source": "张三", "target": "北京大学计算机系", "position": "员工"}
            ),
            lx.data.Extraction(
                extraction_class="参与",
                extraction_text="他参与了人工智能项目",
                attributes={"source": "张三", "target": "人工智能项目", "role": "参与者"}
            ),
            lx.data.Extraction(
                extraction_class="使用",
                extraction_text="该项目使用了深度学习技术",
                attributes={"source": "人工智能项目", "target": "深度学习技术", "purpose": "技术实现"}
            ),
        ]
    )
]

# The input text to be processed
input_text = "李四是清华大学软件学院的教授，他带领团队开发了一个基于机器学习的推荐系统，该系统被阿里巴巴公司采用并部署在电商平台上。"


config = ModelConfig(
    model_id="ark:doubao-seed-1-6-250615",
    provider_kwargs={
        "api_key": "c702676e-a69f-4ff0-a672-718d0d4723ed",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3"
    }
)

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    config=config,
    temperature=0.0,
    fence_output=False,
    use_schema_constraints=True
)

# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

