import argparse
import base64
import os
from datetime import datetime

import pandas as pd  # 导入pandas库用于Excel操作

import json

# 192.168.81.211


# endpoint: http: // 139.226.106.165: 9000
# accessKey: ellIA5BeEqexgvZR
# secretKey: MRf23KZ9lRTKAR4u97OSWud89ohDT65r

# 目标MinIO配置（用于保存图片）
TARGET_MINIO_ENDPOINT = "172.16.0.60:19000"  # 目标MinIO服务器地址
TARGET_MINIO_ACCESS_KEY = "APSPJIUsk66Diq3XTlAF"  # 目标访问密钥
TARGET_MINIO_SECRET_KEY = "eZBsbaqOeY1Nfpig1gZ5aCVSBnkIyTQseqgtN3hy"  # 目标秘密密钥
TARGET_MINIO_BUCKET = "kps"  # 目标bucket名称
TARGET_MINIO_SECURE = False  # 是否使用HTTPS

MAX_CELL_LENGTH = 32000

# JSON_PATH 将通过命令行参数传入

# TAGS
TAGS = [
    {
        "value": "anlilei",
        "label": "案例类",
        "children": [
            {
                "value": "fanganzhengji",
                "label": "方案征集",
                "children": [
                    {
                        "value": "xianggui",
                        "label": "详规",
                        "children": [
                            {
                                "value": "xianggui",
                                "label": "详规"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "youxiuanli",
                "label": "优秀案例",
                "children": [
                    {
                        "value": "youxiuanli",
                        "label": "优秀案例",
                        "children": [
                            {
                                "value": "youxiuanli",
                                "label": "优秀案例"
                            }
                        ]
                    }
                ]
            }
        ]
    },
    {
        "value": "guanlilei",
        "label": "管理类",
        "children": [
            {
                "value": "cehuidiaochachu",
                "label": "测绘调查处",
                "children": [
                    {
                        "value": "yewuguize",
                        "label": "业务规则",
                        "children": [
                            {
                                "value": "banshizhinan",
                                "label": "办事指南"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "chengjiandangan",
                "label": "城建档案",
                "children": [
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "banshizhinan",
                                "label": "办事指南"
                            },
                            {
                                "value": "feifawenyewuguize",
                                "label": "非发文业务规则"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "gengxinchu",
                "label": "更新处",
                "children": [
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "shuchengzhongxin/shuchengchu",
                "label": "数城中心/数城处",
                "children": [
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "xiangcunchu",
                "label": "乡村处",
                "children": [
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "xiangguichu",
                "label": "详规处",
                "children": [
                    {
                        "value": "yewuguize",
                        "label": "业务规则",
                        "children": [
                            {
                                "value": "feifawenyewuguize",
                                "label": "非发文业务规则"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "qitaanli",
                                "label": "其他案例"
                            }
                        ]
                    }
                ]
            }
        ]
    },
    {
        "value": "shiwulei",
        "label": "实务类",
        "children": [
            {
                "value": "cehuidili",
                "label": "测绘地理",
                "children": [
                    {
                        "value": "jishubiaozhun",
                        "label": "技术标准",
                        "children": [
                            {
                                "value": "difangbiaozhun",
                                "label": "地方标准"
                            },
                            {
                                "value": "difangguicheng",
                                "label": "地方规程"
                            },
                            {
                                "value": "guojiabiaozhun",
                                "label": "国家标准"
                            },
                            {
                                "value": "tuantibiaozhun",
                                "label": "团体标准"
                            },
                            {
                                "value": "hangyebiaozhun",
                                "label": "行业标准"
                            }
                        ]
                    },
                    {
                        "value": "jishushouce",
                        "label": "技术手册",
                        "children": [
                            {
                                "value": "jishushouce",
                                "label": "技术手册"
                            }
                        ]
                    },
                    {
                        "value": "qita",
                        "label": "其他",
                        "children": [
                            {
                                "value": "cehuixuanchuan",
                                "label": "测绘宣传"
                            },
                            {
                                "value": "cehuixuanchuanjiyingyonganli",
                                "label": "测绘宣传及应用案例"
                            },
                            {
                                "value": "qita",
                                "label": "其他"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "bumenguizhang",
                                "label": "部门规章"
                            },
                            {
                                "value": "difangxingfagui",
                                "label": "地方性法规"
                            },
                            {
                                "value": "difangzhengfuguizhang",
                                "label": "地方政府规章"
                            },
                            {
                                "value": "falv",
                                "label": "法律"
                            },
                            {
                                "value": "guowuyuanhebuweiguifanxingwenjian",
                                "label": "国务院和部委规范性文件"
                            },
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            },
                            {
                                "value": "xingzhengfagui",
                                "label": "行政法规"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "diaoyanbaogao",
                                "label": "调研报告"
                            },
                            {
                                "value": "huojiangxiangmu",
                                "label": "获奖项目"
                            },
                            {
                                "value": "keyanketi",
                                "label": "科研课题"
                            },
                            {
                                "value": "qitaanli",
                                "label": "其他案例"
                            },
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "chengjiandangan",
                "label": "城建档案",
                "children": [
                    {
                        "value": "bianyanchengguo",
                        "label": "编研成果",
                        "children": [
                            {
                                "value": "shipin",
                                "label": "视频"
                            },
                            {
                                "value": "tushu",
                                "label": "图书"
                            }
                        ]
                    },
                    {
                        "value": "jishubiaozhun",
                        "label": "技术标准",
                        "children": [
                            {
                                "value": "difangbiaozhun",
                                "label": "地方标准"
                            },
                            {
                                "value": "guojiabiaozhun",
                                "label": "国家标准"
                            },
                            {
                                "value": "hangyebiaozhun",
                                "label": "行业标准"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "bumenguizhang",
                                "label": "部门规章"
                            },
                            {
                                "value": "difangxingfagui",
                                "label": "地方性法规"
                            },
                            {
                                "value": "difangzhengfuguizhang",
                                "label": "地方政府规章"
                            },
                            {
                                "value": "falv",
                                "label": "法律"
                            },
                            {
                                "value": "guowuyuanhebuweiguifanxingwenjian",
                                "label": "国务院和部委规范性文件"
                            },
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            },
                            {
                                "value": "xingzhengfagui",
                                "label": "行政法规"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "gongzhongxuanchuan",
                "label": "公众宣传",
                "children": [
                    {
                        "value": "qita",
                        "label": "其他",
                        "children": [
                            {
                                "value": "qita",
                                "label": "其他"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "guihuasheji",
                "label": "规划设计",
                "children": [
                    {
                        "value": "jishubiaozhun",
                        "label": "技术标准",
                        "children": [
                            {
                                "value": "difangbiaozhun",
                                "label": "地方标准"
                            },
                            {
                                "value": "difangguicheng",
                                "label": "地方规程"
                            },
                            {
                                "value": "guojiabiaozhun",
                                "label": "国家标准"
                            },
                            {
                                "value": "tuantibiaozhun",
                                "label": "团体标准"
                            },
                            {
                                "value": "hangyebiaozhun",
                                "label": "行业标准"
                            }
                        ]
                    },
                    {
                        "value": "jishushouce",
                        "label": "技术手册",
                        "children": [
                            {
                                "value": "jishushouce",
                                "label": "技术手册"
                            }
                        ]
                    },
                    {
                        "value": "qita",
                        "label": "其他",
                        "children": [
                            {
                                "value": "peixunkejian",
                                "label": "培训课件"
                            },
                            {
                                "value": "qita",
                                "label": "其他"
                            }
                        ]
                    },
                    {
                        "value": "shujiqikan",
                        "label": "书籍期刊",
                        "children": [
                            {
                                "value": "shujiqikan",
                                "label": "书籍期刊"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "bumenguizhang",
                                "label": "部门规章"
                            },
                            {
                                "value": "difangxingfagui",
                                "label": "地方性法规"
                            },
                            {
                                "value": "difangzhengfuguizhang",
                                "label": "地方政府规章"
                            },
                            {
                                "value": "falv",
                                "label": "法律"
                            },
                            {
                                "value": "guowuyuanhebuweiguifanxingwenjian",
                                "label": "国务院和部委规范性文件"
                            },
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            },
                            {
                                "value": "xingzhengfagui",
                                "label": "行政法规"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "huojiangxiangmu",
                                "label": "获奖项目"
                            },
                            {
                                "value": "keyanketi",
                                "label": "科研课题"
                            },
                            {
                                "value": "qitaanli",
                                "label": "其他案例"
                            },
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "shizhenggongcheng",
                "label": "市政工程",
                "children": [
                    {
                        "value": "jishubiaozhun",
                        "label": "技术标准",
                        "children": [
                            {
                                "value": "difangbiaozhun",
                                "label": "地方标准"
                            },
                            {
                                "value": "difangguicheng",
                                "label": "地方规程"
                            },
                            {
                                "value": "guojiabiaozhun",
                                "label": "国家标准"
                            },
                            {
                                "value": "tuantibiaozhun",
                                "label": "团体标准"
                            },
                            {
                                "value": "hangyebiaozhun",
                                "label": "行业标准"
                            }
                        ]
                    },
                    {
                        "value": "jishushouce",
                        "label": "技术手册",
                        "children": [
                            {
                                "value": "jishushouce",
                                "label": "技术手册"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "bumenguizhang",
                                "label": "部门规章"
                            },
                            {
                                "value": "difangxingfagui",
                                "label": "地方性法规"
                            },
                            {
                                "value": "difangzhengfuguizhang",
                                "label": "地方政府规章"
                            },
                            {
                                "value": "falv",
                                "label": "法律"
                            },
                            {
                                "value": "guowuyuanhebuweiguifanxingwenjian",
                                "label": "国务院和部委规范性文件"
                            },
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            },
                            {
                                "value": "xingzhengfagui",
                                "label": "行政法规"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "huojiangxiangmu",
                                "label": "获奖项目"
                            },
                            {
                                "value": "keyanketi",
                                "label": "科研课题"
                            },
                            {
                                "value": "qitaanli",
                                "label": "其他案例"
                            },
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            },
            {
                "value": "ziranziyuan",
                "label": "自然资源",
                "children": [
                    {
                        "value": "jishubiaozhun",
                        "label": "技术标准",
                        "children": [
                            {
                                "value": "difangbiaozhun",
                                "label": "地方标准"
                            },
                            {
                                "value": "difangguicheng",
                                "label": "地方规程"
                            },
                            {
                                "value": "guojiabiaozhun",
                                "label": "国家标准"
                            },
                            {
                                "value": "hangyebiaozhun",
                                "label": "行业标准"
                            }
                        ]
                    },
                    {
                        "value": "jishushouce",
                        "label": "技术手册",
                        "children": [
                            {
                                "value": "jishushouce",
                                "label": "技术手册"
                            }
                        ]
                    },
                    {
                        "value": "qita",
                        "label": "其他",
                        "children": [
                            {
                                "value": "qita",
                                "label": "其他"
                            }
                        ]
                    },
                    {
                        "value": "shujiqikan",
                        "label": "书籍期刊",
                        "children": [
                            {
                                "value": "shujiqikan",
                                "label": "书籍期刊"
                            }
                        ]
                    },
                    {
                        "value": "zhengcefagui",
                        "label": "政策法规",
                        "children": [
                            {
                                "value": "bumenguizhang",
                                "label": "部门规章"
                            },
                            {
                                "value": "difangxingfagui",
                                "label": "地方性法规"
                            },
                            {
                                "value": "difangzhengfuguizhang",
                                "label": "地方政府规章"
                            },
                            {
                                "value": "falv",
                                "label": "法律"
                            },
                            {
                                "value": "guowuyuanhebuweiguifanxingwenjian",
                                "label": "国务院和部委规范性文件"
                            },
                            {
                                "value": "qitaguifanxingwenjian",
                                "label": "其他规范性文件"
                            },
                            {
                                "value": "xingzhengfagui",
                                "label": "行政法规"
                            }
                        ]
                    },
                    {
                        "value": "zhuanyeanli",
                        "label": "专业案例",
                        "children": [
                            {
                                "value": "huojiangxiangmu",
                                "label": "获奖项目"
                            },
                            {
                                "value": "keyanketi",
                                "label": "科研课题"
                            },
                            {
                                "value": "qitaanli",
                                "label": "其他案例"
                            },
                            {
                                "value": "yanjiubaogao",
                                "label": "研究报告"
                            }
                        ]
                    }
                ]
            }
        ]
    }
]


def process_json_data_to_excel_data(json_data, object_name, output_dir_path):
    """
    处理JSON数据并返回Excel数据
    
    Args:
        json_data: JSON数据字典
        object_name: 源文件对象名称（用于记录）
    
    Returns:
        处理后的Excel数据字典和部门名称
    """
    # 获取body部分
    body = json_data.get('body', {})
    department = body.get('department', '未知部门')
    institution = body.get('institution', '未知机构')
    qas = json_data.get('qas', {})
    cid = json_data.get('header', {}).get('cid', '')
    pid = json_data.get('header', {}).get('pid', '')
    filename = json_data.get('header', {}).get('extra_info', {}).get('title', '')
    version = json_data.get('header', {}).get('version', '')
    default_tag = body.get('defaultTag', [])
    content = body.get('content', [])

    if version != 'V1.1':
        print(f'{object_name} 是{version}')
        return None, None, None, None, None

    # 准备Excel数据
    excel_data = {}

    page_img_map = {}

    md_content = ""
    index = 0
    # 处理content部分
    for item in content:
        item_type = item.get('type', '')
        if item_type == 'title':
            # 处理标题类型
            level = item.get('level', 1)
            text = item.get('text', '')
            md_content += f"{'#' * level} {text}\n\n"
        elif item_type == 'text':
            # 处理文本类型
            text = item.get('text', '')
            md_content += f"{text}\n\n"
        elif item_type == 'table':
            # 处理表格类型
            text = item.get('text', '')
            md_content += f"{text}\n\n"
        elif item_type == 'image':
            # 处理图片类型
            img = item.get('img', '')
            page_no = item.get('page_no', '')
            content = item.get('content', f'img_{page_no}_{index}')
            index += 1
            if content and img:
                # 保存图片到本地
                local_path = save_base64_to_local(content, cid, img, output_dir=output_dir_path)
                if local_path:
                    minio_url = get_target_minio_public_url(f"{cid}/{img}")
                    if page_no not in page_img_map:
                        page_img_map[page_no] = []
                    if local_path not in page_img_map[page_no]:
                        page_img_map[page_no].append(local_path)
                    # 保持item中的img字段为原始URL（不修改）
                    # 添加到Markdown，使用原始URL
                    item['img'] = minio_url
                    md_content += f"![{img}]({minio_url})\n\n"

    # 保存最后的内容
    if md_content:
        excel_data[f'md'] = md_content

    for qa in qas:
        if qa.get('with_img', False):
            image_names_list = qa.get('image_names', [])
            if  isinstance(image_names_list, str):
                image_names_list = image_names_list.split(',')
            tmp_list = []
            for image_name in image_names_list:
                object_name = f"{cid}/{image_name}"
                tmp_list.append(get_target_minio_public_url(object_name))
            print(tmp_list)
            qa['imgs'] = ",".join(tmp_list)
        else:
            qa['imgs'] = ""

    excel_data['tags'] = json.dumps(TAGS, ensure_ascii=False)
    excel_data['defaultTag'] = json.dumps(default_tag, ensure_ascii=False)
    excel_data['qas'] = json.dumps(qas, ensure_ascii=False)
    excel_data['filename'] = filename
    return excel_data, department, institution, version, pid


def list_json_files_from_path(json_path):
    """
    从本地目录中列出所有JSON文件

    Args:
        json_path: JSON文件所在的目录路径

    Returns:
        JSON文件路径列表
    """
    json_files = []
    # 查询在指定路径下的所有JSON文件（包括二级文件夹）
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    return json_files


def process_json_files_to_local(output_dir_path, json_path='./'):
    """
    处理本地JSON文件并保存到本地目录
    
    Args:
        output_dir_path: 输出文件的目录路径
        json_path: JSON文件所在的目录路径
    """
    # 获取文件夹目录下所有json文件
    json_files = list_json_files_from_path(json_path)

    if not json_files:
        print(f"在系统目录下未找到JSON文件")
        return

    # 确保输出目录存在
    os.makedirs(output_dir_path, exist_ok=True)

    # 创建统计信息列表
    stats_data = []

    # 处理每个JSON文件
    for json_obj in json_files:
        # 检查是否已处理

        try:
            # 读取JSON数据
            with open(json_obj, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            if not json_data:
                continue

            # 处理JSON数据
            excel_data, department, institution, version, pid = process_json_data_to_excel_data(json_data, json_obj,
                                                                                                output_dir_path)

            if excel_data is None or department is None or institution is None or version is None:
                continue

            department_institution = department + "_" + institution

            # 处理部门名称，确保可以用作文件名
            safe_department = department_institution.replace('/', '_').replace('\\', '_').replace('?', '_').replace('*',
                                                                                                                    '_').replace(
                '[', '_').replace(']', '_').replace(':', '_')
            if not safe_department:
                safe_department = "未知部门"

            json_dir = f"{pid}_{safe_department}"  # 目标文件夹路径

            # 将excel_data保存为JSON到本地
            try:
                # 构造本地目标路径：output_dir/json_dir/原JSON文件名
                local_json_dir = os.path.join(output_dir_path, json_dir)
                os.makedirs(local_json_dir, exist_ok=True)

                target_json_path = os.path.join(local_json_dir, os.path.basename(json_obj))

                # 将字典转换为JSON并保存到本地文件
                with open(target_json_path, 'w', encoding='utf-8') as f:
                    json.dump(excel_data, f, ensure_ascii=False, indent=2)

                save_success = True
                print(f"JSON数据已保存到本地: {target_json_path}")
            except Exception as e:
                print(f"保存JSON数据失败: {e}")
                save_success = False

            # 记录处理信息（修改路径为本地保存的JSON路径）
            process_info = f"已处理文件: {json_obj} (部门: {department}) - 已保存到本地: {target_json_path}"
            print(process_info)

            # 添加到统计数据（修改为本地保存路径）
            stats_data.append({
                "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "JSON文件": json_obj,
                "公司": institution,
                "部门": department,
                "本地路径": target_json_path if save_success else "保存失败"
            })

        except Exception as e:
            error_msg = f"处理文件 {json_obj} 时出错: {e}"
            print(error_msg)
            # 记录错误信息到统计数据
            stats_data.append({
                "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "JSON文件": json_obj,
                "公司": "处理失败",
                "部门": "处理失败",
                "本地路径": "保存失败"
            })

    # 将统计信息写入CSV文件，使用UTF-8编码
    stats_file_path = os.path.join(output_dir_path, "脚本统计.csv")
    pd.DataFrame(stats_data).to_csv(stats_file_path, index=False, encoding='utf-8')

    print(f"处理完成，共处理 {len(json_files)} 个文件")
    print(f"统计信息已保存到本地: {stats_file_path}")


def save_base64_to_local(base64_data, cid, file_name, output_dir="./images"):
    """
    将base64编码的图片保存到本地
    
    Args:
        base64_data: base64编码的图片数据
        cid: 文档的CID，用作本地目录
        file_name: 文件名
        output_dir: 输出目录
    
    Returns:
        本地文件路径
    """
    try:
        # 从base64解码图片数据
        image_data = base64.b64decode(base64_data)

        # 生成本地文件路径
        local_pic = os.path.join(output_dir, cid, file_name)
        os.makedirs(os.path.dirname(local_pic), exist_ok=True)

        # 将图片保存到本地
        with open(local_pic, 'wb') as f:
            f.write(image_data)

        return local_pic

    except Exception as e:
        print(f"保存图片到本地时发生错误: {e}")
        return None


def get_target_minio_public_url(object_name):
    """
    根据对象名称获取目标MinIO公开访问链接
    
    Args:
        object_name: 对象名称（包括路径，例如：'cid/image.png'）
    
    Returns:
        公开访问URL
    """
    # 构建公开访问URL
    # 格式通常为：http(s)://endpoint/bucket/object_name

    url = f"http://192.168.81.211:19000/{TARGET_MINIO_BUCKET}/{object_name}"

    return url


def main(args):
    """
    主函数，处理命令行参数并执行程序
    
    Args:
        args: 命令行参数
    """
    # 输出目录路径
    output_dir_path = r"./annotation0715"

    # 处理JSON文件并保存到本地
    process_json_files_to_local(output_dir_path, args.json_path)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='处理JSON文件并生成Excel文件')
    parser.add_argument('--json-path', type=str, default='./', help='指定JSON文件所在的目录路径，默认为当前目录')

    # 解析命令行参数
    args = parser.parse_args()

    print(f"开始处理目录: {args.json_path} 下的所有JSON文件")
    # 调用主函数，传入参数
    main(args)
