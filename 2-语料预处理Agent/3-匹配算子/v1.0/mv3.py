import pandas as pd
import os 
import os.path as osp
import shutil
import openpyxl
import re
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Thread: %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Excel 文件路径
file_path = r"D:\2 技术资料知识库\Code Rep\特殊处理\0820-书单-加采集单位.xlsx"
sheet_name = "对照表0820"

# 读取指定 sheet 的前3列
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[0, 2])

# 构建字典：第一列为 key，第二、三列为 value 的元组
result_dict = {
    str(row[0]): row[1]
    for row in df.itertuples(index=False, name=None)
}


excel_path = r"D:\2 技术资料知识库\Code Rep\特殊处理\guizi-all.xlsx"
# 打开工作簿
wb = openpyxl.load_workbook(excel_path)   #, data_only=True)

# 获取指定 sheet
sheet = wb['对照表']

cid2id = {}

# 遍历每一行（从第2行开始，忽略表头）
for i, row in enumerate(sheet.iter_rows(min_row=2)):
    # if i == 10819: 
    # print(row)
    # p_cells = []
    # c_cells = []
    # t_cells = []


    # 从第二列开始读取（忽略第一列）
    # for cell in row[1:]:
        # if isinstance(cell.value, str):
        #     if "16795" in cell.value: raise ValueError(cell.value)
        #   value = str(cell.value).strip() if cell.value is not None else ''
        # value = str(cell.value) if cell.value is not None else ''
       
    """  if value.startswith('P'):
            p_cells.append(value)
        elif value.startswith('C'):
            c_cells.append(value)
        elif value.startswith('T'):
            t_cells.append(value) """
    try:
        cid = row[1].value
        # cid = f"{p_cells[0]}_{c_cells[0]}_{t_cells[0].split(',')[0]}" 
        # idx = re.match(r'^\d+', row[0].value.strip()).group()
        idx = re.match(r'^\d+', row[0].value.strip()).group()
        # 打印或处理提取的结果
        cid2id[cid] = idx
    except Exception as e:
        print(row[0].value)

        continue

mvdirs = set()
mvfiles = []

for root, dnames, fnames in os.walk(r"F:/2-成品语料库/规资-语料成果-json/解压后洗料"):
    for fname in fnames:
        if fname.endswith("_final.json") and fname.replace("_final.json", "") in cid2id:
            idx = cid2id[fname.replace("_final.json", "")]

            if idx in result_dict:
                institution = result_dict[idx]
                dst = osp.join(osp.dirname(osp.abspath(__file__)), institution, fname)
                os.makedirs(osp.dirname(dst), exist_ok=True)
                
                shutil.copyfile(osp.join(root, fname), dst)
                
                logger.info(f"已处理文件: {fname} -> {institution}")
            else:
                logger.warning(f"未找到匹配的机构: {idx}")
    
                logger.info(f"\n下载任务完成！共处理{len(fnames)}个文件，成功{len(fnames)}个，失败{0}个") 

    
    # mvdirs.add(root)
    # mvfiles.append(osp.join(root, fname))


