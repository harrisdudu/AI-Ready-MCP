import os

#input_dir = r"C:\域外第二批\word"
#output_dir = r"E:\word2pdf-0702"

input_dir = r"D:\1 项目知识库\1.0 规资局项目\1-6 复兴岛语料攻坚\语料建设成果\域外数据加工仓库\第三批域外加工\word-207"
output_dir = r"G:\合并转化后0707\WORD2PDF-203"
pdfs = []

for f in os.listdir(output_dir):
    pdfs.append(f.split("_kps_c")[0])

with open(os.path.join(output_dir, "failed_word-0707.txt"), "w", encoding="utf-8") as log: 
    for f in os.listdir(input_dir):
        if os.path.splitext(f)[0] not in pdfs:
            log.write(f + "\n")