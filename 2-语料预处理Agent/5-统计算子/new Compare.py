import os

input_dir = "D:\1 项目知识库\1.0 规资局项目\1-6 复兴岛语料攻坚\语料建设成果\域外数据加工仓库\第三批域外加工\word-207"
output_dir = "G:\合并转化后0707\WORD2PDF-203"

dst_fs = []
lost = []
for dst_f in os.listdir(output_dir):
    dst_fs.append(dst_f.replace("_kps_converted.pdf", ""))
for src_f in os.listdir(input_dir):
    if src_f not in dst_fs:
        lost.append(src_f)
        print(src_f)

print(len(lost))