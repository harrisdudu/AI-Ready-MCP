import os
import os.path as osp
import time
import pythoncom
import win32com.client
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def office_to_pdf(input_path, output_path=None, retry=1):
    input_path = os.path.abspath(input_path)
    ext = osp.splitext(input_path)[1].lower()
    if output_path is None:
        output_path = osp.splitext(input_path)[0] + ".pdf"
    else:
        output_path = os.path.abspath(output_path)

    for attempt in range(retry):
        app = None
        doc = None
        try:
            pythoncom.CoInitialize()

            if ext in ('.doc', '.docx'):
                app = win32com.client.DispatchEx("Word.Application")
                app.Visible = False
                app.DisplayAlerts = 0
                doc = app.Documents.Open(input_path, ReadOnly=1)
                doc.ExportAsFixedFormat(
                    OutputFileName=output_path,
                    ExportFormat=17,  # PDF
                    OpenAfterExport=False,
                    OptimizeFor=0,
                    CreateBookmarks=1
                )

            elif ext in ('.xls', '.xlsx'):
                app = win32com.client.DispatchEx("Excel.Application")
                app.Visible = False
                app.DisplayAlerts = False
                doc = app.Workbooks.Open(input_path, ReadOnly=1)
                doc.ExportAsFixedFormat(0, output_path)  # 0 = PDF

            elif ext in ('.ppt', '.pptx','.ppsx'):
                app = win32com.client.DispatchEx("PowerPoint.Application")
                # app.Visible = False
                doc = app.Presentations.Open(input_path, WithWindow=False)
                doc.SaveAs(output_path, 32)  # 32 = PDF

            else:
                print(f"[⚠️ 不支持的文件类型] {input_path}")
                return False

            return True  # 成功

        except Exception as e:
            print(f"[⚠️ retry {attempt+1}] 转换失败: {input_path}\n  错误: {e}")
            time.sleep(1)
        finally:
            try:
                if doc:
                    doc.Close()
                if app:
                    app.Quit()
                pythoncom.CoUninitialize()
            except Exception as e:
                print(f"[⚠️ 清理失败] {input_path}\n  错误: {e}")

    print(f"[❌ 最终失败] {input_path}")
    return False

# 多进程任务封装
def convert_task(args):
    input_path, output_path = args
    return office_to_pdf(input_path, output_path)

if __name__ == "__main__":
    input_dir = r"D:\1 项目知识库\1.0 规资局项目\1-6 复兴岛语料攻坚\语料建设成果\域外数据加工仓库\ppt2pdf-71"
    output_dir = r"D:\1 项目知识库\1.0 规资局项目\1-6 复兴岛语料攻坚\语料建设成果\域外数据加工仓库\PPT2PDF"
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    supported_ext = ('.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx','.ppsx')

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(supported_ext):
            input_path = osp.join(input_dir, fname)
            output_path = osp.join(output_dir, osp.splitext(fname)[0] + "_kps_converted.pdf")
            if not osp.exists(output_path):
                tasks.append((input_path, output_path))

    print(f"共需转换文件数: {len(tasks)}")

    # max_workers = min(4, max(1, cpu_count() // 2))
    max_workers = 4
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap_unordered(convert_task, tasks), total=len(tasks)))

    failed = [tasks[i][0] for i, ok in enumerate(results) if not ok]
    if failed:
        print("\n❌ 以下文件转换失败：")
        for f in failed:
            print(" -", f)
