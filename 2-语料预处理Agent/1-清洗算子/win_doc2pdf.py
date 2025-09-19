# 此导入问题通常是由于 pywin32 库未安装导致的。
# 请在终端中运行以下命令安装该库：
# pip install pywin32
import sys
import win32com.client
import os
import os.path as osp
import time
# 此导入问题通常是由于 pywin32 库未安装导致的。
# 该库已在前面尝试导入 win32com.client 时提示安装，pythoncom 是 pywin32 库的一部分，
# 安装 pywin32 后即可正常使用，无需额外导入，故移除该导入语句。

from tqdm import tqdm


def word_to_pdf(input_path, output_path=None, retry=3):
    # 路径防护：确保用绝对路径，且不含非法字符
    input_path = os.path.abspath(input_path)
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"
    else:
        output_path = os.path.abspath(output_path)

    for attempt in range(retry):
        word = None
        doc = None
        try:
            # 独立初始化线程 COM 环境（避免异常 COM 状态）
            pythoncom.CoInitialize()

            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0  # 禁止弹窗

            doc = word.Documents.Open(input_path, ReadOnly=1)
            doc.ExportAsFixedFormat(
                OutputFileName=output_path,
                ExportFormat=17,
                OpenAfterExport=False,
                OptimizeFor=0,
                CreateBookmarks=1
            )
            print(f"✅ 成功转换: {output_path}")
            return True
        except Exception as e:
            print(f"⚠️ 第 {attempt+1} 次尝试失败: {e}")
            time.sleep(1)  # 等待一秒后重试
        finally:
            try:
                if doc:
                    doc.Close(False)
                if word:
                    word.Quit()
                pythoncom.CoUninitialize()
            except Exception as e:
                print(f"⚠️ 关闭 Word 失败（已断连）: {e}")
    print(f"❌ 转换失败: {input_path}")
    return False

# 示例调用
if __name__ == "__main__":
    input_dir = r"C:\Users\Admin\Desktop\excel_input"
    output_dir = r"E:\补充转化文件"
    os.makedirs(output_dir, exist_ok=True)
    for doc in tqdm(os.listdir(input_dir)):
        output_path = osp.join(output_dir, osp.splitext(doc)[0]+"_kps_converted.pdf")
        if osp.exists(output_path): continue
        if doc.lower().endswith((".doc", ".docx")):
            # print(osp.join(input_dir, doc))
            word_to_pdf(osp.join(input_dir, doc), output_path)