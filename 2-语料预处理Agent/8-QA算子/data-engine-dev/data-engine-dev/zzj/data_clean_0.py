# coding:utf-8
import json
import os
import sys
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import comtypes.client
from tqdm import tqdm

# ======== 全局变量：用于日志去重 ========
written_log_paths = set()  # 记录已写入的 input_path
log_lock = threading.Lock()  # 写文件锁
written_lock = threading.Lock()  # 控制 written_log_paths 的访问


def save_single_log(record: dict, log_path: str):
    """实时追加一条日志，确保 input_path 唯一（upsert）"""
    input_path = record["input_path"]
    with log_lock:
        with written_lock:
            if input_path in written_log_paths:
                return  # 已写入，跳过
            written_log_paths.add(input_path)

        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def load_log(log_path: str):
    """加载已有日志，填充 written_log_paths"""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    input_path = record["input_path"]
                    with written_lock:
                        written_log_paths.add(input_path)
                except Exception as e:
                    print(f"[警告] 跳过无效日志行: {e}")


def find_need_full_files(start_path, file_extensions):
    """
    查找指定目录及其子目录中符合扩展名的文件（返回全路径列表）
    """
    matched_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 过滤临时文件
            if '~$' in os.path.basename(file_path) or '.~' in os.path.basename(file_path):
                continue
            if any(file.lower().endswith(ext) for ext in file_extensions):
                matched_files.append(file_path)
    return matched_files


def get_output_path(input_path: str, source_root: str, output_root: str) -> str:
    """
    根据输入路径生成对应的输出 PDF 路径，保持从 source_root 父目录开始的目录结构。
    例如：
    source_root = '/path/to/input/消防各处室资料0801'
    input_path = '/path/to/input/消防各处室资料0801/子目录/文件.docx'
    output_root = '/path/output'
    则输出路径为 '/path/output/消防各处室资料0801/子目录/文件.pdf'
    """
    logical_root = os.path.dirname(source_root)
    # --- 计算相对于 logical_root 的路径，以保留 '消防各处室资料0801' 这一级 ---
    rel_path = os.path.relpath(input_path, logical_root)
    name, _ = os.path.splitext(rel_path)
    return os.path.join(output_root, name + ".pdf")


def convert_file(input_path: str, source_root: str, output_root: str, pages_to_extract: int):
    """
    转换单个文件的前 N 页为 PDF，返回结果字典。
    """
    import pythoncom  # 用于 COM 初始化
    pythoncom.CoInitialize()  # ✅ 必须：初始化当前线程的 COM 库

    start_time = datetime.now()
    result = {
        "input_path": input_path,
        "is_successed": False,
        "start_time": start_time.isoformat(),
        "pages_extracted": pages_to_extract
    }

    try:
        # 获取输出路径并创建目录
        output_pdf = get_output_path(input_path, source_root, output_root)
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        ext = os.path.splitext(input_path)[1].lower()

        if ext == '.pdf':
            print(f"[处理] 检测到 PDF 文件，正在复制: {os.path.basename(input_path)}")
            # 直接复制文件
            shutil.copy2(input_path, output_pdf)  # copy2 会尝试保留元数据
            duration = (datetime.now() - start_time).total_seconds()
            result.update({
                "is_successed": True,
                "output_path": output_pdf,
                "duration_sec": round(duration, 2),
                "end_time": datetime.now().isoformat(),
                "action": "copied"  # 可选：添加一个字段标识是复制操作
            })
            # 直接返回，不执行后续的 Office 转换逻辑
            return result

        print(f"[处理] 正在转换: {os.path.basename(input_path)}")

        if ext in ['.doc', '.docx']:
            # --- 调试建议：将 Visible=True 用于首次测试 ---
            word = comtypes.client.CreateObject('Word.Application')
            # word.Visible = False  # 可能导致后台问题
            word.Visible = True  # 强烈建议在首次运行时使用 True 来观察行为
            doc = word.Documents.Open(input_path)

            # --- 关键修改：提取前 N 页 ---
            # 获取文档总页数，并添加异常处理
            total_pages = 0
            try:
                # 3 = wdNumberOfPagesInDocument
                total_pages = doc.Content.Information(3)
                print(f"    [调试] 文档总页数: {total_pages}")
            except Exception as e:
                print(f"    [警告] 无法获取文档页数，错误: {e}")
                # 如果无法获取页数，保守起见，只导出第一页
                total_pages = 1

            pages_to_save = min(pages_to_extract, total_pages)
            print(f"    [调试] 将导出第 1 至 {pages_to_save} 页")  # 打印调试信息

            if pages_to_save <= 0:
                raise ValueError("计算出的页数 <= 0，无法保存。")

            doc.ExportAsFixedFormat(
                OutputFileName=output_pdf,
                ExportFormat=17,  # 17 = PDF
                OpenAfterExport=False,
                OptimizeFor=0,  # 0 = wdOptimizeForPrint
                From=1,
                To=pages_to_save,
                Item=0,  # 0 = wdExportDocumentContent
                IncludeDocProps=True,
                KeepIRM=True,
                CreateBookmarks=0,
                DocStructureTags=True,
                BitmapMissingFonts=True,
                UseISO19005_1=False
            )
            doc.Close()
            word.Quit()
            result["pages_actual"] = pages_to_save


        elif ext in ['.ppt', '.pptx']:
            powerpoint = None
            presentation = None
            try:
                powerpoint = comtypes.client.CreateObject('PowerPoint.Application')
                powerpoint.Visible = True  # 建议设置为 True 以避免 "Hiding..." 错误

                print(f"    [调试] 正在尝试打开演示文稿: {input_path}")
                presentation = powerpoint.Presentations.Open(input_path)
                # --- 增强健壮性：检查 presentation 对象 ---
                if presentation is None:
                    raise Exception("PowerPoint 未能成功打开演示文稿，返回对象为 None。")

                # 获取演示文稿幻灯片总数
                try:
                    total_slides = presentation.Slides.Count
                    print(f"    [调试] 演示文稿总幻灯片数: {total_slides}")
                    if total_slides <= 0:
                        raise ValueError("演示文稿幻灯片数为0。")
                except Exception as count_err:
                    # 如果无法获取幻灯片数量，也认为文件可能有问题
                    raise Exception(f"无法获取演示文稿幻灯片数，可能文件已损坏: {count_err}")

                slides_to_save = min(pages_to_extract, total_slides)
                print(f"    [调试] 将导出第 1 至 {slides_to_save} 张幻灯片")

                # --- 增强健壮性：添加短暂延迟 ---
                # import time
                # time.sleep(0.5) # 给 PowerPoint 一点时间稳定对象

                # --- 修复 ExportAsFixedFormat 调用 ---
                try:
                    # 1. 先清除可能存在的旧范围设置
                    presentation.PrintOptions.Ranges.ClearAll()
                    # 2. 添加新的打印范围
                    range_obj = presentation.PrintOptions.Ranges.Add(1, slides_to_save)
                    print(f"    [调试] 已设置打印范围: 1 - {slides_to_save}")

                    # 3. 调用导出，使用简化且更标准的参数
                    presentation.ExportAsFixedFormat(
                        Path=output_pdf,
                        FixedFormatType=32,  # ppFixedFormatTypePDF
                        Intent=1,  # ppIntentScreen
                        FrameSlides=False,
                        HandoutOrder=1,  # ppPrintHandoutVerticalFirst
                        OutputType=1,  # ppPrintOutputSlides
                        PrintHiddenSlides=False
                        # 移除 PrintRanges=... 参数
                    )
                    print(f"    [调试] 导出成功。")

                except Exception as export_err:
                    error_code = getattr(export_err, 'hresult', None)
                    if error_code == -2147188720:  # 0x80048010
                        print(f"    [警告] 导出时遇到 'Object does not exist' 错误，可能是对象状态问题。")
                    elif error_code == -2147023504:  # 0x80070520
                        print(f"    [警告] 导出时遇到 '文件或目录损坏' 错误。")
                    raise export_err  # 重新抛出，让外层捕获

            except Exception as open_or_process_err:
                error_code = getattr(open_or_process_err, 'hresult', None)
                if error_code == -2147023504:  # 0x80070520
                    print(f"    [错误] 文件 '{input_path}' 可能已损坏或无法读取。")
                    # 可以选择在这里直接设置 result["error"] 并 return，避免后续操作
                    # 但抛出异常让外层统一处理日志记录通常更好
                raise open_or_process_err  # 重新抛出，让外层的通用异常处理捕获并记录

            finally:
                # --- 确保资源清理 ---
                try:
                    if presentation:
                        presentation.Close()
                        print(f"    [调试] 演示文稿已关闭。")
                except Exception as close_err:
                    print(f"    [警告] 关闭演示文稿时出错: {close_err}")
                try:
                    if powerpoint:
                        powerpoint.Quit()
                        print(f"    [调试] PowerPoint 应用程序已退出。")
                except Exception as quit_err:
                    print(f"    [警告] 退出 PowerPoint 应用程序时出错: {quit_err}")
                # 清理 PrintOptions.Ranges
                # 注意：如果 presentation 已关闭，访问其 PrintOptions 可能会失败
                # 最好在 presentation 还有效时清理，如上面 ExportAsFixedFormat 成功后
                # ---
            # --- 结束增强健壮性修改 ---

            result["slides_actual"] = slides_to_save  # 确保 slides_to_save 在此之前已定义

        # 检查是否生成成功
        if os.path.exists(output_pdf):
            duration = (datetime.now() - start_time).total_seconds()
            result.update({
                "is_successed": True,
                "output_path": output_pdf,
                "duration_sec": round(duration, 2),
                "end_time": datetime.now().isoformat()
            })
        else:
            result["error"] = "未生成 PDF 文件"
    except Exception as e:
        result["error"] = f"异常: {str(e)}"
        result["end_time"] = datetime.now().isoformat()
    finally:
        pythoncom.CoUninitialize()

    return result


def main():
    # ======== 配置区 (全部集中在此) ========
    PAGES_TO_EXTRACT = 5  # 提取前5页，可修改此数字
    DEBUG = False  # True: 单线程调试；False: 多线程
    MAX_WORKERS = 6
    MAX_TIMEOUT = 300  # 单文件最大处理时间（秒）

    # SOURCE_FOLDER = r"D:\KPS\日常工作\消防知识库"  # 输入根目录
    # SOURCE_FOLDER = r"D:\KPS\日常工作\消防\消防各处室资料0801_补充"  # 输入根目录
    SOURCE_FOLDER = r"D:\KPS\日常工作\消防\消防各处室资料0801_rename\18.采购办公室"  # 输入根目录

    OUTPUT_ROOT = r"D:\KPS\job\data_clean_fire\rename_data_0801\output\消防各处室资料0801_rename"  # 输出根目录
    LOG_FILE = r"D:\KPS\job\data_clean_fire\data_0801\logs\pdf_conversion.jsonl"  # 日志路径

    FILE_EXTENSIONS = [".doc", ".docx", ".ppt", ".pptx", "pdf"]
    # FILE_EXTENSIONS = ["pdf"]
    # FILE_EXTENSIONS = [".doc", ".docx", "pdf"]

    SKIP_FILE_PATHS = []
    # ===============================================

    start_all = time.time()
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 初始化日志去重集合
    global written_log_paths
    written_log_paths = set()
    load_log(LOG_FILE)

    # 查找所有文件
    all_files = find_need_full_files(SOURCE_FOLDER, FILE_EXTENSIONS)
    skipped_files = [
        f for f in all_files
        if os.path.abspath(f) in [os.path.abspath(p) for p in SKIP_FILE_PATHS]
    ]
    files_to_process = [
        f for f in all_files
        if f not in written_log_paths
           and os.path.abspath(f) not in [os.path.abspath(p) for p in SKIP_FILE_PATHS]
    ]

    print(f"[文件] 共找到 {len(all_files)} 个文件")
    print(f"[处理] 待处理: {len(files_to_process)} 个")

    # ======== 处理逻辑 ========
    if DEBUG:
        print(f"[模式] 调试模式（单线程），提取前 {PAGES_TO_EXTRACT} 页")
        for file_path in tqdm(files_to_process, desc="Converting"):
            try:
                result = convert_file(file_path, SOURCE_FOLDER, OUTPUT_ROOT, PAGES_TO_EXTRACT)
            except Exception as e:
                now = datetime.now().isoformat()
                result = {
                    "input_path": file_path,
                    "is_successed": False,
                    "error": f"意外异常: {str(e)}",
                    "start_time": now,
                    "end_time": now
                }
            save_single_log(result, LOG_FILE)
    else:
        print(f"[模式] 多线程模式（{MAX_WORKERS} 线程），提取前 {PAGES_TO_EXTRACT} 页")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(convert_file, fp, SOURCE_FOLDER, OUTPUT_ROOT, PAGES_TO_EXTRACT): fp
                for fp in files_to_process
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="batch_convert"):
                file_path = futures[future]
                try:
                    result = future.result(timeout=MAX_TIMEOUT)
                except TimeoutError:
                    now = datetime.now().isoformat()
                    result = {
                        "input_path": file_path,
                        "is_successed": False,
                        "error": f"超时 > {MAX_TIMEOUT}s",
                        "start_time": now,
                        "end_time": now,
                        "duration_sec": MAX_TIMEOUT
                    }
                except Exception as e:
                    now = datetime.now().isoformat()
                    result = {
                        "input_path": file_path,
                        "is_successed": False,
                        "error": f"任务异常: {str(e)}",
                        "start_time": now,
                        "end_time": now
                    }
                save_single_log(result, LOG_FILE)

    # ======== 统计结果 ========
    end_all = time.time()
    total_time = round(end_all - start_all, 2)
    success = fail = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("is_successed"):
                        success += 1
                    else:
                        fail += 1
                except:
                    continue

    print(f"\n[完成] 转换完成：成功 {success}，失败 {fail}")
    print(f"[耗时] 总耗时: {total_time} 秒")
    print(f"[日志] 已保存: {LOG_FILE}")
    print(f"[输出] 输出目录: {OUTPUT_ROOT}")


if __name__ == '__main__':
    if hasattr(sys, "setdefaultencoding"):
        pass
    else:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
