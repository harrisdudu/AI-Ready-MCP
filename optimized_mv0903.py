import pandas as pd
import os 
import os.path as osp
import shutil
import re
import logging
import time
import argparse
from datetime import datetime
import threading
import traceback
import sys

# 设置日志 - 同时输出到控制台和文件（使用绝对路径）
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'file_migration.log')

# 确保日志目录存在
try:
    os.makedirs(script_dir, exist_ok=True)
    
    # 创建日志处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - Thread: %(threadName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新的处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = root_logger
    print(f"日志已配置完成，同时输出到控制台和文件: {log_file_path}")
except Exception as e:
    print(f"配置日志时出错: {str(e)}")
    # 回退到基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Thread: %(threadName)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

# 性能监控类
class PerformanceMonitor:
    def __init__(self):
        # 使用threading.Lock实现线程安全
        self.lock = threading.Lock()
        
        # 基本时间记录
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # 计数器 - 使用普通整数实现，通过锁保证线程安全
        self.processed_since_last_log = 0
        self.total_processed = 0
        self.total_success = 0
        self.total_fail = 0
        self.total_skip = 0
        
        # 配置参数
        self.log_interval = 5  # 日志记录间隔（秒）
        self.skip_log_interval = 500  # 跳过文件的日志记录间隔
    
    def increment_processed(self):
        """增加已处理文件计数并检查是否需要记录进度"""
        with self.lock:
            self.total_processed += 1
            self.processed_since_last_log += 1
            self._check_and_log_progress()
    
    def increment_success(self):
        """增加成功处理的文件计数"""
        with self.lock:
            self.total_success += 1
    
    def increment_fail(self):
        """增加处理失败的文件计数"""
        with self.lock:
            self.total_fail += 1
    
    def increment_skip(self):
        """增加跳过的文件计数"""
        with self.lock:
            self.total_skip += 1
            # 减少跳过文件的日志频率
            if self.total_skip % self.skip_log_interval == 0:
                logger.info(f"已跳过 {self.total_skip} 个已存在的文件")
    
    def _check_and_log_progress(self):
        """内部方法，检查并记录处理进度"""
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval and self.processed_since_last_log > 0:
            with self.lock:
                # 再次检查条件，避免竞态条件
                if current_time - self.last_log_time >= self.log_interval and self.processed_since_last_log > 0:
                    elapsed = current_time - self.start_time
                    files_per_second = self.total_processed / elapsed if elapsed > 0 else 0
                    logger.info(f"进度: 已处理 {self.total_processed} 个文件, 速度: {files_per_second:.2f} 文件/秒")
                    self.last_log_time = current_time
                    self.processed_since_last_log = 0
    
    def get_summary(self):
        """获取完整的性能统计摘要"""
        with self.lock:
            elapsed = time.time() - self.start_time
            files_per_second = self.total_processed / elapsed if elapsed > 0 else 0
            return {
                'total_processed': self.total_processed,
                'total_success': self.total_success,
                'total_fail': self.total_fail,
                'total_skip': self.total_skip,
                'elapsed_time': elapsed,
                'files_per_second': files_per_second
            }
    
    # 新增：批量更新方法，减少锁竞争
    def batch_update(self, processed=0, success=0, fail=0, skip=0):
        """批量更新计数器，减少锁的获取次数"""
        with self.lock:
            if processed > 0:
                self.total_processed += processed
                self.processed_since_last_log += processed
                self._check_and_log_progress()
            if success > 0:
                self.total_success += success
            if fail > 0:
                self.total_fail += fail
            if skip > 0:
                self.total_skip += skip
                if self.total_skip % self.skip_log_interval == 0:
                    logger.info(f"已跳过 {self.total_skip} 个已存在的文件")
    
    # 新增：无锁版本的方法，用于外部已加锁的场景
    def _increment_processed_no_lock(self):
        """无锁版本的increment_processed，用于已获取锁的场景"""
        self.total_processed += 1
        self.processed_since_last_log += 1
        self._check_and_log_progress()
    
    def _increment_success_no_lock(self):
        """无锁版本的increment_success，用于已获取锁的场景"""
        self.total_success += 1
    
    def _increment_fail_no_lock(self):
        """无锁版本的increment_fail，用于已获取锁的场景"""
        self.total_fail += 1
    
    def _increment_skip_no_lock(self):
        """无锁版本的increment_skip，用于已获取锁的场景"""
        self.total_skip += 1
        # 减少跳过文件的日志频率
        if self.total_skip % self.skip_log_interval == 0:
            logger.info(f"已跳过 {self.total_skip} 个已存在的文件")

# 用于文件复制的函数
def copy_file(source_path, dest_path, institution, monitor_obj, buffer_size=16*1024*1024):
    """复制单个文件并更新监控器状态"""
    src_file = None
    dst_file = None
    try:
        # 1. 预检查：确保源文件存在且可读
        if not os.path.exists(source_path):
            return False, source_path, "源文件不存在"
        
        if not os.access(source_path, os.R_OK):
            return False, source_path, "无权限读取源文件"
        
        # 2. 检查目标文件是否已存在且完整
        if os.path.exists(dest_path):
            if os.path.getsize(source_path) == os.path.getsize(dest_path):
                with monitor_obj.lock:
                    # 使用无锁版本的方法，避免锁的嵌套
        #  monitor_obj._increment_processed_no_lock()
                    monitor_obj._increment_skip_no_lock()
                return True, source_path, "跳过 - 文件已存在且大小相同"
        
        # 3. 确保目标目录存在
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except Exception as mkdir_e:
                return False, source_path, f"创建目标目录失败: {str(mkdir_e)}"
        
        # 4. 使用缓冲区优化大文件复制
        src_filename = os.path.basename(source_path)
        file_size = os.path.getsize(source_path)
        
        # 只对大文件进行debug日志记录
        if file_size > 100 * 1024 * 1024:
            logger.debug(f"开始复制大文件: {src_filename}, 大小: {file_size/1024/1024:.2f}MB")
        
        # 使用with语句自动管理文件句柄
        start_time = time.time()
        copied_bytes = 0
        
        # 使用上下文管理器确保文件句柄正确关闭
        with open(source_path, 'rb') as src_file, open(dest_path, 'wb') as dst_file:
            # 分块读取和写入
            while True:
                chunk = src_file.read(buffer_size)
                if not chunk:
                    break  # 文件读取完毕
                
                dst_file.write(chunk)
                copied_bytes += len(chunk)
                
                # 仅对大文件进行超时检查
                if file_size > 100 * 1024 * 1024:
                    # 每10MB检查一次超时
                    if copied_bytes % (10 * 1024 * 1024) < len(chunk):
                        elapsed = time.time() - start_time
                        if elapsed > 300:  # 增加超时时间到5分钟
                            # 清理部分复制的文件
                            if os.path.exists(dest_path):
                                try:
                                    os.remove(dest_path)
                                except:
                                    pass
                                with monitor_obj.lock:
                                    # 使用无锁版本的方法，避免锁的嵌套
           #   monitor_obj._increment_processed_no_lock()
                                    monitor_obj._increment_fail_no_lock()
                                return False, source_path, "文件复制超时（300秒）"
        
        # 5. 验证复制结果
        if os.path.exists(dest_path) and os.path.getsize(source_path) == os.path.getsize(dest_path):
            with monitor_obj.lock:
                # 使用无锁版本的方法，避免锁的嵌套
         #  monitor_obj._increment_processed_no_lock()
                monitor_obj._increment_success_no_lock()
            
            dst_filename = os.path.basename(dest_path)
            return True, source_path, f"成功复制: {src_filename} -> {institution}/{dst_filename}"
        else:
            # 复制不完整，删除目标文件
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except:
                pass
            with monitor_obj.lock:
                # 使用无锁版本的方法，避免锁的嵌套
                monitor_obj._increment_processed_no_lock()
                monitor_obj._increment_fail_no_lock()
            return False, source_path, "文件复制不完整或验证失败"
    except Exception as e:
        # 确保即使发生异常，资源也能被释放
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except:
            pass
        
        with monitor_obj.lock:
            # 使用无锁版本的方法，避免锁的嵌套
            monitor_obj._increment_processed_no_lock()
            monitor_obj._increment_fail_no_lock()
        
        src_filename = os.path.basename(source_path)
        logger.error(f"复制文件失败 {src_filename}: {str(e)}")
        return False, source_path, f"复制失败: {str(e)}"

# 优化的文件匹配函数
def extract_file_number(fname):
    """从文件名中提取开头的数字序号，后面带上-号"""
    match = re.match(r'^(\d+)', fname)
    if match:
        return match.group(1) + '-'
    return None

# 迁移文件的主函数
def migrate_files(source_dir, result_dict, output_dir=None, buffer_size=8*1024*1024):
    """迁移文件到指定目录（优化版本）"""
    if output_dir is None:
        output_dir = osp.dirname(osp.abspath(__file__))
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"开始迁移文件，源目录: {source_dir}, 输出目录: {output_dir}")
    
    # 创建性能监控器
    monitor_obj = PerformanceMonitor()
    
    # 预过滤有效序号
    valid_numeric_keys = {k for k in result_dict.keys() if k.isdigit()}
    
    # 收集需要复制的文件任务
    file_tasks = []
    
    # 遍历源目录查找匹配的文件
    start_scan_time = time.time()
    
    # 优化：提前过滤PDF文件
    pdf_files = []
    try:
        logger.info(f"开始扫描源目录: {source_dir}")
        # 添加进度报告
        processed_dirs = 0
        last_log_dirs = 0
        
        for root, _, fnames in os.walk(source_dir):
            # 每扫描500个目录报告一次进度
            processed_dirs += 1
            if processed_dirs - last_log_dirs >= 500:
                logger.info(f"已扫描 {processed_dirs} 个目录，已收集 {len(pdf_files)} 个PDF文件")
                last_log_dirs = processed_dirs
            
            pdf_files.extend(
                (root, fname) for fname in fnames if fname.lower().endswith('.pdf')
            )
        
        logger.info(f"目录扫描完成，共扫描 {processed_dirs} 个目录")
    except Exception as e:
        logger.error(f"扫描源目录时出错: {str(e)}")
    
    # 优化：提取文件名中的数字序号并匹配
    start_match_time = time.time()
    for root, fname in pdf_files:
        file_number = extract_file_number(fname)
        if file_number and file_number in valid_numeric_keys:
            source_path = osp.join(root, fname)
            institution = result_dict[file_number]
            dest_path = osp.join(output_dir, institution, fname)
            file_tasks.append((source_path, dest_path, institution))
    
    match_time = time.time() - start_match_time
    scan_time = time.time() - start_scan_time
    logger.info(f"匹配完成，共找到 {len(file_tasks)} 个需要复制的文件，扫描耗时 {scan_time:.2f} 秒，匹配耗时 {match_time:.2f} 秒")
    
    # 如果没有文件需要复制，直接返回
    if not file_tasks:
        logger.info("没有文件需要复制，任务完成")
        return 0, 0, 0
    
    # 复制文件（优化版）
    start_copy_time = time.time()
    total_files = len(file_tasks)
    
    logger.info(f"开始复制文件，共 {total_files} 个文件，使用单线程处理")
    
    # 添加进度跟踪变量
    completed_files = 0
    last_progress_log = 0
    progress_interval = max(1, total_files // 50)  # 减少日志频率，从20改为50
    
    # 记录失败的任务
    failed_tasks = []
    
    # 批量处理文件
    batch_size = 100  # 每处理100个文件后清理内存
    current_batch = 0
    
    # 单线程处理每个文件
    for task_index, (src, dst, inst) in enumerate(file_tasks, 1):
        try:
            # 直接调用copy_file函数处理每个文件
            success, path, info = copy_file(src, dst, inst, monitor_obj, buffer_size)
            
            # 仅对大批量操作或特殊情况记录详细日志
            if success and task_index % 200 == 0:
                logger.info(f"已复制文件: {os.path.basename(src)} -> {os.path.basename(dst)}")
            elif not success:
                # 处理失败任务
                logger.warning(f"文件复制未成功: {os.path.basename(src)}，原因: {info}")
                failed_tasks.append(f"{src} -> {dst}: {info}")
            
            # 定期记录进度
            if (task_index - last_progress_log >= progress_interval or 
                task_index == total_files):
                # 计算已完成百分比和当前速度
                elapsed = time.time() - start_copy_time
                progress_percent = (task_index / total_files) * 100
                current_speed = task_index / elapsed if elapsed > 0 else 0
                
                # 计算预计剩余时间
                if current_speed > 0:
                    remaining_files = total_files - task_index
                    estimated_remaining = remaining_files / current_speed
                    # 格式化剩余时间
                    hours, remainder = divmod(estimated_remaining, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                else:
                    time_str = "未知"
                
                logger.info(f"复制进度: {task_index}/{total_files} ({progress_percent:.1f}%) - 速度: {current_speed:.2f} 文件/秒 - 预计剩余: {time_str}")
                last_progress_log = task_index
                
        except Exception as e:
            # 直接处理异常，确保程序不会中断
            src_filename = os.path.basename(src)
            error_msg = f"处理文件 {src_filename} 时出错: {str(e)}"
            logger.error(error_msg)
            failed_tasks.append(f"{src} -> {dst}: 处理异常 - {str(e)}")
            try:
                with monitor_obj.lock:
                    # 使用无锁版本的方法，避免锁的嵌套
                    monitor_obj._increment_processed_no_lock()
                    monitor_obj._increment_fail_no_lock()
            except Exception as lock_e:
                logger.error(f"更新失败计数时出错: {str(lock_e)}")
        
        # 批量处理：每处理100个文件后清理内存
        if task_index % batch_size == 0:
            current_batch += 1
            # 显式调用垃圾回收
            import gc
            gc.collect()
            logger.debug(f"已完成第 {current_batch} 批文件处理，已释放内存")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 记录失败的任务到文件
    if failed_tasks:
        failed_log_path = os.path.join(script_dir, 'failed_migrations.log')
        try:
            with open(failed_log_path, 'w', encoding='utf-8') as f:
                for task in failed_tasks:
                    f.write(f"{task}\n")
            logger.warning(f"记录了 {len(failed_tasks)} 个失败的文件复制任务")
        except Exception as e:
            logger.error(f"写入失败任务日志时出错: {str(e)}")
    
    copy_time = time.time() - start_copy_time
    
    # 获取性能监控摘要
    summary = monitor_obj.get_summary()
    
    logger.info(f"文件迁移完成，共处理: {summary['total_processed']}, 成功: {summary['total_success']}, 失败: {summary['total_fail']}, 跳过: {summary['total_skip']}")
    logger.info(f"复制耗时: {copy_time:.2f} 秒, 平均速度: {summary['files_per_second']:.2f} 文件/秒")
    
    return summary['total_success'], summary['total_fail'], summary['total_skip']

# 主函数
if __name__ == "__main__":
    # 添加最早期的日志输出，验证程序是否启动
    print("程序启动中...")
    
    print("日志已配置完成，同时输出到控制台和file_migration.log文件")
    logger.critical("这是一条CRITICAL日志 - 应该总是显示")
    logger.error("这是一条ERROR日志")
    logger.warning("这是一条WARNING日志")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='文件迁移工具')
    parser.add_argument('--output-dir', type=str, default=r"F:\2-成品语料库\可共享第一批-原料", help='输出文件夹路径')
    parser.add_argument('--source-dir', type=str, default=r"F:\1-原料仓库", help='源文件目录路径')
    parser.add_argument('--excel', type=str, default=r"D:\2 技术资料知识库\Code Rep\特殊处理\序号-书名-采集单位.xlsx", help='Excel文件路径')
    parser.add_argument('--sheet', type=str, default="sheet1", help='Excel文件的sheet名称')
    parser.add_argument('--buffer-size', type=int, default=8, help='文件复制缓冲区大小（MB）')
    
    args = parser.parse_args()
    
    print(f"使用参数: 源目录={args.source_dir}, 输出目录={args.output_dir}, Excel文件={args.excel}")
    
    # 设置默认值用于测试
    source_dir = args.source_dir or os.path.join(script_dir, 'test_source')
    output_dir = args.output_dir or os.path.join(script_dir, 'test_output')
    excel_path = args.excel or os.path.join(script_dir, '序号-书名-采集单位.xlsx')
    buffer_size = args.buffer_size * 1024 * 1024  # 转换为字节
    
    try:
        # 读取Excel文件
        logger.info(f"正在读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name=args.sheet)
        
        # 创建结果字典
        result_dict = {}
        # 根据Excel实际列名调整这里的键
        if '序号' in df.columns and '采集单位' in df.columns:
            for _, row in df.iterrows():
                # 确保序号是字符串格式
                seq_id = str(row['序号']).strip()
                institution = str(row['采集单位']).strip()
                result_dict[seq_id] = institution
        elif '序号' in df.columns and 'institution' in df.columns:
            for _, row in df.iterrows():
                seq_id = str(row['序号']).strip()
                institution = str(row['institution']).strip()
                result_dict[seq_id] = institution
        else:
            # 尝试其他可能的列名组合
            logger.warning("未找到标准列名，尝试使用前两列")
            first_col = df.columns[0]
            second_col = df.columns[1]
            for _, row in df.iterrows():
                seq_id = str(row[first_col]).strip()
                institution = str(row[second_col]).strip()
                result_dict[seq_id] = institution
        
        logger.info(f"成功读取 {len(result_dict)} 条映射记录")
        
        # 分析序号位数分布
        seq_lengths = {}
        for seq in result_dict.keys():
            if seq.isdigit():
                length = len(seq)
                seq_lengths[length] = seq_lengths.get(length, 0) + 1
        
        if seq_lengths:
            logger.info(f"序号位数分布: {seq_lengths}")
        
        # 开始迁移文件
        total_start_time = time.time()
        success_count, fail_count, skip_count = migrate_files(
            source_dir, result_dict, output_dir, buffer_size
        )
        total_time = time.time() - total_start_time
        
        logger.info(f"\n===== 任务完成摘要 =====")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功文件数: {success_count}")
        logger.info(f"失败文件数: {fail_count}")
        logger.info(f"跳过文件数: {skip_count}")
        logger.info(f"====================")
        
        print(f"任务完成！成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
        print(f"详细日志请查看: {log_file_path}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)