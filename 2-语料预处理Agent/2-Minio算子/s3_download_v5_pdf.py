import logging
import threading
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Thread: %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

#######################
# 配置信息
#######################

#文件服务器API 账号配置信息
endpoint_url = "http://172.20.90.11:8009"
aws_access_key_id = "123456"
aws_secret_access_key = "inspuR12345"

# endpoint_url = "http://172.20.90.120:5080"
# aws_access_key_id = "BE160C000D4B82DE1826"
# aws_secret_access_key = "DvecIfmA4OkQOUbJwtQrZrQKBGwAAAGYDUuC3+j9"

# 源数据所在桶
# BUCKET_NAME = 'corpus-origin-hot'
BUCKET_NAME = 'corpus-origin'
# 默认入库编号（作为key，实际使用时应根据需要修改或通过命令行参数指定）
DEFAULT_RECEIPT_NO = 'P2025070700001'

# 默认目标文件夹
DEFAULT_TARGET_FOLDER = r'F:\1-原料仓库\第13批原料-P2025070700001'

# 下载统计全局变量
downloaded_count = 0
skipped_count = 0
total_count = 0
lock = threading.Lock()

# 创建S3客户端配置
config = Config(max_pool_connections=1000)

#######################
# 工具函数
#######################

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='S3文件下载工具')
    parser.add_argument('--receipt-no', type=str, default=DEFAULT_RECEIPT_NO, help='入库编号（作为key用于匹配文件夹）')
    parser.add_argument('--target-folder', type=str, default=DEFAULT_TARGET_FOLDER, help='下载目标文件夹')
    return parser.parse_args()

# 查找匹配入库编号的文件夹
def find_matching_folder(s3_client, receipt_no):
    """使用入库编号作为key查找匹配的S3文件夹（入库编号+中文说明格式）"""
    logger.info(f"使用入库编号 '{receipt_no}' 作为key查找匹配的文件夹")
    
    # 列出根目录下的所有对象
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Delimiter='/')
    
    matching_folders = []
    
    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for prefix_info in page['CommonPrefixes']:
                folder_name = prefix_info['Prefix']
                # 检查文件夹名是否以入库编号开头，且后面要么是结束（纯入库编号文件夹）要么是下划线等分隔符
                # 这样可以避免部分匹配的情况
                if folder_name == receipt_no + '/' or folder_name.startswith(f"{receipt_no}_"):
                    matching_folders.append(folder_name)
                    logger.info(f"找到匹配的文件夹: {folder_name}")
    
    if not matching_folders:
        logger.warning(f"没有找到以入库编号 '{receipt_no}' 开头的文件夹")
        return None
    
    # 如果找到多个匹配的文件夹，选择第一个
    selected_folder = matching_folders[0]
    logger.info(f"选择文件夹: {selected_folder}")
    
    return selected_folder


def get_client():
    """创建并返回S3客户端"""
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=config,
        verify=False
    )


def get_all_files(s3_client, source_folder):
    """获取指定S3路径下的所有PDF文件列表"""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=source_folder)

    logger.info(f"开始扫描S3路径：{source_folder} 中的PDF文件")
    
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # 只获取PDF文件
                if key.endswith('.pdf'):
                    logger.debug(f"发现PDF文件：{key}")
                    files.append(key)
    
    logger.info(f"扫描完成，共发现 {len(files)} 个PDF文件")
    return files


def download_file(s3_client, s3_key, target_path, max_retries=10, delay=3):
    """下载单个文件（含重试机制）"""
    for attempt in range(max_retries):
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # 执行下载操作
            s3_client.download_file(BUCKET_NAME, s3_key, target_path)
            logger.debug(f"成功下载文件: {s3_key} -> {target_path}")
            return True
        except NoCredentialsError:
            logger.error("凭证错误，请检查访问密钥")
            break
        except ClientError as e:
            error_code = e.response['Error'].get('Code', 'Unknown')
            if error_code == '404':
                logger.error(f"文件不存在: {s3_key}")
                break
            logger.error(f"客户端错误 ({error_code}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"{attempt+1}/{max_retries} 次下载失败，{delay}秒后重试...")
                time.sleep(delay)
        except Exception as e:
            logger.error(f"下载失败 ({s3_key}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"{attempt+1}/{max_retries} 次下载失败，{delay}秒后重试...")
                time.sleep(delay)
    logger.warning(f"文件下载失败（已重试 {max_retries} 次）: {s3_key}")
    return False


def process_file(s3_client, s3_key, local_path):
    """处理单个文件下载任务"""
    global downloaded_count, skipped_count

    if os.path.exists(local_path):
        with lock:
            skipped_count += 1
            logger.warning(f"文件已存在，跳过: {local_path}")
        return

    success = download_file(s3_client, s3_key, local_path)
    if success:
        with lock:
            downloaded_count += 1
            progress = downloaded_count / total_count * 100
            logger.info(
                f"下载进度: {progress:.2f}% | "
                f"成功: {downloaded_count}/{total_count} | "
                f"最新文件: {os.path.basename(local_path)}"
            )


def main():
    global total_count
    
    # 在main函数内部解析命令行参数
    args = parse_args()
    
    # 获取入库编号和目标文件夹
    receipt_no = args.receipt_no
    target_folder = args.target_folder
    
    s3_client = get_client()
    
    # 使用入库编号作为key查找匹配的文件夹
    source_folder = find_matching_folder(s3_client, receipt_no)
    
    if not source_folder:
        # 如果没有找到匹配的文件夹，可以选择使用入库编号作为默认文件夹
        logger.info(f"使用入库编号 '{receipt_no}' 作为默认文件夹")
        source_folder = f'{receipt_no}/'
    
    logger.info(f"开始下载任务")
    logger.info(f"入库编号(key): {receipt_no}")
    logger.info(f"源文件夹: {source_folder}")
    logger.info(f"目标文件夹: {target_folder}")

    # 获取所有文件列表
    all_files = get_all_files(s3_client, source_folder)
    total_count = len(all_files)

    if total_count == 0:
        logger.warning("没有找到需要下载的文件")
        return

    logger.info(f"开始下载任务，共发现 {total_count} 个文件")

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = []
        for s3_key in all_files:
            # 生成本地路径（保持目录结构）
            relative_path = os.path.relpath(s3_key, source_folder)
            local_path = os.path.join(target_folder, relative_path)

            futures.append(
                executor.submit(process_file, s3_client, s3_key, local_path)
            )

        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()

    # 输出最终统计结果
    logger.info("\n下载任务完成！")
    logger.info(f"总文件数: {total_count}")
    logger.info(f"成功下载: {downloaded_count}")
    logger.info(f"跳过已存在: {skipped_count}")
    if downloaded_count + skipped_count < total_count:
        logger.warning(f"可能有 {total_count - downloaded_count - skipped_count} 个文件下载失败")


if __name__ == '__main__':
    main()