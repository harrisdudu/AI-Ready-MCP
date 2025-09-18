import logging
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError

#######################

#1) 安装依赖 pip install boto3 
#2) 配置成你的mino地址里corpus-outputs的子目录,也就是bid
#SOURCE_FOLDER = 'P2025021800001_C2024112699994'
#3) 配置成你脚本所运行的电脑上的用来存放下载文件的目录地址
#TARGET_FOLDER = r'D:\P2025021800001_C2024112699994'

#####################

# 文件服务器API 账号配置信息
endpoint_url = "http://172.20.90.11:8009"
aws_access_key_id = "123456"
aws_secret_access_key = "inspuR12345"

# 源数据所在桶 BUCKET_NAME
# 源数据所在桶下的子目录 SOURCE_FOLDER
BUCKET_NAME = 'corpus-outputs'
SOURCE_FOLDER = 'P2025071100002_C2025081800001/'
# 文件下载存放目录 TARGET_FOLDER
TARGET_FOLDER = r'F:/2-成品语料库/P2025071100002_C2025081800001/'

# 下载统计
downloaded_count = 0
skipped_count = 0
total_count = 0
lock = threading.Lock()

# 创建S3客户端配置
config = Config(max_pool_connections=1000)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Thread: %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


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


def get_all_files(s3_client):
    """获取指定路径下的所有文件列表"""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=SOURCE_FOLDER)

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('pic.zip'):  # 只取
                    logger.info(f"find {key}")
                    files.append(key)
    return files


def download_file(s3_client, s3_key, target_path, max_retries=10, delay=3):
    """下载单个文件（含重试机制）"""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3_client.download_file(BUCKET_NAME, s3_key, target_path)
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
                time.sleep(delay)
        except Exception as e:
            logger.error(f"下载失败 ({s3_key}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
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
    s3_client = get_client()

    # 获取所有文件列表
    all_files = get_all_files(s3_client)
    total_count = len(all_files)

    if total_count == 0:
        logger.warning("没有找到需要下载的文件")
        return

    logger.info(f"开始下载任务，共发现 {total_count} 个文件")

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = []
        for s3_key in all_files:
            # 生成本地路径（保持目录结构）
            relative_path = os.path.relpath(s3_key, SOURCE_FOLDER)
            local_path = os.path.join(TARGET_FOLDER, relative_path)

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