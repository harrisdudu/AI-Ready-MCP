import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import tqdm
import pandas as pd
import logging
import file_utils

# 配置日志，用于记录失败的文件
logging.basicConfig(filename='download_failures.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = ["pdf", "epub", "azw","azw3","mobi"]

class S3Util:
    def __init__(self, endpoint, ak, sk):
        self.endpoint = endpoint
        self.ak = ak
        self.sk = sk
        self.s3_client = self._create_s3_client()

    def _create_s3_client(self):
        """Create an S3 client."""
        return boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.ak,
            aws_secret_access_key=self.sk
        )

    def download_file(self, bucket_name, object_key, download_dir):
        """Download a single file from S3 directly into the specified download directory."""
        filename = os.path.basename(object_key)
        download_path = os.path.join(download_dir, filename)

        # Create download directory if it does not exist
        os.makedirs(download_dir, exist_ok=True)

        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
            file_size = response['ContentLength']

            with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {filename}", leave=False) as pbar:
                with open(download_path, 'wb') as file:
                    def progress_callback(bytes_downloaded):
                        pbar.update(bytes_downloaded)

                    self.s3_client.download_fileobj(bucket_name, object_key, file, Callback=progress_callback)
            return True, object_key, None  # 成功
        except Exception as e:
            error_msg = f"Failed to download {object_key}: {e}"
            print(error_msg)
            logging.error(error_msg)
            return False, object_key, str(e)  # 失败

def worker(args):
    bucket_name, s3_path, download_dir, s3_util_params = args
    # s3_path = "P2025062300001_zlib_下载数据集/pilimi-zlib-11000000-11039999/11013680.mobi"
    endpoint, ak, sk = s3_util_params
    s3_util = S3Util(endpoint, ak, sk)
    results = []
    if s3_path.split(".")[-1] in SUPPORTED_EXTENSIONS:
        object_key = s3_path
        success, key, error = s3_util.download_file(bucket_name, object_key, download_dir)
        if success:
            return success, key, None  # 成功一个就返回
        else:
            results.append((key, error))
    
    # 如果所有格式都失败了，返回最后一个错误
    return False, s3_path, results

if __name__ == "__main__":
    debug=False
    # S3 配置信息
    ENDPOINT = "http://172.20.90.11:8009"
    AK = "123456"
    SK = "inspuR12345"
    BUCKET_NAME = "corpus-origin"
    DOWNLOAD_DIR = "/mnt/data/zzj/fire_s3/data/500w书单提取/download"
    # DOWNLOAD_DIR = "download"
    origin_failed_path = "/mnt/data/zzj/fire_s3/data/origin_failed.jsonl"
    origin_successed_path = "/mnt/data/zzj/fire_s3/data/origin_successed.jsonl"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    file_path = "/mnt/data/zzj/fire_s3/data/500w书单提取.xlsx"
    df = pd.read_excel(file_path)
    arry_s3_path = df["s3_path"].tolist()
    arry_s3_path.sort()
    # arry_s3_path = df["s3_path"].dropna().astype(str)
    # filtered_s3_paths = arry_s3_path[arry_s3_path.str.endswith(tuple(SUPPORTED_EXTENSIONS))].tolist()

    s3_util_params = (ENDPOINT, AK, SK)

    tasks = [(BUCKET_NAME, s3_path.strip(), DOWNLOAD_DIR, s3_util_params) for s3_path in arry_s3_path]

    print(f"开始下载 {len(arry_s3_path)} 个 s3_path 对应的文件（多进程）...")

    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     results = list(tqdm.tqdm(executor.map(worker, tasks), total=len(tasks), desc="Overall Progress"))
    if debug:
        # 单次调用调试模式 - 逐个执行所有任务
        print("🔧 调试模式：逐个执行所有任务（单线程）")
        results = []
        for i, task in enumerate(tqdm.tqdm(tasks, desc="Debug Processing")):
            try:
                print(f"\n🔍 处理任务 {i+1}/{len(tasks)}: {task[1]}")
                result = worker(task)
                results.append(result)
                if result[0]:  # 成功
                    print(f"   ✅ 成功下载: {result[1]}")
                else:  # 失败
                    print(f"   ❌ 下载失败: {result[1]}, 错误: {result[2]}")
            except Exception as e:
                print(f"   ⚠️  任务异常: {e}")
                results.append((False, task[1], str(e)))
        
        # 统计调试结果
        failed_downloads = [res for res in results if not res[0]]
        print(f"\n🔧 调试完成。总任务数: {len(tasks)}, 成功: {len(results)-len(failed_downloads)}, 失败: {len(failed_downloads)}")
        
        # 保存失败结果
        if failed_downloads:
            print("\n❌ 调试模式下的失败文件：")
            for _, s3_path, errors in failed_downloads:
                print(f"  - s3_path: {s3_path}, Errors: {errors}")
            file_utils.save_to_jsonl(failed_downloads, origin_failed_path)
    else:
        # 创建总体进度条
        with tqdm.tqdm(total=len(tasks), desc="Total Files Progress", unit="files") as total_pbar:
            def update_progress(future):
                """更新总体进度条的回调函数"""
                total_pbar.update(1)

            with ProcessPoolExecutor(max_workers=128) as executor:
                # 提交所有任务
                futures = [executor.submit(worker, task) for task in tasks]
                
                # 为每个future添加完成回调
                for future in futures:
                    future.add_done_callback(lambda f: total_pbar.update(1))
                
                # 等待所有任务完成
                results = []
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), 
                                    total=len(futures), 
                                    desc="Processing Files", 
                                    leave=False):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Task failed with exception: {e}")
                        results.append((False, "unknown", str(e)))

        failed_downloads = [res for res in results if not res[0]]
        successed_downloads = [res for res in results if res[0]]

        print(f"\n✅ 下载完成。总共失败文件数: {len(failed_downloads)}")
        if failed_downloads:
            print("\n❌ 以下文件下载失败：")
            for _, s3_path, errors in failed_downloads:
                print(f"  - s3_path: {s3_path}, Errors: {errors}")
            file_utils.save_to_jsonl(failed_downloads,origin_failed_path)
        file_utils.save_to_jsonl(successed_downloads,origin_successed_path)