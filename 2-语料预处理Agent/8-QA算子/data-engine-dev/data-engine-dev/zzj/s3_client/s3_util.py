import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
import concurrent.futures
import tqdm  # 用于显示进度条


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

    def download_folder(self, bucket_name, prefix, download_dir, max_workers=5):
        """Download all files in the specified S3 folder, preserving directory structure."""
        print(f"Starting to download folder: {prefix}")

        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

            # 过滤出实际文件，并排除文件夹
            files = [obj['Key'] for obj in response.get('Contents', []) if not obj['Key'].endswith('/')]

            # 计算总大小以显示进度
            total_size = sum(
                self.s3_client.head_object(Bucket=bucket_name, Key=file_key)['ContentLength'] for file_key in files)

            # 使用线程池下载文件并在下载时间显示进度条
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Folder") as pbar:
                def download_and_update(file_key):
                    """下载文件并更新进度条"""
                    file_size = self._download_file(bucket_name, file_key, download_dir, pbar)
                    # 更新进度条
                    pbar.update(file_size)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(download_and_update, file_key): file_key for file_key in files}
                    # 等待所有 Futures 完成
                    concurrent.futures.wait(futures)

            print(f"Downloaded successfully: {prefix} to {download_dir}")

        except ClientError as e:
            print(f"Error occurred: {e}")

    def download_file(self, bucket_name, object_key, download_dir):
        """Download a single file from S3 directly into the specified download directory without preserving structure."""
        filename = os.path.basename(object_key)
        download_path = os.path.join(download_dir, filename)

        # Create download directory if it does not exist
        os.makedirs(download_dir, exist_ok=True)

        # 使用 tqdm 创建进度条
        response = self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']

        print(f"Starting to download single file: {object_key}")  # 打印下载开始的信息
        with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {filename}") as pbar:
            try:
                with open(download_path, 'wb') as file:
                    def progress_callback(bytes_downloaded):
                        pbar.update(bytes_downloaded)

                    # Download the file with progress using download_fileobj
                    self.s3_client.download_fileobj(bucket_name, object_key, file, Callback=progress_callback)

                # 成功下载日志
                print(f"Downloaded successfully: {object_key} to {download_path}")

            except FileNotFoundError:
                print(f"The file was not found: {download_path}")
            except NoCredentialsError:
                print("Credentials not available")
            except ClientError as e:
                print(f"Error occurred: {e}")

    def _download_file(self, bucket_name, object_key, download_dir, pbar):
        """Download a single file from S3 and update the overall progress bar."""
        download_path = os.path.join(download_dir, object_key)

        # Create download directory if it does not exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        # Use tqdm to create a progress bar
        response = self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']

        with open(download_path, 'wb') as file:
            def progress_callback(bytes_downloaded):
                pbar.update(bytes_downloaded)

            # Download the file with progress using download_fileobj
            self.s3_client.download_fileobj(bucket_name, object_key, file, Callback=progress_callback)

        return file_size  # 返回已下载的文件大小以更新总进度条


# Test example
if __name__ == "__main__":
    # Please set S3 configuration information accordingly
    ENDPOINT = "http://172.20.90.11:8009"
    AK = "123456"
    SK = "inspuR12345"
    BUCKET_NAME = "corpus-origin"
    FOLDER_PREFIX = "T2025011300001_测试/"  # Folder prefix to download
    SINGLE_FILE_KEY = "T2024120200001_测试/测试数据_9787512111226.pdf"  # Single file key to download
    DOWNLOAD_DIR = "download"  # Directory to download files to

    s3_util = S3Util(ENDPOINT, AK, SK)

    # Download entire folder with progress and configurable thread count
    s3_util.download_folder(BUCKET_NAME, FOLDER_PREFIX, DOWNLOAD_DIR, max_workers=5)

    # Download a single file with progress
    s3_util.download_file(BUCKET_NAME, SINGLE_FILE_KEY, DOWNLOAD_DIR)