import asyncio
import sys
import os
import json
from datetime import datetime
from openai import AsyncOpenAI
import traceback
from pathlib import Path
from io import BytesIO
import base64
from tqdm import tqdm
import threading
from PIL import Image
import io

# 全局变量用于跟踪进度
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.success = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total, desc="Processing", unit="task")
    
    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if success:
                self.success += 1
            else:
                self.errors += 1
            
            self.pbar.set_postfix({
                'Done': f'{self.completed}/{self.total}',
                'Success': self.success,
                'Errors': self.errors
            })
            self.pbar.update(1)
    
    def close(self):
        self.pbar.close()

progress_tracker = None

async def worker(
    worker_id: int,
    client: AsyncOpenAI,
    requests: asyncio.Queue[dict],
    semaphore: asyncio.Semaphore,
    result_file: str,
):
    """异步工作协程，处理请求并保存结果"""
    print(f"Worker {worker_id} is starting.")
    while True:
        request_data = await requests.get()
        try:
            api_params = request_data["api_params"]
            metadata = request_data.get("metadata", {})
            # 用 semaphore 控制并发
            async with semaphore:
                completion = await client.chat.completions.create(**api_params)
            # 构建结果
            result_data = {"img": metadata['image_path'], "latex": completion.choices[0].message.content}
            # 保存结果
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            
            # 更新进度（成功）
            if progress_tracker:
                progress_tracker.update(success=True)
                
            # print(
            #     f"Worker {worker_id} processed request: {metadata.get('request_id', 'unknown')}"
            # )
        except Exception as e:
            error_data = {
                "metadata": request_data.get("metadata", {}),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            os.makedirs("results", exist_ok=True)
            with open(
                f"/home/baoyang/code/nlp-baoy/projects/stepfun/latex_data/results/worker_{worker_id}_errors.jsonl", "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
            
            # 更新进度（失败）
            if progress_tracker:
                progress_tracker.update(success=False)
                
            print(f"Error in worker {worker_id}: {e}", file=sys.stderr)
        finally:
            requests.task_done()

async def main(conversations, base_url="http://localhost:8001/v1", max_concurrency=16, result_file=''):
    """主函数，处理并发请求"""
    start = datetime.now()
    os.makedirs("results", exist_ok=True)
    
    # 初始化进度跟踪器
    global progress_tracker
    progress_tracker = ProgressTracker(len(conversations))
    
    # 控制最大并发
    semaphore = asyncio.Semaphore(max_concurrency)
    # 创建队列
    requests = asyncio.Queue()
    # OpenAI API 兼容客户端
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=24 * 3600,
    )
    
    # 把任务放进队列
    for conv in conversations:
        await requests.put(
            {
                "api_params": conv["api_params"],
                "metadata": conv.get("metadata", {}),
            }
        )
    
    # 启动 worker
    worker_num = min(max_concurrency, len(conversations))
    tasks = [
        asyncio.create_task(worker(i, client, requests, semaphore, result_file))
        for i in range(worker_num)
    ]
    
    # 等待所有任务完成
    await requests.join()
    
    # 关闭进度跟踪器
    if progress_tracker:
        progress_tracker.close()
    
    # 清理
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()
    end = datetime.now()
    print(f"\nTotal time: {end - start}")
    print(f"Total tasks: {len(conversations)}")
    if progress_tracker:
        print(f"Successful: {progress_tracker.success}, Errors: {progress_tracker.errors}")

def get_img_base64(image_path) -> str:
    """处理完整路径的函数"""
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base_str

def resize_image_to_base64(image_path, max_size):
    """
    处理图片：如果超过指定大小则等比例缩放，并返回base64编码
    
    参数:
        image_path: 图片文件路径
        max_size: 最大尺寸阈值（像素），宽或高超过此值将被缩放
        
    返回:
        缩放后图片的base64编码字符串
    """
    # 打开图片
    with Image.open(image_path) as img:
        # 获取原始尺寸
        width, height = img.size
        print(f"原始图片尺寸: {width}x{height}")
        
        # 检查是否需要缩放
        if width > max_size or height > max_size:
            # 计算缩放比例
            if width > height:
                ratio = max_size / width
            else:
                ratio = max_size / height
                
            # 计算新尺寸
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            print(f"缩放后图片尺寸: {new_width}x{new_height}")
            
            # 等比例缩放图片，使用高质量缩放算法
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            print("图片无需缩放")
            resized_img = img
        
        # 将图片转换为base64
        buffer = io.BytesIO()
        # 保存图片到缓冲区，使用原图格式或默认JPEG
        format = img.format if img.format else 'JPEG'
        resized_img.save(buffer, format=format)
        
        # 获取缓冲区内容并转换为base64
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str        


if __name__ == '__main__':
    text_prompt = """
使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
4. 遇到表格请用markdown的表格符号|---|显示出来。
5. 不能丢失任何文本信息，否则你将受到惩罚。
 
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
图片中用红色框和名称(0_0.png)标注出了一些区域。
如果区域是图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
    """
    subfolder = 'subfolder_4'
    
    print(f"loading images ...")
    pngs = list(Path(f'/mnt/afs/projects/train_data/latex/corpus/{subfolder}').glob('**/*.jpg'))
    result_file = f"/home/baoyang/code/nlp-baoy/projects/stepfun/latex_data/latex_train_{subfolder}.jsonl"


    conversations = []
    for i, image_path in enumerate(pngs):

        max_pix = 2230
        img_base64 = get_img_base64(image_path)
        
        with Image.open(image_path) as img:
            # 获取原始尺寸
            width, height = img.size
            if width > max_pix or height >max_pix:
                img_base64 = resize_image_to_base64(image_path, max_pix)


        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        },
                    },
                ],
            }
        ]
        conversations.append(
            {
                "api_params": {
                    "model": "qwenvl",
                    "messages": messages,
                    "temperature": 0.4,
                },
                "metadata": {"request_id": f"req_{i}", "image_path": str(image_path)},
            }
        )
    
    print(f"Starting processing of {len(conversations)} images...")
    asyncio.run(
        main(
            conversations,
            base_url="http://localhost:22223/v1",
            max_concurrency=4,
            result_file=result_file
        )
    )