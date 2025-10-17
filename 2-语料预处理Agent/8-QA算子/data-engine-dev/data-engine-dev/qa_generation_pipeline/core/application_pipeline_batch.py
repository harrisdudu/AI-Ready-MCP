import asyncio
import os
import sys
PYTHONPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)
from datetime import datetime
from copy import deepcopy
import json_repair
from typing import List, Dict, Any
import aiofiles
from volcenginesdkarkruntime import AsyncArk
from core.llm import BGEEmbedding,  HierarchicalTextRecall
from core.utils import extract_tags_and_content, extract_items, remove_image, remove_toc_lines_basic
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 修改1: 使用正确的prompt名称
from core.prompts import application_question_prompt, application_question_answer_prompt, evaluate_prompt_common
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import json
from tqdm import tqdm
from glob import glob  
from config import Config
  
def get_text_recall_local(model_path=r"C:\Users\13122\.cache\modelscope\hub\models\BAAI\bge-m3"):
    _text_recall = HierarchicalTextRecall(BGEEmbedding(model_path))
    return _text_recall

# 定义模块级的辅助函数，用于在子进程中处理内容
def _process_content_helper(file_path: str, content: str, content_splitter_params: dict, reference_splitter_params: dict) -> List[Dict[str, Any]]:
    # 修改2: 移除切分逻辑，将整个文件作为一个块处理
    file_name = os.path.basename(file_path)
    
    # 清理内容
    cleaned_content = remove_image(content)
    cleaned_content = remove_toc_lines_basic(cleaned_content)
    
    # 将整个文件内容作为一个块
    chunks = [{
        'file_path': file_path,
        'file_name': file_name,
        'content': cleaned_content,
        'references': [cleaned_content]  # 整个内容作为参考
    }]
    
    return chunks
  
# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
  
class BatchBookProcessor:
    def __init__(self, model_name: str, max_workers: int = 100):
        self.client = AsyncArk(
            api_key=Config.ARK_API_KEY,
            timeout=24 * 3600,
        )
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_try = Config.MAX_TRY or 3
        self.recall_system = get_text_recall_local(Config.EMBEDDING_MODEL_PATH)
        # 修改3: 移除切分参数，因为我们不再切分
        self.content_splitter_params = {'chunk_size': 8000, 'chunk_overlap': 2000}
        self.reference_splitter_params = {'chunk_size': 500}
        self.cpu_count = os.cpu_count() or 4
        self.io_thread_pool = ThreadPoolExecutor(max_workers=min(32, self.cpu_count * 4))  # I/O密集型线程池
        self.cpu_process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)  # CPU密集型进程池
        # 队列大小限制，防止内存溢出
        self.queue_max_size = self.cpu_count * 8
    
    async def process_files(self, file_paths: List[str], output_dir: str):
        """批量处理多个文件"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 并行处理所有文件的内容分割和参考预处理
        logger.info("步骤1: 内容预处理")
        all_content_chunks = await self.process_step1(file_paths)
  
        # 步骤2: 并行生成所有内容块的问题
        logger.info("步骤2: 生成问题")
        all_questions = await self.process_step2(all_content_chunks)
        
        # 步骤3: 并行生成所有问题的答案
        logger.info("步骤3: 生成答案")
        all_answers = await self.process_step3(all_questions)
        
        # 步骤4: 并行评估所有问答对
        logger.info("步骤4: 评估问答对")
        evaluated_answers = await self.process_step4_evaluate(all_answers)
        
        # 步骤5: 按文件整理并保存结果
        logger.info("步骤5: 整理和保存结果")
        await self.process_step5(file_paths, evaluated_answers, output_dir)
        
        logger.info("处理完成!")
    
    async def process_step1(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """优化后的步骤1: 使用生产者-消费者模式处理文件"""
        # 创建带缓冲的队列
        queue = asyncio.Queue(maxsize=self.queue_max_size)
        results = []
        
        # 生产者任务：读取文件内容
        async def producer():
            for file_path in file_paths:
                try:
                    # 使用I/O线程池读取文件
                    loop = asyncio.get_event_loop()
                    content = await loop.run_in_executor(
                        self.io_thread_pool, 
                        self._sync_read_file, 
                        file_path
                    )
                    await queue.put((file_path, content))
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 失败: {e}")
                    # 放入特殊标记表示处理失败
                    await queue.put((file_path, None))
            
            # 放入结束标记
            for _ in range(self.cpu_count):
                await queue.put((None, None))
        
        # 消费者任务：处理文件内容（只分割文本，不调用召回系统）
        async def consumer(consumer_id: int, progress_bar: tqdm):
            nonlocal results
            while True:
                file_path, content = await queue.get()
                
                # 检查结束标记
                if file_path is None:
                    queue.task_done()
                    break
                
                if content is None:
                    results.append([])
                    queue.task_done()
                    progress_bar.update(1)
                    continue
                
                try:
                    # 使用进程池处理CPU密集型任务（文本分割）
                    loop = asyncio.get_event_loop()
                    chunks = await loop.run_in_executor(
                        self.cpu_process_pool,
                        _process_content_helper,
                        file_path,
                        content,
                        self.content_splitter_params,
                        self.reference_splitter_params
                    )
                    
                    results.append(chunks)
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 失败: {e}")
                    results.append([])
                finally:
                    queue.task_done()
                    progress_bar.update(1)
        
        # 启动处理流程
        with tqdm(total=len(file_paths), desc="处理文件") as pbar:
            # 启动生产者
            producer_task = asyncio.create_task(producer())
            
            # 启动多个消费者（数量等于CPU核心数）
            consumer_tasks = [
                asyncio.create_task(consumer(i, pbar)) 
                for i in range(self.cpu_count)
            ]
            
            # 等待所有任务完成
            await producer_task
            await queue.join()
            for task in consumer_tasks:
                task.cancel()
            await asyncio.gather(*consumer_tasks, return_exceptions=True)
        
        # 合并所有内容块
        all_chunks = [chunk for result in results for chunk in result]
        
        # 修改4: 移除召回系统的添加，因为我们不再需要切分
        return all_chunks
    
    def _sync_read_file(self, file_path: str) -> str:
        """同步读取文件内容（供I/O线程池使用）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            logger.error(f"文件编码错误: {file_path}")
            raise
        except Exception as e:
            logger.error(f"读取文件 {file_path} 出错: {e}")
            raise

    def shutdown(self):
        """关闭线程池和进程池"""
        self.io_thread_pool.shutdown()
        self.cpu_process_pool.shutdown()
    
    async def process_step2(self, all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤2: 生成问题"""
        queue = asyncio.Queue()
        for chunk in all_chunks:
            for _ in range(self.max_try):
                await queue.put(chunk)
        
        # 放入停止标记
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(all_chunks) * self.max_try, desc="生成问题")
        
        tasks = []
        for i in range(self.max_workers):
            task = asyncio.create_task(self._question_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        all_questions = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"Task failed: {result}")
            else:
                all_questions.extend(result)
        
        return all_questions
    
    async def _question_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        questions = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            try:
                chunk_questions = await self._generate_questions_for_chunk(chunk)
                if chunk_questions:
                    questions.extend(chunk_questions)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return questions
    
    async def _generate_questions_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为内容块生成问题"""
        # 修改5: 使用正确的prompt名称
        prompt = application_question_prompt.format(
            content=chunk['content'],
            title=chunk['file_name']
        )
        # 使用异步LLM调用
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个有帮助的AI助手",
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        raw_questions = completion.choices[0].message.content
        questions = list(set(extract_items(raw_questions, "Q")))
  
        if not questions:
            return []
        
        # 修改6: 移除召回系统的使用，因为我们不再需要切分
        question_data = []
        for question in questions:
            # 直接使用整个内容作为参考
            question_data.append({
                'file_path': chunk['file_path'],
                'file_name': chunk['file_name'],
                'question': question,
                'references': [chunk['content']],  # 整个内容作为参考
                'source_content': chunk['content'][:200] + "..."  # 截取前200字作为来源提示
            })
  
        return question_data
    
    def _is_valid_question(self, question: str, references: List[Dict]) -> bool:
        """验证问题的有效性"""
        if "是否" in question or "是不是" in question:
            return False
        # 修改7: 简化验证逻辑，因为我们不再使用召回系统
        if not references:
            return False
        return True
    
    async def process_step3(self, all_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤3: 生成答案"""
        # 创建任务队列
        queue = asyncio.Queue()
        for question_data in all_questions:
            await queue.put(question_data)
  
        # 放入停止标记
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(all_questions), desc="生成答案")
        
        # 启动工作协程
        tasks = []
        for i in range(self.max_workers):
            task = asyncio.create_task(self._answer_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有工作协程结束
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
  
        all_answers = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"Task failed: {result}")
            else:
                all_answers.extend(result)
  
        return all_answers
    
    async def _answer_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """答案生成工作协程"""
        answers = []
        while True:
            question_data = await queue.get()
            if question_data is None:
                queue.task_done()
                break
            try:
                # 处理问题生成答案
                answer = await self._generate_answer_for_question(question_data)
                if answer:
                    answers.append(answer)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return answers
    
    async def _generate_answer_for_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """为问题生成答案"""
        # 构建参考文本：使用整个文件内容
        search_text = question_data['references'][0]  # 整个文件内容
        
        # 生成答案提示
        # 修改8: 使用正确的prompt名称
        prompt = application_question_answer_prompt.format(
            title = question_data["file_name"],
            question=question_data['question'],
            content=search_text
        )
        
        # 使用异步LLM调用
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个有帮助的AI助手",
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        response = completion.choices[0].message.content
        res = extract_tags_and_content(response)
        
        if list(res.keys()) == ["Q", "COT", "A"]:
            if res["A"] == "Reject" or len(res["A"]) < 30:
                return None
            
            # 使用整个文件内容作为引用
            res["ref"] = [search_text]  # 整个文件内容
            
            # 添加文件信息
            res["file_path"] = question_data['file_path']
            res["file_name"] = question_data['file_name']
            res["source_content"] = question_data['source_content']
            
            return res
        
        return None
    
    async def process_step4_evaluate(self, all_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤4: 评估问答对"""
        # 过滤掉None值
        valid_answers = [answer for answer in all_answers if answer]
        
        # 创建任务队列
        queue = asyncio.Queue()
        for answer in valid_answers:
            await queue.put(answer)
  
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(valid_answers), desc="评估问答对")
        
        # 启动工作协程
        tasks = []
        for i in range(min(self.max_workers, len(valid_answers))):
            task = asyncio.create_task(self._evaluate_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有工作协程结束
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        # 收集所有评估后的答案
        evaluated_answers = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"Task failed: {result}")
            else:
                evaluated_answers.extend(result)
        
        return evaluated_answers
  
    async def _evaluate_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """评估工作协程"""
        evaluated_answers = []
        while True:
            answer = await queue.get()
            if answer is None:
                queue.task_done()
                break
            try:
                # 评估问答对
                evaluated_answer = await self._evaluate_single_qa(answer)
                evaluated_answers.append(evaluated_answer)
            except Exception as e:
                logger.error(f"Evaluate worker {worker_id} error: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return evaluated_answers
    
    async def _evaluate_single_qa(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个QA对"""
        try:
            # 使用异步评估函数
            evaluated_answer = await async_qa_evaluate(
                answer,
                ref=answer.get("ref"),
                client=self.client,
                model_name=self.model_name
            )
            return evaluated_answer
        except Exception as e:
            logger.error(f"Failed to evaluate QA: {str(e)}")
            answer["judge"] = {"error": str(e)}
            return answer
    
    async def process_step5(self, file_paths: List[str], all_answers: List[Dict[str, Any]], output_dir: str):
        """步骤5: 整理和保存结果"""
        # 按文件分组答案
        file_answers = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_answers[file_name] = []
        
        # 修改9: 添加筛选逻辑，根据新的评分标准
        filtered_answers = []
        for answer in all_answers:
            if answer and answer.get("judge"):
                # 检查judge是否是一个字典，且包含各个维度的分数
                if isinstance(answer["judge"], dict):
                    scores = answer["judge"]
                    # 筛选标准: 所有维度分数都>=2，且幻觉指数必须为5
                    required_keys = ['Completeness & Clarity', 'Technical Depth', 'Knowledge & Analysis', 
                                    'Accuracy & Citation Transparency', 'Hallucination Index', 'Difficulty & Reasoning Depth', "Category"]
                    
                    # 检查所有必需的键都存在
                    if all(key in scores for key in required_keys):
                        # 检查分数是否符合要求
                        if (scores['Hallucination Index'] == 5 and 
                            all(scores[key] >= 2 for key in required_keys if key != 'Hallucination Index')):
                            filtered_answers.append(answer)
        
        # 将筛选后的答案按文件分组
        for answer in filtered_answers:
            if answer["file_name"] in file_answers:
                file_answers[answer["file_name"]].append(answer)
        
        # 为每个文件保存结果
        total = 0
        for file_name, answers in tqdm(file_answers.items(), desc="保存结果"):
            if answers:
                file_basename = file_name.replace(".md", "")
                output_path = os.path.join(output_dir, f"{file_basename}.json")
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(answers, ensure_ascii=False, indent=2))
                logger.info(f"已保存 {len(answers)} 个问答对到 {output_path}")
                total += len(answers)
        logger.info(f"共计生成{total}个问答对")
  
  
async def async_qa_evaluate(qa: dict, ref=None, client: AsyncArk = None, model_name: str = None):
    """异步版本的QA评估函数"""
    tmp = deepcopy(qa)
    if ref:
        tmp["ref"] = ref
    
    tmp.pop("COT")
    prompt = evaluate_prompt_common.format(QA_pair=json.dumps(tmp, ensure_ascii=False, indent=4))
    response = None
    try:
        # 使用异步客户端调用
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的QA对评估专家",
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        response = completion.choices[0].message.content
        response_json = json_repair.loads(response)
        qa["judge"] = response_json
    except Exception as e:
        logger.error(f"Failed to evaluate QA: {str(e)}")
        qa["judge"] = {"error": str(e), "response": response}
    
    return qa

async def main():
    """主函数"""
    # 配置参数
    model_name = Config.MODEL_NAME  # 替换为实际模型名称
    input_dirs = Config.INPUT_DIRS  # 替换为输入目录
    output_dir = Config.OUTPUT_DIR  # 替换为输出目录
    output_dir = os.path.join(output_dir, "application")
    os.makedirs(output_dir, exist_ok=True)
    max_workers = Config.MAX_WORKERS  # 根据系统资源调整
    
    # 获取所有Markdown文件
    file_paths = []
    file_paths = []
    for input_dir in input_dirs:
        file_paths += glob(os.path.join(input_dir, "**/*.md"), recursive=True)
    
    if not file_paths:
        logger.error("未找到Markdown文件")
        return
    
    logger.info(f"找到 {len(file_paths)} 个Markdown文件")
    
    # 创建处理器并处理文件
    processor = BatchBookProcessor(model_name, max_workers)
    try:
        # 处理完整的分片
        for i in range(0, len(file_paths), 100):
            logger.info(f"正在处理分片 {i//100 + 1}")
            batch = file_paths[i: i+100] 
            try:
                await processor.process_files(batch, output_dir)
            except Exception as e:
                logger.error(f"处理分片 {i//100 + 1} 时出错: {e}")
                # 可以在这里添加重试逻辑或错误记录

    finally:
        processor.shutdown()  

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())