import asyncio
import os
import sys
PYTHONPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)
from datetime import datetime
from copy import deepcopy
import json_repair
from typing import List, Dict, Any, Tuple, Optional
import aiofiles
from volcenginesdkarkruntime import AsyncArk
from core.llm import BGEEmbedding,  HierarchicalTextRecall
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from core.utils import extract_tags_and_content, extract_items, remove_image, remove_toc_lines_basic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.prompts import turn_2_prompt, turn_3_question_answer_prompt, evaluate_prompt
from config import Config
import logging
import json
from tqdm import tqdm
from glob import glob

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_text_recall_local(model_path="/home/yehongxing/workspace/shi/QA_Generate/bge-m3"):
    _text_recall = HierarchicalTextRecall(BGEEmbedding(model_path))
    return _text_recall

# 定义模块级的辅助函数，用于在子进程中处理内容
def _process_content_helper(file_path: str, content: str, content_splitter_params: dict, reference_splitter_params: dict) -> List[Dict[str, Any]]:
    # 创建splitter
    content_splitter = RecursiveCharacterTextSplitter(**content_splitter_params)
    reference_splitter = RecursiveCharacterTextSplitter(**reference_splitter_params)

    # 执行所有CPU密集型操作
    cleaned_content = remove_image(content)
    cleaned_content = remove_toc_lines_basic(cleaned_content)
    documents = content_splitter.create_documents([cleaned_content])
    
    file_name = os.path.basename(file_path)
    chunks = []
    
    for doc in documents:
        if len(doc.page_content) < 100:
            continue
            
        # 处理参考内容
        references_docs = reference_splitter.create_documents([doc.page_content])
        references = [ref.page_content for ref in references_docs if len(ref.page_content) > 100]
        
        chunks.append({
            'file_path': file_path,
            'file_name': file_name,
            'content': doc.page_content,
            'references': references
        })
    
    return chunks

class AsyncMultiTurnPipeline:
    def __init__(self, model_name: str, max_workers: int = 50):
        self.client = AsyncArk(
            api_key=Config.ARK_API_KEY,
            timeout=24 * 3600,
        )
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_try = Config.MAX_TRY or 5
        self.recall_system = get_text_recall_local(Config.EMBEDDING_MODEL_PATH)
        # self.reference_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        # self.content_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=2000)
        self.content_splitter_params = {'chunk_size': 8000, 'chunk_overlap': 2000}
        self.reference_splitter_params = {'chunk_size': 1000}
        self.questions = []
        self.questions_deduped = []
        # 根据系统资源设置池大小
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
        logger.info("步骤1: 内容分割和参考预处理")
        all_content_chunks = await self.process_step1(file_paths)
        
        # 步骤2: 并行生成所有内容块的问题对
        logger.info("步骤2: 生成问题对")
        all_question_pairs = await self.process_step2(all_content_chunks)  
        
        # 步骤3: 并行生成所有问题对的答案
        logger.info("步骤3: 生成答案")
        all_answers = await self.process_step3(all_question_pairs)  
        
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
        with tqdm(total=len(file_paths), desc="分割文件") as pbar:
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
        
        # 在主进程中，使用线程池将chunks添加到召回系统，并添加进度条
        if self.recall_system and hasattr(self.recall_system, 'add_file_texts'):
            # 按文件分组chunks
            file_to_chunks = {}
            for chunk in all_chunks:
                file_path = chunk['file_path']
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)
            
            # 使用进度条显示召回系统处理进度
            with tqdm(total=len(file_to_chunks), desc="添加到召回系统") as pbar:
                # 使用线程池添加每个文件的文本
                loop = asyncio.get_event_loop()
                for file_path, chunks in file_to_chunks.items():
                    await loop.run_in_executor(
                        self.io_thread_pool,
                        lambda file_path=file_path, chunks=chunks: self.recall_system.add_file_texts(file_path, chunks)
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(f"处理文件: {os.path.basename(file_path)}")
        
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
        
    async def process_step2(self, all_chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """步骤2: 生成问题对"""
        queue = asyncio.Queue()
        for chunk in all_chunks:
            for _ in range(self.max_try):
                await queue.put(chunk)
        
        # 放入停止标记
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(all_chunks) * self.max_try, desc="生成问题对")
        
        tasks = []
        for i in range(self.max_workers):
            task = asyncio.create_task(self._question_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        all_question_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
            else:
                all_question_pairs.extend(result)
        
        return all_question_pairs

    async def _question_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """问题生成工作协程"""
        question_pairs = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            
            try:
                # 为内容块生成问题对
                pairs = await self._generate_questions_for_chunk(chunk)
                if pairs:
                    question_pairs.extend(pairs)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return question_pairs
    
    async def _generate_questions_for_chunk(self, chunk: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """为内容块生成问题对（每个对两个有递进关系的问题）"""
        prompt = turn_2_prompt.format(content=chunk['content'])
        
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
        questions = extract_items(raw_questions, "Q")  # 返回问题列表，不去重
        
        # 确保至少有两个问题来形成一对
        if len(questions) != 2:
            return []
        
        
        # 将问题分成对：每两个问题作为一个对
        question_pairs = []
        for i in range(0, len(questions), 2):
            if i + 1 < len(questions):
                q1 = questions[i]
                q2 = questions[i+1]
                # 为每个问题找到最佳引用
                search_results1 = await self.recall_system.hierarchical_search(
                    q1, chunk['content'], chunk['file_path'], top_k=3
                )
                search_results2 = await self.recall_system.hierarchical_search(
                    q2, chunk['content'], chunk['file_path'], top_k=3
                )
                
                # 验证两个问题是否有效
                if search_results1 and self._is_valid_question(q1, search_results1) and \
                search_results2 and self._is_valid_question(q2, search_results2):
                    question_pairs.append([
                        {
                            'file_path': chunk['file_path'],
                            'file_name': chunk['file_name'],
                            'question': q1,
                            'references': search_results1,
                            'source_content': chunk['content'][:200] + "..."
                        },
                        {
                            'file_path': chunk['file_path'],
                            'file_name': chunk['file_name'],
                            'question': q2,
                            'references': search_results2,
                            'source_content': chunk['content'][:200] + "..."
                        }
                    ])
            else:
                # 如果剩余一个问题，无法形成对，跳过
                continue
        
        return question_pairs
    
    def _is_valid_question(self, question: str, references: List[Dict]) -> bool:
        """验证问题的有效性"""
        if not question or len(question.strip()) < 5:
            return False
        
        if not references or max(ref['similarity'] for ref in references) < 0.6:
            return False
        
        return True
    
    async def process_step3(self, all_question_pairs: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """步骤3: 生成答案，直接使用问题对"""
        # 创建任务队列
        queue = asyncio.Queue()
        for question_pair in all_question_pairs:
            await queue.put(question_pair)
        
        # 放入停止标记
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(all_question_pairs), desc="生成答案")
        
        # 启动工作协程
        tasks = []
        for i in range(self.max_workers):
            task = asyncio.create_task(self._answer_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有工作协程结束
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        # 收集所有答案对
        all_answer_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
            else:
                all_answer_pairs.extend(result)
        
        return all_answer_pairs
    
    async def _answer_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """答案生成工作协程"""
        answer_pairs = []
        while True:
            question_pair = await queue.get()
            if question_pair is None:
                queue.task_done()
                break
            
            try:
                # 处理问题对生成答案
                answer_pair = await self._generate_answers_for_pair(question_pair)
                answer_pairs.append(answer_pair)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                # 即使出错也添加一个空的对
                answer_pairs.append([None, None])
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return answer_pairs
    
    async def _generate_answers_for_pair(self, question_pair: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """为问题对生成答案，保持多轮结构"""
        q1, q2 = question_pair
        answer_pair = [None, None]
        
        # 处理第一个问题
        if q1 and q1['question']:
            answer1 = await self._generate_single_answer(q1)
            answer_pair[0] = answer1
        
        # 处理第二个问题（如果有）
        if q2 and q2['question']:
            answer2 = await self._generate_single_answer(q2)
            answer_pair[1] = answer2
        
        return answer_pair
    
    async def _generate_single_answer(self, question_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为单个问题生成答案"""
        # 构建参考文本
        search_text = ""
        for i, ref_data in enumerate(question_data['references']):
            search_text += f"\n" + "="*40 + f" Reference {i + 1} " + "="*40 + f"\n" + ref_data['text']
        
        # 生成答案提示
        prompt = turn_3_question_answer_prompt.format(
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
            
            # 使用预召回的引用
            ref_list = [ref_data['text'] for ref_data in question_data['references']]
            res["ref"] = list(set(ref_list[:2]))  # 取前2个引用
            
            # 添加文件信息
            res["file_path"] = question_data['file_path']
            res["file_name"] = question_data['file_name']
            res["source_content"] = question_data['source_content']
            
            return res
        
        return None

    async def process_step4_evaluate(self, all_answer_pairs: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """步骤4: 评估问答对，保持多轮结构"""
        # 创建任务队列 - 平铺所有答案以便并行处理
        flat_answers = []
        pair_indices = []  # 记录每个答案属于哪个pair和位置
        
        for pair_idx, pair in enumerate(all_answer_pairs):
            for pos, answer in enumerate(pair):
                if answer is not None:
                    flat_answers.append(answer)
                    pair_indices.append((pair_idx, pos))
        
        # 如果没有需要评估的答案，直接返回
        if not flat_answers:
            return all_answer_pairs
        
        queue = asyncio.Queue()
        for answer in flat_answers:
            await queue.put(answer)
        
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(flat_answers), desc="评估问答对")
        
        # 启动工作协程
        tasks = []
        for i in range(min(self.max_workers, len(flat_answers))):
            task = asyncio.create_task(self._evaluate_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有工作协程结束
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        # 收集所有评估后的答案
        evaluated_flat_answers = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"Task failed: {result}")
            else:
                evaluated_flat_answers.extend(result)
        
        # 将评估后的答案重新组织回多轮结构
        evaluated_answer_pairs = deepcopy(all_answer_pairs)
        
        for eval_answer, (pair_idx, pos) in zip(evaluated_flat_answers, pair_indices):
            if pair_idx < len(evaluated_answer_pairs) and pos < len(evaluated_answer_pairs[pair_idx]):
                evaluated_answer_pairs[pair_idx][pos] = eval_answer
        
        return evaluated_answer_pairs
    
    async def process_step5(self, file_paths: List[str], all_answer_pairs: List[List[Dict[str, Any]]], output_dir: str):
        """步骤5: 整理和保存结果，保持多轮结构"""
        # 按文件分组答案对
        file_answer_pairs = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_answer_pairs[file_name] = []
        
        # 将答案对按文件分组
        for pair in all_answer_pairs:
            if pair and pair[0] and pair[0]["file_name"] in file_answer_pairs:
                file_answer_pairs[pair[0]["file_name"]].append(pair)
        
        # 为每个文件保存结果
        total = 0
        for file_name, answer_pairs in tqdm(file_answer_pairs.items(), desc="保存结果"):
            if answer_pairs:
                # 过滤掉空的对
                valid_pairs = [pair for pair in answer_pairs if any(pair)]
                if valid_pairs:
                    file_basename = file_name.replace(".md", "")
                    output_path = os.path.join(output_dir, f"{file_basename}.json")
                    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(valid_pairs, ensure_ascii=False, indent=2))
                    logger.info(f"已保存 {len(valid_pairs)} 个多轮问答对到 {output_path}")
                    total += len(valid_pairs)
        logger.info(f"共计生成{total}个问答对")
    
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



async def async_qa_evaluate(qa: dict, ref=None, client: AsyncArk = None, model_name: str = None):
    """异步版本的QA评估函数"""
    tmp = deepcopy(qa)
    if ref:
        tmp["ref"] = ref
    
    prompt = evaluate_prompt.format(QA_pair=json.dumps(tmp, ensure_ascii=False, indent=4))
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
    output_dir = os.path.join(output_dir, "multiturn")
    os.makedirs(output_dir, exist_ok=True)
    max_workers = Config.MAX_WORKERS  # 根据系统资源调整
    
    # 获取所有Markdown文件
    file_paths = []
    for input_dir in input_dirs:
        file_paths += glob(os.path.join(input_dir, "**/*.md"), recursive=True)
    
    if not file_paths:
        logger.error("未找到Markdown文件")
        return
    
    logger.info(f"找到 {len(file_paths)} 个Markdown文件")
    # 创建处理器并处理文件
    processor = AsyncMultiTurnPipeline(model_name, max_workers)
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
    asyncio.run(main())