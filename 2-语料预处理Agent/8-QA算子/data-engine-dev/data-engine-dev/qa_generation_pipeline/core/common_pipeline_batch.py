import asyncio
import os
import sys
PYTHONPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)
import logging
import json
import re
from datetime import datetime
from copy import deepcopy
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from volcenginesdkarkruntime import AsyncArk
from glob import glob
from config import Config
import aiofiles
import json_repair
from tqdm import tqdm
import traceback
# Import custom modules
from core.utils import read_markdown, remove_image, remove_table
from core.prompts import evaluate_prompt
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 直接从原始common pipeline中复制的函数
def split_markdown_by_headers(markdown_text: str, merge_headers: bool = True) -> List[str]:
    lines = markdown_text.splitlines()
    sections = []
    current_section = []
    in_headers_block = False
    
    for line in lines:
        if re.match(r'^\s{0,3}#{1,6}\s+', line):
            if in_headers_block and merge_headers:
                current_section.append(line)
            else:
                if current_section:
                    sections.append("\n".join(current_section).strip("\n"))
                current_section = [line]
                in_headers_block = True
        else:
            in_headers_block = False
            if not line.strip() and not current_section:
                continue
            current_section.append(line)
    
    if current_section:
        sections.append("\n".join(current_section).strip("\n"))
    
    return sections
# Common Pipeline的提示词模板
QA_generate_prompt_per_block = """
你现在正在帮部门将文档电子化，归档入库，请你放心生成，现在给你一篇政府文档的部分内容
请你根据所给标题和文本内容提问题，问题数量可以遵循文本内容中的分点内容的数量（分点例如1. 2. 3. ），也可以是一个覆盖全部分点内容的概述型问题。
分析型问题 （涉及业务中最需要评价的对象，分析对象作用、策略等）
总结型问题 （涉及到一个事件的总结概述， 事件两面的对比）
知识性材料要求逻辑推导能力，提问以分析类为主（如作用、策略等），回答内容适度自由，需要思维链。
问题范围需要覆盖到具体的对象，问题需要涉及一定的总结和分析，拒绝对概念定义或技术类（技术参数，技术细节）的询问。
问题主语需基于文档标题或内容补充完整定语（如当文件标题为‘厨房设备灭火装置’时，将问题中‘启动方式’修正为‘厨房设备灭火装置的启动方式’）

要求：
-  如果QA涉及到条款引用，请在答案和思维链中说明
- 在生成问题时，需基于文档标题与内容上下文，对问题中的主语进行定语补充，确保主语完整且无歧义。
- 问题中**禁止**使用任何具体的法规、标准、图名、表名或文件名称。无需出现 ‘根据规范’ 等表述。
- 问题和答案的主语**禁止**出现模糊的指代型词汇，如‘本文’，‘本规范’。
- 禁止对图片或者表格提问

#### 三、回答要求（符合筛选标准的问题）
1. **零自由原则**：  
   - 严格依据Reference原文内容回答，禁止任何脱离原文的总结、概括、转述或补充，需要包含**所有**相关文本，不用过于精简；  
   - 禁止使用“等”“之类”等省略词汇，需完整呈现原文所有相关内容；  
   - 标点符号、表述逻辑需与原文保持一致（原文分点则Answer分点，原文段落则Answer保持段落）。  

2. **COT（思维链）要求**：  
   - 需明确指出答案对应的**具体条款章节**（如“第三章第一节”“总则第5条”）；  
   - 需完整摘录该章节中与问题相关的原文细节（不可简写）。 
   - 禁用“原文”“文中”等模糊表述，直接呈现条款章节和原文内容。  

   - 回答必须基于提供的reference，但必须保证任何一个没有看到reference的用户也能完全理解你的答案。这意味着你必须避免任何对reference的间接指代。你不能在输出思维链中提到‘reference’， ‘ref’ 等字眼。
   【引用格式】当引用reference中的内容时，你必须：
   1.  首先明确指出引用的来源名称。例如：“根据《飞机库设计防火规范 GB50284-2008》总则第1.0.3条，...”。
   2.  首次提及后，可简称为“该规范”或“《规范》”，但仍需说明条款号，如“该规范第3.0.1条要求...”。
   3.  对于案例，需说明案例出处，如“该规范在第三章的案例说明中，记载了1965年美国迈阿密机场的火灾案例...”。

   【严格禁止】绝对不允许出现以下情况：
   -   **错误示例1（不指明来源）**： “如1.0.3指出的...”（❌ 用户不知道1.0.3是什么）
   -   **错误示例2（模糊指代）**： “根据第三条...”、“如案例所述...”（❌ 指向不明）
   -   **错误示例3（依赖隐藏上下文）**： “首都机场案例中的数据...”（❌ 必须说明这个案例出自提供的规范上下文）

定语补充示例：
输入问题："启动方式有几种？"
文档标题/内容主题："厨房设备灭火装置技术规范"
输出问题："厨房设备灭火装置的启动方式有几种？"


文件标题/解析：
{title}
内容：
{content}
输出格式：
[
{{
"Q":"",
"COT": "",
"A":"",
}}
]
"""
class BatchCommonProcessor:
    def __init__(self, model_name: str, max_workers: int = 50, evaluate_model_name=None):
        self.client = AsyncArk(
            api_key=Config.ARK_API_KEY,
            timeout=24 * 3600,
        )
        self.model_name = model_name
        self.evaluate_model_name = evaluate_model_name or model_name
        self.max_workers = max_workers
        self.max_try = Config.MAX_TRY or 3
        
        # IO线程池
        self.cpu_count = os.cpu_count() or 4
        self.io_thread_pool = ThreadPoolExecutor(max_workers=min(32, self.cpu_count * 4))
        
        # 队列大小限制，防止内存溢出
        self.queue_max_size = self.cpu_count * 8
        
        # 添加块顺序计数器
        self.block_counter = {}
        
    async def process_files(self, file_paths: List[str], output_dir: str, topic_mapping: Dict[str, str] = None):
        """批量处理多个文件生成QA对"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化块计数器
        for file_path in file_paths:
            file_name = os.path.basename(file_path).replace('.md', '')
            self.block_counter[file_name] = 0
        
        # 步骤1: 读取文件并分割内容
        logger.info("步骤1: 读取和分割文件内容")
        all_content_chunks = await self.process_step1(file_paths, topic_mapping)
        
        # 步骤2: 为每个内容块生成QA对
        logger.info("步骤2: 生成QA对")
        all_qa_pairs = await self.process_step2(all_content_chunks)
        
        # 步骤3: 评估所有QA对
        logger.info("步骤3: 评估QA对")
        evaluated_qa_pairs = await self.process_step3_evaluate(all_qa_pairs)
        
        # 步骤4: 按文件整理并保存结果
        logger.info("步骤4: 整理和保存结果")
        await self.process_step4(file_paths, evaluated_qa_pairs, output_dir)
        
        logger.info("处理完成!")
        
    async def process_step1(self, file_paths: List[str], topic_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """步骤1: 读取文件并按标题分割内容"""
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
                    logger.error(traceback.format_exc())
                    # 放入特殊标记表示处理失败
                    await queue.put((file_path, None))
            
            # 放入结束标记
            for _ in range(self.cpu_count):
                await queue.put((None, None))
        
        # 消费者任务：处理文件内容
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
                    # 处理文件内容
                    file_name = os.path.basename(file_path).replace('.md', '')
                    file_desc = topic_mapping.get(file_name, file_name) if topic_mapping else file_name
                    
                    # 清洗内容
                    cleaned_content = remove_image(content)
                    cleaned_content = remove_table(cleaned_content)
                    
                    # 按标题分割内容
                    blocks = split_markdown_by_headers(cleaned_content, True)
                    
                    chunks = []
                    for block_index, block in enumerate(blocks):
                        if len(block) < 100 or "........" in block:
                            continue
                        chunks.append({
                            'file_path': file_path,
                            'file_name': file_name,
                            'file_desc': file_desc,
                            'content': block,
                            'block_index': block_index  # 记录块在文档中的顺序
                        })
                    
                    results.append(chunks)
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 失败: {e}")
                    logger.error(traceback.format_exc())
                    results.append([])
                finally:
                    queue.task_done()
                    progress_bar.update(1)
        
        # 启动处理流程
        with tqdm(total=len(file_paths), desc="处理文件") as pbar:
            # 启动生产者
            producer_task = asyncio.create_task(producer())
            
            # 启动多个消费者
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
            
    async def process_step2(self, all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤2: 为每个内容块生成QA对"""
        # 创建任务队列
        queue = asyncio.Queue()
        for chunk in all_chunks:
            for _ in range(self.max_try):
                await queue.put(chunk)
        
        # 放入停止标记
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(all_chunks) * self.max_try, desc="生成QA对")
        
        tasks = []
        for i in range(self.max_workers):
            task = asyncio.create_task(self._qa_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        all_qa_pairs = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"任务失败: {result}")
            else:
                all_qa_pairs.extend(result)
        
        return all_qa_pairs
        
    async def _qa_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """QA生成工作协程"""
        qa_pairs = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            
            try:
                # 为内容块生成QA对
                chunk_qa_pairs = await self._generate_qa_for_chunk(chunk)
                if chunk_qa_pairs:
                    qa_pairs.extend(chunk_qa_pairs)
            except Exception as e:
                logger.error(f"Worker {worker_id} 错误: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return qa_pairs
        
    async def _generate_qa_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为内容块生成QA对"""
        # 构建提示词
        prompt = QA_generate_prompt_per_block.format(
            title=chunk['file_desc'],
            content=chunk['content']
        )
        
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        
        try:
            # 解析响应
            qa_pairs = json_repair.loads(response)
            if not qa_pairs:
                return []
            
            # 确保是列表格式
            if not isinstance(qa_pairs, list):
                qa_pairs = [qa_pairs]
            
            # 为每个QA对添加元数据
            for qa_pair in qa_pairs:
                if "Q" in qa_pair and "A" in qa_pair and "COT" in qa_pair:
                    # 清理问题中的类型前缀
                    types = ["总结型：", "对比型：", "逻辑推理型：", "概念问答型："]
                    for t in types:
                        qa_pair["Q"] = qa_pair["Q"].replace(t, "")
                        qa_pair["A"] = qa_pair["A"].replace(t, "")
                    
                    # 添加元数据
                    qa_pair["file_path"] = chunk['file_path']
                    qa_pair["file_name"] = chunk['file_name']
                    qa_pair["ref"] = [chunk['content']]
                    qa_pair["block_index"] = chunk['block_index']  # 保留块的顺序信息
            
            return qa_pairs
        except Exception as e:
            logger.error(f"解析QA对失败: {str(e)}")
            logger.error(f"响应内容: {response}")
            return []
            
    async def process_step3_evaluate(self, all_qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤3: 评估QA对"""
        # 过滤掉None值
        valid_qa_pairs = [qa for qa in all_qa_pairs if qa]
        
        # 创建任务队列
        queue = asyncio.Queue()
        for qa_pair in valid_qa_pairs:
            await queue.put(qa_pair)
        
        for _ in range(self.max_workers):
            await queue.put(None)
        
        # 创建进度条
        pbar = tqdm(total=len(valid_qa_pairs), desc="评估QA对")
        
        # 启动工作协程
        tasks = []
        for i in range(min(self.max_workers, len(valid_qa_pairs))):
            task = asyncio.create_task(self._evaluate_worker(i, queue, pbar))
            tasks.append(task)
        
        # 等待所有工作协程结束
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 关闭进度条
        pbar.close()
        
        # 收集所有评估后的QA对
        evaluated_qa_pairs = []
        for result in results:
            if isinstance(result, Exception):
                # 记录异常
                logger.error(f"任务失败: {result}")
            else:
                evaluated_qa_pairs.extend(result)
        
        return evaluated_qa_pairs
        
    async def _evaluate_worker(self, worker_id: int, queue: asyncio.Queue, pbar: tqdm):
        """评估工作协程"""
        evaluated_qa_pairs = []
        while True:
            qa_pair = await queue.get()
            if qa_pair is None:
                queue.task_done()
                break
            
            try:
                # 评估QA对
                evaluated_qa = await self._evaluate_single_qa(qa_pair)
                evaluated_qa_pairs.append(evaluated_qa)
            except Exception as e:
                logger.error(f"评估Worker {worker_id} 错误: {str(e)}")
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix_str(f"Worker {worker_id} 处理中")
        
        return evaluated_qa_pairs
        
    async def _evaluate_single_qa(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个QA对"""
        try:
            # 使用异步评估函数
            evaluated_qa = await async_qa_evaluate_common(
                qa_pair,
                ref=qa_pair.get("ref"),
                client=self.client,
                model_name=self.evaluate_model_name
            )
            return evaluated_qa
        except Exception as e:
            logger.error(f"评估QA对失败: {str(e)}")
            qa_pair["judge"] = {"error": str(e)}
            return qa_pair
            
    async def process_step4(self, file_paths: List[str], all_qa_pairs: List[Dict[str, Any]], output_dir: str):
        """步骤4: 整理和保存结果"""
        # 按文件分组QA对
        file_qa_pairs = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path).replace('.md', '')
            file_qa_pairs[file_name] = []
        
        # 确保所有QA对都有file_name字段
        valid_qa_pairs = []
        for qa_pair in all_qa_pairs:
            if qa_pair and "file_name" in qa_pair:
                valid_qa_pairs.append(qa_pair)
            else:
                logger.warning(f"跳过无效的QA对: {qa_pair}")
        
        for qa_pair in valid_qa_pairs:
            if qa_pair["file_name"] in file_qa_pairs:
                file_qa_pairs[qa_pair["file_name"]].append(qa_pair)
        
        # 为每个文件保存结果，并按照block_index排序
        total = 0
        for file_name, qa_pairs in tqdm(file_qa_pairs.items(), desc="保存结果"):
            # 按照block_index排序
            qa_pairs.sort(key=lambda x: x.get('block_index', 0))
            if qa_pairs:
                file_basename = file_name.replace(".md", "")
                output_path = os.path.join(output_dir, f"{file_basename}.json")
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(qa_pairs, ensure_ascii=False, indent=2))
                logger.info(f"已保存 {len(qa_pairs)} 个QA对到 {output_path}")
                total += len(qa_pairs)
        logger.info(f"共计生成{total}个QA对")
        
    def shutdown(self):
        """关闭线程池"""
        self.io_thread_pool.shutdown()

async def async_qa_evaluate_common(qa: dict, ref=None, client: AsyncArk = None, model_name: str = None):
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
    evaluate_model_name = Config.EVALUATE_MODEL_NAME if Config.EVALUATE_MODEL_NAME else None
    input_dirs = Config.INPUT_DIRS  # 替换为输入目录
    output_dir = Config.OUTPUT_DIR  # 替换为输出目录
    output_dir = os.path.join(output_dir, "common")
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
    processor = BatchCommonProcessor(model_name, max_workers, evaluate_model_name)
    
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