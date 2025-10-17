from typing import List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Optional, Callable
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union

class BGEEmbedding:
    def __init__(self, model_name="bge-m3"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts: list, batch_size: int = 32):
        """批量编码，默认用GPU（若存在）"""
        # 1. 过滤空文本（避免报错）
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return []
        
        # 2. 移动模型到GPU（若有）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)  # 仅初始化时移动1次即可，避免重复移动
        
        embeddings = []
        # 3. 按batch_size分割文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # 批量tokenize（padding=True会自动处理不同长度文本）
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=8196  # BGE-M3建议max_length=512/1024，避免过长
            ).to(device)  # 输入也移动到GPU
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 批量计算mean pooling（无需循环squeeze）
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 回传CPU存numpy
            embeddings.extend(batch_embeddings.tolist())  # 避免numpy数组频繁拼接
        
        return embeddings

    def embed_query(self, text: str):
        """查询编码也适配GPU"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=8196
        ).to(device)
    
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    

class SimpleTextRecall:
    def __init__(self, embedding_model: BGEEmbedding):
        self.embedding_model = embedding_model
        self.texts = []
        self.embeddings = None

    def add_texts(self, texts: List[str]):
        """添加文本到召回系统"""
        new_texts = []
        for text in texts:
            if text in self.texts:
                continue
            else:
                new_texts.append(text)
        if not new_texts:
            return 
        new_embeddings = self.embedding_model.embed_documents(new_texts)
        self.texts.extend(new_texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def delete_texts(self):
        self.embeddings = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索与查询最相似的文本"""
        if not self.texts:
            return []
            
        query_embedding = self.embedding_model.embed_query(query)
        
        # 计算余弦相似度
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # 获取top_k的索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        return results

    def _cosine_similarity(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # 归一化查询向量
        query_norm = query / np.linalg.norm(query)
        # 归一化文档向量
        documents_norm = documents / np.linalg.norm(documents, axis=1, keepdims=True)
        # 计算余弦相似度
        return np.dot(query_norm, documents_norm.T)
    
    def deduped(self, texts: List[str], threshold: float = 0.9, max_results: int = None):
        """优化后的语义去重方法"""
        if not texts:
            return []
        
        # 1. 精确去重
        unique_texts = list(dict.fromkeys(texts))  # 保持顺序的去重
        
        # 2. 批量计算嵌入并归一化
        new_embeddings = np.array(self.embedding_model.embed_documents(unique_texts))
        new_embeddings_norm = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # 3. 优化的去重逻辑
        deduped_indices = []  # 只存索引，节省内存
        deduped_embeddings_list = []  # 临时存储用于批量计算
        
        for i, emb_norm in enumerate(new_embeddings_norm):
            if max_results and len(deduped_indices) >= max_results:
                break
                
            if not deduped_indices:
                deduped_indices.append(i)
                deduped_embeddings_list.append(emb_norm)
            else:
                # 批量计算相似度
                deduped_matrix = np.array(deduped_embeddings_list)
                similarities = np.dot(deduped_matrix, emb_norm)
                
                if np.all(similarities < threshold):  # 所有都不相似
                    deduped_indices.append(i)
                    deduped_embeddings_list.append(emb_norm)
        
        return [unique_texts[i] for i in deduped_indices]

    
class HierarchicalTextRecall:
    def __init__(self, embedding_model: BGEEmbedding):
        self.embedding_model = embedding_model
        # 全局召回器（用于跨文件的相关性搜索）
        self.global_recall = SimpleTextRecall(embedding_model)
        # 文件级别的召回器字典 {file_path: SimpleTextRecall}
        self.file_recallers = {}
        # 文档级别的召回器字典 {doc_id: SimpleTextRecall}
        self.doc_recallers = {}
        # 保存原始Embedding方法（用于后续恢复）
        self._original_embed_documents = embedding_model.embed_documents
        self._original_embed_query = embedding_model.embed_query

    def add_file_texts(self, file_path: str, chunks: List[Dict[str, Any]]):
        if file_path not in self.file_recallers:
            self.file_recallers[file_path] = SimpleTextRecall(self.embedding_model)
        
        file_recaller = self.file_recallers[file_path]
        all_references = []

        # --------------------------
        # 步骤1：收集当前文件所有参考文本，批量编码（仅1次）
        # --------------------------
        # 1.1 汇总所有chunk的references
        for chunk in chunks:
            references = chunk['references']
            all_references.extend(references)
        
        # 1.2 精确去重（避免重复编码相同文本）
        unique_refs = list(dict.fromkeys(all_references))  # 保持顺序的去重
        if not unique_refs:
            return
        
        # 1.3 批量编码（用原始方法，仅1次）
        unique_ref_embeddings = self._original_embed_documents(unique_refs)
        
        # 1.4 构建“文本→Embedding”缓存字典
        text_to_emb = dict(zip(unique_refs, unique_ref_embeddings))

        # --------------------------
        # 步骤2：临时替换BGEEmbedding的编码方法，优先从缓存取结果
        # --------------------------
        def cached_embed_documents(texts: list):
            """缓存版编码：优先从text_to_emb取，无则用原始方法（兜底）"""
            return [text_to_emb.get(text, self._original_embed_documents([text])[0]) for text in texts]
        
        def cached_embed_query(text: str):
            """缓存版查询编码：优先从text_to_emb取，无则用原始方法（兜底）"""
            return text_to_emb.get(text, self._original_embed_query(text))
        
        # 替换当前embedding_model的方法
        self.embedding_model.embed_documents = cached_embed_documents
        self.embedding_model.embed_query = cached_embed_query

        try:
            # --------------------------
            # 步骤3：向各召回器添加文本（此时编码会走缓存，无重复计算）
            # --------------------------
            # 3.1 文档级召回器：添加当前chunk的references
            for chunk in chunks:
                references = chunk['references']
                doc_id = f"{file_path}_{hash(chunk['content'])}"
                
                if doc_id not in self.doc_recallers:
                    self.doc_recallers[doc_id] = SimpleTextRecall(self.embedding_model)
                
                # 调用add_texts，但编码走缓存（无重复计算）
                self.doc_recallers[doc_id].add_texts(references)

            # 3.2 文件级召回器：添加当前文件的所有unique_refs
            file_recaller.add_texts(unique_refs)

            self.global_recall.add_texts(unique_refs)

        finally:
            # --------------------------
            # 步骤4：恢复原始编码方法，避免影响其他流程
            # --------------------------
            self.embedding_model.embed_documents = self._original_embed_documents
            self.embedding_model.embed_query = self._original_embed_query
    
    
    async def hierarchical_search(self, question: str, source_content: str, 
                                file_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """分层搜索：先局部后全局"""
        results = []
        
        # 1. 首先在源文档内搜索（最高优先级）
        source_doc_id = f"{file_path}_{hash(source_content)}"
        if source_doc_id in self.doc_recallers:
            doc_results = self.doc_recallers[source_doc_id].search(question, top_k=top_k)
            results.extend([{**r, 'level': 'document'} for r in doc_results])
        
        # 如果文档内结果不足，在文件内搜索
        if len(results) < top_k and file_path in self.file_recallers:
            file_results = self.file_recallers[file_path].search(question, top_k=top_k)
            # 过滤掉已经包含的结果
            existing_texts = {r['text'] for r in results}
            for r in file_results:
                if r['text'] not in existing_texts:
                    results.append({**r, 'level': 'file'})
                    existing_texts.add(r['text'])
        
        # 如果仍然不足，在全局搜索（最低优先级）
        if len(results) < top_k:
            global_results = self.global_recall.search(question, top_k=top_k*2)
            existing_texts = {r['text'] for r in results}
            for r in global_results:
                if r['text'] not in existing_texts and len(results) < top_k:
                    results.append({**r, 'level': 'global'})
        
        # 按相似度排序并返回top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    


if __name__ == '__main__':
    text = "Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our RAG tutorials."
    embedding = BGEEmbedding("/mnt/afs/models/baoy/bge-m3/")
    e = embedding.embed_documents([text])
    print(len(e[0]))
