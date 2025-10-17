#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
from tqdm import tqdm
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise SystemExit("❌ 请先安装 sentence-transformers: pip install sentence-transformers")

# ========= 配置常量 =========
DEFAULT_EMBEDDING_MODEL = "/home/wangxi/workspace/gongye/yijizaojia/Qwen3-Embedding-0.6B"  # 本地模型路径
DEFAULT_SIMILARITY_THRESHOLD = 0.99  # 余弦相似度阈值

class EmbeddingDeduplicator:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, 
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        # 支持本地模型路径
        if Path(model_name).exists():
            print(f"🔍 加载本地embedding模型：{model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            print(f"🔍 下载embedding模型：{model_name}")
            self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        
    def deduplicate_batch(self, questions_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        批量去重处理
        返回：(去重后的数据, 重复信息列表)
        """
        # 提取问题文本
        questions = [item.get('question', '') for item in questions_data]
        
        # 过滤空问题
        valid_indices = [i for i, q in enumerate(questions) if q.strip()]
        valid_questions = [questions[i] for i in valid_indices]
        valid_data = [questions_data[i] for i in valid_indices]
        
        if not valid_questions:
            return [], []
        
        print(f"🔄 计算embedding向量...")
        # 批量计算embedding
        embeddings = self.model.encode(valid_questions, normalize_embeddings=True)
        
        print(f"🔄 计算相似度矩阵...")
        # 计算相似度矩阵
        similarity_matrix = embeddings @ embeddings.T
        
        # 设置对角线为0（避免自己与自己比较）
        np.fill_diagonal(similarity_matrix, 0)
        
        print(f"🔄 执行去重...")
        # 去重处理
        unique_indices = []
        duplicate_info = []
        processed = set()
        
        for i in range(len(valid_questions)):
            if i in processed:
                continue
                
            # 找到与当前问题相似的所有问题
            similar_indices = np.where(similarity_matrix[i] >= self.threshold)[0]
            
            if len(similar_indices) > 0:
                # 有重复，记录重复信息
                current_question = valid_questions[i]
                current_data = valid_data[i]
                
                for similar_idx in similar_indices:
                    if similar_idx not in processed:
                        duplicate_question = valid_questions[similar_idx]
                        duplicate_data = valid_data[similar_idx]
                        similarity = float(similarity_matrix[i][similar_idx])
                        
                        duplicate_info.append({
                            'current_question': current_question,
                            'current_data': current_data,
                            'duplicate_question': duplicate_question,
                            'duplicate_data': duplicate_data,
                            'similarity': similarity,
                            'current_index': valid_indices[i],
                            'duplicate_index': valid_indices[similar_idx]
                        })
                        
                        processed.add(similar_idx)
                
                # 将当前问题加入唯一列表
                unique_indices.append(i)
                processed.add(i)
            else:
                # 无重复，直接加入唯一列表
                unique_indices.append(i)
                processed.add(i)
        
        # 返回去重后的数据
        unique_data = [valid_data[i] for i in unique_indices]
        
        return unique_data, duplicate_info

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='对JSONL文件中的问题进行去重')
    parser.add_argument('--input', type=str, 
                       default='/home/wangxi/workspace/gongye/zejun/out_qas_t/combined_questions.jsonl',
                       help='输入JSONL文件路径')
    parser.add_argument('--output', type=str, 
                       default='/home/wangxi/workspace/gongye/zejun/out_qas_t/deduplicated_questions.jsonl',
                       help='输出JSONL文件路径')
    parser.add_argument('--stats', type=str, 
                       default='/home/wangxi/workspace/gongye/zejun/out_qas_t/deduplication_stats.json',
                       help='统计信息输出文件路径')
    parser.add_argument('--model', type=str, 
                       default=DEFAULT_EMBEDDING_MODEL,
                       help='Embedding模型路径或名称')
    parser.add_argument('--threshold', type=float, 
                       default=DEFAULT_SIMILARITY_THRESHOLD,
                       help='相似度阈值')
    
    args = parser.parse_args()
    
    print(f"📖 加载输入文件：{args.input}")
    input_data = load_jsonl(args.input)
    print(f"📊 总题目数量：{len(input_data)}")
    
    # 初始化去重器
    deduplicator = EmbeddingDeduplicator(
        model_name=args.model,
        similarity_threshold=args.threshold
    )
    
    # 批量去重处理
    start_time = time.time()
    unique_questions, duplicate_info = deduplicator.deduplicate_batch(input_data)
    end_time = time.time()
    
    # 保存去重后的数据
    print(f"💾 保存去重后的数据：{args.output}")
    save_jsonl(unique_questions, args.output)
    
    # 生成统计信息
    stats = {
        'input_total': len(input_data),
        'output_unique': len(unique_questions),
        'duplicates_found': len(duplicate_info),
        'deduplication_rate': len(duplicate_info) / len(input_data) * 100,
        'similarity_threshold': args.threshold,
        'embedding_model': args.model,
        'processing_time': end_time - start_time,
        'duplicate_details': duplicate_info
    }
    
    # 保存统计信息
    print(f"📈 保存统计信息：{args.stats}")
    with open(args.stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("📊 去重统计摘要")
    print("="*50)
    print(f"输入题目总数：{stats['input_total']}")
    print(f"去重后题目数：{stats['output_unique']}")
    print(f"发现重复题目：{stats['duplicates_found']}")
    print(f"去重率：{stats['deduplication_rate']:.2f}%")
    print(f"相似度阈值：{stats['similarity_threshold']}")
    print(f"Embedding模型：{stats['embedding_model']}")
    print(f"处理时间：{stats['processing_time']:.2f}秒")
    
    # 相似度分布统计
    if duplicate_info:
        similarities = [info['similarity'] for info in duplicate_info]
        print(f"\n相似度分布：")
        print(f"  最高相似度：{max(similarities):.4f}")
        print(f"  最低相似度：{min(similarities):.4f}")
        print(f"  平均相似度：{np.mean(similarities):.4f}")
        print(f"  中位数相似度：{np.median(similarities):.4f}")
    
    print(f"\n✅ 去重完成！")
    print(f"  去重后数据：{args.output}")
    print(f"  统计信息：{args.stats}")

if __name__ == "__main__":
    main()
