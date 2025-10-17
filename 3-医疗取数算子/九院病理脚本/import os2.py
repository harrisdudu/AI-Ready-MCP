#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查记录查询与导出为JSON脚本

功能：
1. 连接数据库，读取 CUSTUME.JCBGD 表的检查记录数据
2. 将每条记录导出为单独的JSON文件
3. 支持自定义输出目录

作者：AI Assistant
日期：2024
"""

import os
import sys
import json
import logging
# 注意：本脚本需要cx_Oracle库连接Oracle数据库
# 如未安装，请使用命令：pip install cx_Oracle
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# === 配置日志系统 ===
def setup_logging(log_file='export_to_json.log'):
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# === Oracle数据库配置 ===
DB_CONFIG = {
    'host': '172.28.10.1',  # Oracle服务器主机名或IP
    'port': '1521',          # Oracle监听端口，默认为1521
    'sid': 'orcl',          # Oracle SID或服务名
    'username': 'custume',  # Oracle用户名
    'password': 'Kps@123456!'  # Oracle密码
}

def get_db_connection():
    """获取Oracle数据库连接"""
    try:
        # 尝试导入cx_Oracle库
        try:
            import cx_Oracle
        except ImportError:
            logger.error("❌ 未找到cx_Oracle库，请安装: pip install cx_Oracle")
            return None
        
        # 构建Oracle连接字符串
        dsn = cx_Oracle.makedsn(
            DB_CONFIG['host'], 
            DB_CONFIG['port'], 
            sid=DB_CONFIG['sid']
        )
        
        # 连接数据库
        conn = cx_Oracle.connect(
            user=DB_CONFIG['username'],
            password=DB_CONFIG['password'],
            dsn=dsn
        )
        
        logger.info(f"✅ 成功连接Oracle数据库: {DB_CONFIG['sid']}")
        return conn
    except Exception as e:
        logger.error(f"❌ 连接Oracle数据库失败: {e}")
        return None




class DataExporter:
    """数据导出器"""
    
    def __init__(self, output_dir: str = 'json_exports'):
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.dump_count = 0
        self.package_size = 300
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"📁 输出目录: {self.output_dir}")
    
    def datetime_to_str(self, obj):
        """将日期时间对象转换为字符串，兼容Oracle数据库类型"""
        # 处理datetime对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        # 处理cx_Oracle的DATE和TIMESTAMP类型
        try:
            # 尝试将对象转换为字符串（适用于cx_Oracle的日期类型）
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, 'strftime'):
                return obj.strftime('%Y-%m-%dT%H:%M:%S')
        except:
            pass
        # 如果以上都不适用，返回原始值
        return obj
    
    def clean_filename(self, filename: str) -> str:
        """清理文件名，移除非法字符"""
        # 移除或替换文件名中的非法字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def get_all_records(self) -> List[Dict]:
        """从数据库获取所有检查记录"""
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("❌ 无法连接数据库")
                return []
            
            try:
                cursor = conn.cursor()
                
                # 使用图片查询SQL
                query = """
                SELECT 
                    住院号 AS 流水号,
                    报告日期 AS 审核时间,
                    检查号 AS 检查单号,
                    分类,
                    报告名称 AS 检查名称,
                    诊断描述 AS 检查所见,
                    结果与描述 AS 检查结果,
                    检查部位
                FROM 
                    CUSTUME.JCBGD
                WHERE 
                    (分类 = 'NJ' 
                     OR 分类 = 'XHBL'
                     OR 分类 = 'BL' AND (诊断描述 LIKE '%肠%' OR 诊断描述 LIKE '%胃%'))
                    AND 住院号 IN (
                        SELECT 住院号
                        FROM CUSTUME.中间临时表2)
                """
                
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                
                records = []
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    # 转换datetime对象为字符串
                    for key, value in record.items():
                        record[key] = self.datetime_to_str(value)
                    records.append(record)
                
                cursor.close()
                logger.info(f"✅ 成功获取 {len(records)} 条检查记录")
                return records
                
            finally:
                conn.close()
                logger.info("🔌 数据库连接已关闭")
                
        except Exception as e:
            logger.error(f"❌ 获取检查记录失败: {e}")
            return []
    


    def export_single_records(self, records: List[Dict]):
        """导出记录为单独的JSON文件"""
        logger.info("📤 开始导出记录为单独的JSON文件")
        
        exported_count = 0
        for i, record in enumerate(records, 1):
            try:
                # 生成文件名
                record_id = f"record_{i}"
                filename = f"{record_id}.json"

                # 构建输出数据结构
                out_data = {
                    '流水号': record.get('流水号', ''),
                    '审核时间': record.get('审核时间', ''),
                    '检查单号': record.get('检查单号', ''),
                    '分类': record.get('分类', ''),
                    '检查名称': record.get('检查名称', ''),
                    '检查所见': record.get('检查所见', ''),
                    '检查结果': record.get('检查结果', ''),
                    '检查部位': record.get('检查部位', ''),
                }

                out = {'data': out_data}
                self.dump_count += 1
                package_number = self.dump_count // self.package_size
                if self.dump_count % self.package_size:
                    package_number += 1

                file_path = os.path.join(self.output_dir, str(package_number), filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # 写入JSON文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=4)
                
                exported_count += 1
                if exported_count % 100 == 0:
                    logger.info(f"📄 已导出 {exported_count} 条记录...")
                    
            except Exception as e:
                logger.error(f"❌ 导出记录 {i} 失败: {e}")
                continue
        
        logger.info(f"✅ 记录导出完成，共导出 {exported_count} 个文件")
    
    def run(self, export_mode: str = 'single'):
        """运行数据导出流程"""
        logger.info("🚀 开始数据导出流程")
        
        # 获取数据
        records = self.get_all_records()
        if not records:
            logger.error("❌ 没有数据可导出")
            return
        
        # 导出记录
        self.export_single_records(records)
        
        logger.info("🎉 数据导出完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图片数据查询与导出为JSON工具")
    parser.add_argument("--output-dir", default="json_exports", help="输出目录路径（默认: json_exports）")
    parser.add_argument("--log-file", default="export_to_json.log", help="日志文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging(args.log_file)
    
    # 创建导出器并运行
    exporter = DataExporter(args.output_dir)
    
    logger.info(f"📁 输出目录: {args.output_dir}")
    
    exporter.run()


if __name__ == "__main__":
    main()
