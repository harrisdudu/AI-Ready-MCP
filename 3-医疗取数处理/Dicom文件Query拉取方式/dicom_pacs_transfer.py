#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医学DICOM文件与PACS系统数据传输工具

此脚本提供了DICOM文件与PACS系统之间的数据传输功能，包括：
1. 将DICOM文件发送到PACS服务器
2. 从PACS服务器查询和检索DICOM文件
3. 基本的错误处理和日志记录

使用说明：
- 需要安装pydicom和pynetdicom库
- 根据实际情况修改脚本中的PACS服务器配置信息
- 可以通过命令行参数或直接修改代码中的参数来使用
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pydicom import dcmread, read_file
from pydicom.errors import InvalidDicomError
from pynetdicom import AE, debug_logger, StoragePresentationContexts
from pynetdicom.sop_class import (StudyRootQueryRetrieveInformationModelFind,
                                 StudyRootQueryRetrieveInformationModelMove,
                                 StudyRootQueryRetrieveInformationModelGet)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dicom_transfer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DicomPacsTransfer:
    """DICOM文件与PACS系统数据传输类"""
    
    def __init__(self, pacs_ip='127.0.0.1', pacs_port=11112, ae_title='PYNETDICOM', 
                 pacs_ae_title='PACS', timeout=30):
        """
        初始化DicomPacsTransfer类
        
        参数:
            pacs_ip (str): PACS服务器IP地址
            pacs_port (int): PACS服务器端口
            ae_title (str): 本地AE标题
            pacs_ae_title (str): PACS服务器AE标题
            timeout (int): 连接超时时间(秒)
        """
        self.pacs_ip = pacs_ip
        self.pacs_port = pacs_port
        self.ae_title = ae_title
        self.pacs_ae_title = pacs_ae_title
        self.timeout = timeout
        
    def send_dicom_file(self, dicom_file_path):
        """
        将单个DICOM文件发送到PACS服务器
        
        参数:
            dicom_file_path (str): DICOM文件路径
        
        返回:
            bool: 发送成功返回True，否则返回False
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(dicom_file_path):
                logger.error(f"文件不存在: {dicom_file_path}")
                return False
            
            # 读取DICOM文件
            try:
                ds = dcmread(dicom_file_path)
            except InvalidDicomError:
                logger.error(f"无效的DICOM文件: {dicom_file_path}")
                return False
            
            # 创建AE实例
            ae = AE(ae_title=self.ae_title)
            
            # 添加所有存储SOP类的表示上下文
            ae.requested_contexts = StoragePresentationContexts
            
            logger.info(f"正在连接PACS服务器: {self.pacs_ip}:{self.pacs_port}")
            # 连接到PACS服务器
            assoc = ae.associate(self.pacs_ip, self.pacs_port, 
                                 ae_title=self.pacs_ae_title, 
                                 timeout=self.timeout)
            
            if assoc.is_established:
                logger.info(f"连接已建立，正在发送DICOM文件: {dicom_file_path}")
                # 发送DICOM文件
                status = assoc.send_c_store(ds)
                
                # 释放连接
                assoc.release()
                
                if status: 
                    # 如果状态是成功的(0x0000)或警告的(0x0001-0x7FFF)
                    logger.info(f"DICOM文件发送成功: {dicom_file_path}, 状态码: {status.Status}")
                    return True
                else:
                    logger.error(f"DICOM文件发送失败: {dicom_file_path}, 未收到状态信息")
                    return False
            else:
                logger.error(f"无法连接到PACS服务器: {self.pacs_ip}:{self.pacs_port}")
                return False
        
        except Exception as e:
            logger.error(f"发送DICOM文件时发生错误: {str(e)}")
            return False
            
    def send_dicom_directory(self, directory_path):
        """
        发送目录中的所有DICOM文件到PACS服务器
        
        参数:
            directory_path (str): 包含DICOM文件的目录路径
        
        返回:
            dict: 包含成功和失败文件信息的字典
        """
        result = {
            'success': [],
            'failed': []
        }
        
        # 检查目录是否存在
        if not os.path.exists(directory_path):
            logger.error(f"目录不存在: {directory_path}")
            return result
        
        logger.info(f"正在扫描目录: {directory_path}")
        
        # 遍历目录中的所有文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 尝试发送文件
                if self.send_dicom_file(file_path):
                    result['success'].append(file_path)
                else:
                    result['failed'].append(file_path)
        
        logger.info(f"目录扫描完成，成功发送: {len(result['success'])}个文件, 失败: {len(result['failed'])}个文件")
        return result
        
    def query_pacs(self, query_params):
        """
        向PACS服务器查询DICOM文件
        
        参数:
            query_params (dict): 查询参数
        
        返回:
            list: 查询结果列表
        """
        try:
            # 创建AE实例
            ae = AE(ae_title=self.ae_title)
            
            # 添加查询/检索信息模型的表示上下文
            ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
            
            logger.info(f"正在连接PACS服务器进行查询: {self.pacs_ip}:{self.pacs_port}")
            # 连接到PACS服务器
            assoc = ae.associate(self.pacs_ip, self.pacs_port, 
                                 ae_title=self.pacs_ae_title, 
                                 timeout=self.timeout)
            
            results = []
            
            if assoc.is_established:
                # 创建查询数据集
                ds = dcmread(prefix="", is_little_endian=True)
                
                # 设置查询参数
                for key, value in query_params.items():
                    setattr(ds, key, value)
                
                # 执行查询
                responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)
                
                for (status, identifier) in responses:
                    if status and identifier:
                        # 将DICOM数据集转换为字典并添加到结果列表
                        result_dict = {}
                        for elem in identifier:
                            if elem.VR != 'SQ':  # 跳过序列类型
                                result_dict[elem.tag.description()] = str(elem.value)
                        results.append(result_dict)
                
                # 释放连接
                assoc.release()
                
                logger.info(f"查询完成，找到{len(results)}条记录")
            else:
                logger.error(f"无法连接到PACS服务器进行查询: {self.pacs_ip}:{self.pacs_port}")
        
        except Exception as e:
            logger.error(f"查询PACS服务器时发生错误: {str(e)}")
        
        return results
        
    def retrieve_study(self, study_instance_uid, output_dir):
        """
        从PACS服务器检索整个研究的DICOM文件
        
        参数:
            study_instance_uid (str): 研究实例UID
            output_dir (str): 保存检索到的DICOM文件的目录
        
        返回:
            dict: 包含检索结果信息的字典
        """
        result = {
            'success': False,
            'file_count': 0,
            'message': ''
        }
        
        try:
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 创建AE实例
            ae = AE(ae_title=self.ae_title)
            
            # 添加存储和查询/检索信息模型的表示上下文
            ae.requested_contexts = StoragePresentationContexts
            ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)
            
            logger.info(f"正在连接PACS服务器进行检索: {self.pacs_ip}:{self.pacs_port}")
            # 连接到PACS服务器
            assoc = ae.associate(self.pacs_ip, self.pacs_port, 
                                 ae_title=self.pacs_ae_title, 
                                 timeout=self.timeout)
            
            if assoc.is_established:
                # 创建移动请求数据集
                ds = dcmread(prefix="", is_little_endian=True)
                ds.QueryRetrieveLevel = 'STUDY'
                ds.StudyInstanceUID = study_instance_uid
                
                # 定义存储回调函数
                def store_callback(event):
                    """处理接收到的存储事件"""
                    ds = event.dataset
                    ds.file_meta = event.file_meta
                    
                    # 生成文件名（使用SOP实例UID）
                    if hasattr(ds, 'SOPInstanceUID'):
                        filename = f"{ds.SOPInstanceUID}.dcm"
                    else:
                        filename = f"unknown_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.dcm"
                    
                    file_path = os.path.join(output_dir, filename)
                    ds.save_as(file_path)
                    logger.info(f"保存DICOM文件: {file_path}")
                    result['file_count'] += 1
                    
                    # 返回成功状态
                    return 0x0000
                
                # 注册存储回调函数
                ae.on_c_store = store_callback
                
                # 执行移动请求
                status = assoc.send_c_move(ds, self.ae_title, StudyRootQueryRetrieveInformationModelMove)
                
                # 释放连接
                assoc.release()
                
                if status:
                    if status.Status == 0x0000:
                        result['success'] = True
                        result['message'] = f"成功检索到{result['file_count']}个DICOM文件"
                        logger.info(result['message'])
                    else:
                        result['message'] = f"检索失败，状态码: {status.Status}"
                        logger.error(result['message'])
                else:
                    result['message'] = "检索失败，未收到状态信息"
                    logger.error(result['message'])
            else:
                result['message'] = f"无法连接到PACS服务器进行检索: {self.pacs_ip}:{self.pacs_port}"
                logger.error(result['message'])
        
        except Exception as e:
            result['message'] = f"检索DICOM文件时发生错误: {str(e)}"
            logger.error(result['message'])
        
        return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DICOM文件与PACS系统数据传输工具')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 发送单个文件命令
    send_file_parser = subparsers.add_parser('send-file', help='发送单个DICOM文件到PACS服务器')
    send_file_parser.add_argument('--file', required=True, help='DICOM文件路径')
    send_file_parser.add_argument('--ip', default='127.0.0.1', help='PACS服务器IP地址')
    send_file_parser.add_argument('--port', type=int, default=11112, help='PACS服务器端口')
    send_file_parser.add_argument('--ae-title', default='PYNETDICOM', help='本地AE标题')
    send_file_parser.add_argument('--pacs-ae-title', default='PACS', help='PACS服务器AE标题')
    
    # 发送目录命令
    send_dir_parser = subparsers.add_parser('send-dir', help='发送目录中的所有DICOM文件到PACS服务器')
    send_dir_parser.add_argument('--dir', required=True, help='包含DICOM文件的目录路径')
    send_dir_parser.add_argument('--ip', default='127.0.0.1', help='PACS服务器IP地址')
    send_dir_parser.add_argument('--port', type=int, default=11112, help='PACS服务器端口')
    send_dir_parser.add_argument('--ae-title', default='PYNETDICOM', help='本地AE标题')
    send_dir_parser.add_argument('--pacs-ae-title', default='PACS', help='PACS服务器AE标题')
    
    # 查询命令
    query_parser = subparsers.add_parser('query', help='向PACS服务器查询DICOM文件')
    query_parser.add_argument('--patient-id', help='患者ID')
    query_parser.add_argument('--patient-name', help='患者姓名')
    query_parser.add_argument('--study-date', help='研究日期 (格式: YYYYMMDD-YYYYMMDD)')
    query_parser.add_argument('--ip', default='127.0.0.1', help='PACS服务器IP地址')
    query_parser.add_argument('--port', type=int, default=11112, help='PACS服务器端口')
    query_parser.add_argument('--ae-title', default='PYNETDICOM', help='本地AE标题')
    query_parser.add_argument('--pacs-ae-title', default='PACS', help='PACS服务器AE标题')
    
    # 检索命令
    retrieve_parser = subparsers.add_parser('retrieve', help='从PACS服务器检索DICOM文件')
    retrieve_parser.add_argument('--study-uid', required=True, help='研究实例UID')
    retrieve_parser.add_argument('--output-dir', required=True, help='保存检索到的DICOM文件的目录')
    retrieve_parser.add_argument('--ip', default='127.0.0.1', help='PACS服务器IP地址')
    retrieve_parser.add_argument('--port', type=int, default=11112, help='PACS服务器端口')
    retrieve_parser.add_argument('--ae-title', default='PYNETDICOM', help='本地AE标题')
    retrieve_parser.add_argument('--pacs-ae-title', default='PACS', help='PACS服务器AE标题')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    if not args.command:
        print("请指定命令，使用 -h 查看帮助")
        sys.exit(1)
    
    # 创建DicomPacsTransfer实例
    transfer = DicomPacsTransfer(
        pacs_ip=args.ip,
        pacs_port=args.port,
        ae_title=args.ae_title,
        pacs_ae_title=args.pacs_ae_title
    )
    
    # 根据命令执行相应操作
    if args.command == 'send-file':
        success = transfer.send_dicom_file(args.file)
        sys.exit(0 if success else 1)
        
    elif args.command == 'send-dir':
        result = transfer.send_dicom_directory(args.dir)
        print(f"发送完成: 成功{len(result['success'])}个, 失败{len(result['failed'])}个")
        sys.exit(0 if len(result['failed']) == 0 else 1)
        
    elif args.command == 'query':
        # 构建查询参数
        query_params = {
            'QueryRetrieveLevel': 'STUDY',
            'PatientID': args.patient_id,
            'PatientName': args.patient_name,
            'StudyDate': args.study_date
        }
        # 移除None值
        query_params = {k: v for k, v in query_params.items() if v is not None}
        
        results = transfer.query_pacs(query_params)
        print(f"查询结果: {len(results)}条记录")
        for i, result in enumerate(results):
            print(f"\n记录 {i+1}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
    elif args.command == 'retrieve':
        result = transfer.retrieve_study(args.study_uid, args.output_dir)
        print(result['message'])
        sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()