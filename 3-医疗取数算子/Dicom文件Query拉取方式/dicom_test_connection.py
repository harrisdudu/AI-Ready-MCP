#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM与PACS连接测试工具

此脚本用于测试DICOM库安装和PACS服务器连接设置是否正确，包括：
1. 检查必要的DICOM库是否已安装
2. 测试与PACS服务器的连接
3. 生成一个简单的DICOM测试文件（可选）

使用说明：
- 运行脚本测试DICOM库安装情况
- 使用--connect参数测试与PACS服务器的连接
- 使用--generate参数生成测试用的DICOM文件
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_library_installation():
    """
    检查必要的DICOM库是否已安装
    
    返回:
        bool: 所有库都已安装返回True，否则返回False
    """
    required_libraries = {
        'pydicom': '用于读取和写入DICOM文件',
        'pynetdicom': '用于DICOM网络通信'
    }
    
    all_installed = True
    
    logger.info("=== 检查DICOM库安装情况 ===")
    
    for lib_name, description in required_libraries.items():
        try:
            __import__(lib_name)
            logger.info(f"✓ {lib_name} 已安装 ({description})")
        except ImportError:
            logger.error(f"✗ {lib_name} 未安装 ({description})")
            all_installed = False
    
    if not all_installed:
        logger.info("请安装缺失的库: pip install pydicom pynetdicom")
    
    return all_installed


def test_pacs_connection(pacs_ip='127.0.0.1', pacs_port=11112, ae_title='PYNETDICOM', 
                        pacs_ae_title='PACS', timeout=30):
    """
    测试与PACS服务器的连接
    
    参数:
        pacs_ip (str): PACS服务器IP地址
        pacs_port (int): PACS服务器端口
        ae_title (str): 本地AE标题
        pacs_ae_title (str): PACS服务器AE标题
        timeout (int): 连接超时时间(秒)
    
    返回:
        bool: 连接成功返回True，否则返回False
    """
    try:
        # 导入pynetdicom库
        from pynetdicom import AE
        
        logger.info(f"\n=== 测试与PACS服务器的连接 ===")
        logger.info(f"PACS服务器: {pacs_ip}:{pacs_port}")
        logger.info(f"本地AE标题: {ae_title}")
        logger.info(f"PACS AE标题: {pacs_ae_title}")
        logger.info(f"超时时间: {timeout}秒")
        
        # 创建AE实例
        ae = AE(ae_title=ae_title)
        
        # 添加一个基本的表示上下文用于测试连接
        ae.add_requested_context('1.2.840.10008.1.1')  # Verification SOP Class
        
        # 尝试连接到PACS服务器
        logger.info("正在尝试连接到PACS服务器...")
        assoc = ae.associate(pacs_ip, pacs_port, 
                             ae_title=pacs_ae_title, 
                             timeout=timeout)
        
        if assoc.is_established:
            logger.info("✓ 成功连接到PACS服务器！")
            
            # 执行验证ECHO操作
            from pynetdicom.sop_class import Verification
            status = assoc.send_c_echo(Verification)
            
            if status:
                logger.info(f"✓ 验证ECHO操作成功，状态码: {status.Status}")
            else:
                logger.warning("✗ 验证ECHO操作未收到状态信息")
            
            # 释放连接
            assoc.release()
            return True
        else:
            logger.error("✗ 无法连接到PACS服务器！")
            logger.error(f"连接参数可能不正确，或PACS服务器未运行/不接受连接")
            return False
    
    except ImportError:
        logger.error("✗ 无法导入pynetdicom库，请先安装该库")
        return False
    except Exception as e:
        logger.error(f"✗ 连接测试过程中发生错误: {str(e)}")
        return False


def generate_test_dicom_file(output_path='./test_dicom.dcm'):
    """
    生成一个简单的DICOM测试文件
    
    参数:
        output_path (str): 输出文件路径
        
    返回:
        bool: 生成成功返回True，否则返回False
    """
    try:
        # 导入pydicom库
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        
        logger.info(f"\n=== 生成DICOM测试文件 ===")
        logger.info(f"输出路径: {output_path}")
        
        # 创建DICOM文件元信息
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # 创建DICOM数据集
        ds = FileDataset(output_path, {},
                        file_meta=file_meta,
                        preamble=b"""\0""" * 128)
        
        # 添加基本的DICOM标签
        ds.PatientName = "TEST^PATIENT"
        ds.PatientID = "TEST123"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.StudyDate = datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.now().strftime('%H%M%S')
        ds.ContentDate = ds.StudyDate
        ds.ContentTime = ds.StudyTime
        ds.PatientBirthDate = "19000101"
        ds.PatientSex = "O"
        ds.StudyDescription = "Test Study"
        ds.SeriesDescription = "Test Series"
        ds.InstanceNumber = 1
        
        # 添加图像相关信息
        ds.Rows = 128
        ds.Columns = 128
        ds.PixelSpacing = [1.0, 1.0]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        
        # 创建一个简单的测试图像数据（128x128的渐变图像）
        import numpy as np
        img = np.zeros((128, 128), dtype=np.uint16)
        for i in range(128):
            for j in range(128):
                img[i, j] = int((i + j) * 10)
        
        ds.PixelData = img.tobytes()
        
        # 保存DICOM文件
        ds.save_as(output_path, write_like_original=False)
        
        logger.info(f"✓ 成功生成DICOM测试文件: {os.path.abspath(output_path)}")
        logger.info("此测试文件可用于测试DICOM文件发送功能")
        
        return True
    
    except ImportError:
        logger.error("✗ 无法导入必要的库，请先安装pydicom和numpy")
        return False
    except Exception as e:
        logger.error(f"✗ 生成DICOM文件过程中发生错误: {str(e)}")
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DICOM与PACS连接测试工具')
    
    parser.add_argument('--connect', action='store_true', help='测试与PACS服务器的连接')
    parser.add_argument('--ip', default='127.0.0.1', help='PACS服务器IP地址')
    parser.add_argument('--port', type=int, default=11112, help='PACS服务器端口')
    parser.add_argument('--ae-title', default='PYNETDICOM', help='本地AE标题')
    parser.add_argument('--pacs-ae-title', default='PACS', help='PACS服务器AE标题')
    parser.add_argument('--generate', action='store_true', help='生成一个DICOM测试文件')
    parser.add_argument('--output', default='./test_dicom.dcm', help='生成的DICOM测试文件路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 首先检查库安装情况
    libraries_installed = check_library_installation()
    
    # 如果指定了连接测试
    if args.connect:
        if libraries_installed:
            test_pacs_connection(
                pacs_ip=args.ip,
                pacs_port=args.port,
                ae_title=args.ae_title,
                pacs_ae_title=args.pacs_ae_title
            )
    
    # 如果指定了生成测试文件
    if args.generate:
        generate_test_dicom_file(args.output)
    
    # 如果没有指定任何操作，显示帮助信息
    if not args.connect and not args.generate:
        print("\n请指定操作，使用 -h 查看帮助")
        print("例如：")
        print("  python dicom_test_connection.py --connect --ip 192.168.1.100 --port 11112")
        print("  python dicom_test_connection.py --generate --output ./my_test.dcm")


if __name__ == '__main__':
    main()