#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM文件元数据读取工具

此脚本提供了读取DICOM文件元数据的功能，包括：
1. 读取单个DICOM文件的元数据信息
2. 批量读取文件夹中所有DICOM文件的元数据
3. 支持将元数据导出为JSON或CSV格式
4. 支持过滤显示特定的元数据字段
5. 提供基本的数据统计功能

使用说明：
- 需要安装pydicom库
- 可以通过命令行参数或直接调用函数来使用
"""

import os
import sys
import json
import csv
import logging
import argparse
from datetime import datetime
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DicomMetadataReader:
    """DICOM文件元数据读取类"""
    
    def __init__(self):
        """初始化DicomMetadataReader类"""
        # 常用的DICOM元数据标签，用于过滤输出
        # 按类别分组的关键DICOM元数据标签，便于结构化提取和展示
        self.common_tags = {
            # 患者相关信息
            'Patient': [
                'PatientName',         # 患者姓名
                'PatientID',           # 患者唯一标识符
                'PatientBirthDate',    # 患者出生日期 (YYYYMMDD格式)
                'PatientSex'           # 患者性别 (M/F/O未知)
            ],
            # 检查相关信息
            'Study': [
                'StudyInstanceUID',    # 检查实例唯一标识符
                'StudyDate',           # 检查日期 (YYYYMMDD格式)
                'StudyTime',           # 检查时间 (HHMMSS.FFFFFF格式)
                'StudyDescription',    # 检查描述
                'AccessionNumber'      # 检查申请编号
            ],
            # 序列相关信息
            'Series': [
                'SeriesInstanceUID',   # 序列实例唯一标识符
                'SeriesNumber',        # 序列编号
                'SeriesDescription',   # 序列描述
                'Modality',            # 成像模态 (如CT, MR, XA等)
                'BodyPartExamined'     # 检查的身体部位
            ],
            # 实例相关信息
            'Instance': [
                'SOPInstanceUID',      # 服务对象对实例唯一标识符
                'InstanceNumber',      # 实例编号
                'ContentDate',         # 内容创建日期
                'ContentTime'          # 内容创建时间
            ],
            # 图像相关技术参数
            'Image': [
                'Rows',                # 图像行数
                'Columns',             # 图像列数
                'BitsAllocated',       # 每个像素分配的位数
                'BitsStored',          # 每个像素存储的有效位数
                'PixelRepresentation'  # 像素表示方式 (0=无符号整数, 1=有符号整数)
            ]
        }
    
    def read_dicom_metadata(self, dicom_file_path, filter_tags=None):
        """
        读取单个DICOM文件的元数据
        
        参数:
            dicom_file_path (str): DICOM文件路径
            filter_tags (list, optional): 要过滤显示的标签列表
        
        返回:
            dict: 包含DICOM元数据的字典，出错时返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(dicom_file_path):
                logger.error(f"文件不存在: {dicom_file_path}")
                return None
            
            # 读取DICOM文件
            try:
                ds = dcmread(dicom_file_path, stop_before_pixels=True)  # 只读取元数据，不读取像素数据
            except InvalidDicomError:
                logger.error(f"无效的DICOM文件: {dicom_file_path}")
                return None
            
            # 提取元数据
            metadata = {}
            metadata['filename'] = os.path.basename(dicom_file_path)
            metadata['filepath'] = os.path.abspath(dicom_file_path)
            metadata['file_size'] = os.path.getsize(dicom_file_path)
            metadata['read_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 组织元数据为结构化格式
            metadata['patient_info'] = {}
            metadata['study_info'] = {}
            metadata['series_info'] = {}
            metadata['instance_info'] = {}
            metadata['image_info'] = {}
            metadata['all_attributes'] = {}
            
            # 获取所有属性
            for elem in ds:
                # 跳过像素数据和一些大的二进制数据
                if elem.VR == 'OB' or elem.VR == 'OW' or elem.name == 'Pixel Data':
                    continue
                
                # 存储所有属性
                try:
                    value = str(elem.value) if elem.value is not None else ''
                    metadata['all_attributes'][elem.name] = value
                    metadata['all_attributes'][str(elem.tag)] = value
                except Exception as e:
                    logger.warning(f"无法获取属性 {elem.name} 的值: {str(e)}")
                
                # 根据标签类型分类存储
                if elem.name in self.common_tags['Patient']:
                    metadata['patient_info'][elem.name] = value
                elif elem.name in self.common_tags['Study']:
                    metadata['study_info'][elem.name] = value
                elif elem.name in self.common_tags['Series']:
                    metadata['series_info'][elem.name] = value
                elif elem.name in self.common_tags['Instance']:
                    metadata['instance_info'][elem.name] = value
                elif elem.name in self.common_tags['Image']:
                    metadata['image_info'][elem.name] = value
            
            # 如果指定了过滤标签，则只返回这些标签
            if filter_tags:
                filtered_metadata = {'filename': metadata['filename']}
                for tag in filter_tags:
                    if tag in metadata['all_attributes']:
                        filtered_metadata[tag] = metadata['all_attributes'][tag]
                return filtered_metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"读取DICOM文件 {dicom_file_path} 时出错: {str(e)}")
            return None
    
    def read_folder_metadata(self, folder_path, recursive=False, filter_tags=None):
        """
        批量读取文件夹中所有DICOM文件的元数据
        
        参数:
            folder_path (str): 文件夹路径
            recursive (bool, optional): 是否递归处理子文件夹
            filter_tags (list, optional): 要过滤显示的标签列表
        
        返回:
            list: 包含所有DICOM文件元数据的列表
        """
        try:
            # 检查文件夹是否存在
            if not os.path.isdir(folder_path):
                logger.error(f"文件夹不存在: {folder_path}")
                return []
            
            metadata_list = []
            total_files = 0
            processed_files = 0
            invalid_files = 0
            
            logger.info(f"开始扫描文件夹: {folder_path}")
            
            # 遍历文件夹中的文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    total_files += 1
                    file_path = os.path.join(root, file)
                    
                    # 尝试读取DICOM文件元数据
                    metadata = self.read_dicom_metadata(file_path, filter_tags)
                    if metadata:
                        metadata_list.append(metadata)
                        processed_files += 1
                        if processed_files % 100 == 0:
                            logger.info(f"已处理 {processed_files} 个DICOM文件...")
                    else:
                        invalid_files += 1
                
                # 如果不递归，则只处理当前文件夹
                if not recursive:
                    break
            
            logger.info(f"扫描完成!")
            logger.info(f"总文件数: {total_files}")
            logger.info(f"成功处理的DICOM文件数: {processed_files}")
            logger.info(f"无效文件数: {invalid_files}")
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"批量处理文件夹时出错: {str(e)}")
            return []
    
    def export_to_json(self, metadata_list, output_file):
        """
        将元数据导出为JSON格式
        
        参数:
            metadata_list (list): DICOM元数据列表
            output_file (str): 输出JSON文件路径
        
        返回:
            bool: 导出成功返回True，否则返回False
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 导出为JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"元数据已成功导出到JSON文件: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出JSON文件时出错: {str(e)}")
            return False
    
    def export_to_csv(self, metadata_list, output_file):
        """
        将元数据导出为CSV格式
        
        参数:
            metadata_list (list): DICOM元数据列表
            output_file (str): 输出CSV文件路径
        
        返回:
            bool: 导出成功返回True，否则返回False
        """
        try:
            if not metadata_list:
                logger.warning("没有数据可导出到CSV")
                return False
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 提取所有可能的字段
            all_fields = set()
            for metadata in metadata_list:
                all_fields.update(metadata.keys())
            
            # 对于嵌套的字典，展开字段
            flattened_fields = []
            nested_mappings = {}
            
            for field in all_fields:
                if isinstance(metadata_list[0].get(field), dict):
                    nested_fields = metadata_list[0][field].keys()
                    for nested_field in nested_fields:
                        flattened_name = f"{field}.{nested_field}"
                        flattened_fields.append(flattened_name)
                        nested_mappings[flattened_name] = (field, nested_field)
                else:
                    flattened_fields.append(field)
            
            # 按照一定顺序排序字段
            flattened_fields.sort()
            
            # 写入CSV文件
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_fields)
                writer.writeheader()
                
                for metadata in metadata_list:
                    row = {}
                    for field in flattened_fields:
                        if field in nested_mappings:
                            parent_field, child_field = nested_mappings[field]
                            if parent_field in metadata and child_field in metadata[parent_field]:
                                row[field] = metadata[parent_field][child_field]
                            else:
                                row[field] = ''
                        else:
                            row[field] = metadata.get(field, '')
                    writer.writerow(row)
            
            logger.info(f"元数据已成功导出到CSV文件: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出CSV文件时出错: {str(e)}")
            return False
    
    def get_metadata_statistics(self, metadata_list):
        """
        获取元数据统计信息
        
        参数:
            metadata_list (list): DICOM元数据列表
        
        返回:
            dict: 包含统计信息的字典
        """
        try:
            if not metadata_list:
                return {}
            
            stats = {
                'total_files': len(metadata_list),
                'modalities': {},
                'patient_count': 0,
                'study_count': 0,
                'series_count': 0,
                'instance_counts_by_modality': {},
                'date_range': {
                    'earliest': None,
                    'latest': None
                }
            }
            
            # 用于去重
            patients = set()
            studies = set()
            series = set()
            
            for metadata in metadata_list:
                # 统计患者数量
                patient_id = metadata['patient_info'].get('PatientID', '')
                if patient_id:
                    patients.add(patient_id)
                
                # 统计研究数量
                study_uid = metadata['study_info'].get('StudyInstanceUID', '')
                if study_uid:
                    studies.add(study_uid)
                
                # 统计序列数量
                series_uid = metadata['series_info'].get('SeriesInstanceUID', '')
                if series_uid:
                    series.add(series_uid)
                
                # 统计模态分布
                modality = metadata['series_info'].get('Modality', 'Unknown')
                stats['modalities'][modality] = stats['modalities'].get(modality, 0) + 1
                
                # 统计每个模态的实例数量
                if modality not in stats['instance_counts_by_modality']:
                    stats['instance_counts_by_modality'][modality] = 0
                stats['instance_counts_by_modality'][modality] += 1
                
                # 计算日期范围
                study_date = metadata['study_info'].get('StudyDate', '')
                if study_date:
                    try:
                        date_obj = datetime.strptime(study_date, '%Y%m%d')
                        if not stats['date_range']['earliest'] or date_obj < stats['date_range']['earliest']:
                            stats['date_range']['earliest'] = date_obj
                        if not stats['date_range']['latest'] or date_obj > stats['date_range']['latest']:
                            stats['date_range']['latest'] = date_obj
                    except ValueError:
                        pass
            
            # 更新计数
            stats['patient_count'] = len(patients)
            stats['study_count'] = len(studies)
            stats['series_count'] = len(series)
            
            # 格式化日期
            if stats['date_range']['earliest']:
                stats['date_range']['earliest'] = stats['date_range']['earliest'].strftime('%Y-%m-%d')
            if stats['date_range']['latest']:
                stats['date_range']['latest'] = stats['date_range']['latest'].strftime('%Y-%m-%d')
            
            return stats
            
        except Exception as e:
            logger.error(f"计算元数据统计信息时出错: {str(e)}")
            return {}


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='DICOM文件元数据读取工具')
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', help='单个DICOM文件路径')
    input_group.add_argument('--folder', help='DICOM文件夹路径')
    
    # 处理参数
    parser.add_argument('--recursive', action='store_true', help='递归处理子文件夹')
    parser.add_argument('--tags', nargs='+', help='要过滤显示的标签列表')
    
    # 输出参数
    parser.add_argument('--output-json', help='输出JSON文件路径')
    parser.add_argument('--output-csv', help='输出CSV文件路径')
    parser.add_argument('--show-stats', action='store_true', help='显示元数据统计信息')
    
    args = parser.parse_args()
    
    # 创建DicomMetadataReader实例
    reader = DicomMetadataReader()
    
    # 读取元数据
    if args.file:
        metadata = reader.read_dicom_metadata(args.file, args.tags)
        if metadata:
            # 打印元数据
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # 如果指定了输出文件，则导出
            if args.output_json:
                reader.export_to_json([metadata], args.output_json)
            if args.output_csv:
                reader.export_to_csv([metadata], args.output_csv)
            
            # 显示统计信息
            if args.show_stats:
                stats = reader.get_metadata_statistics([metadata])
                print("\n=== 元数据统计信息 ===")
                print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif args.folder:
        metadata_list = reader.read_folder_metadata(args.folder, args.recursive, args.tags)
        
        # 如果指定了输出文件，则导出
        if args.output_json:
            reader.export_to_json(metadata_list, args.output_json)
        if args.output_csv:
            reader.export_to_csv(metadata_list, args.output_csv)
        
        # 显示统计信息
        if args.show_stats:
            stats = reader.get_metadata_statistics(metadata_list)
            print("\n=== 元数据统计信息 ===")
            print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()