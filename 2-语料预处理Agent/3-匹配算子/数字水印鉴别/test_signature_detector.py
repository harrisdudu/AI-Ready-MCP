#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF文档签名识别工具测试脚本

此脚本用于测试pdf_signature_detector模块的功能，包括:
- 测试文件存在性检查
- 测试文件类型检查
- 测试签名检测功能
- 测试错误处理
"""
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from  import detect_pdf_signatures, print_signature_info


class TestPDFSignatureDetector(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时测试文件
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建一个非PDF文件用于测试
        self.non_pdf_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")
        
        # 创建一个不存在的文件路径
        self.nonexistent_file = os.path.join(self.temp_dir, "nonexistent.pdf")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除临时文件
        if os.path.exists(self.non_pdf_file):
            os.remove(self.non_pdf_file)
        
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_nonexistent_file(self):
        """测试处理不存在的文件"""
        result = detect_pdf_signatures(self.nonexistent_file)
        self.assertIsNone(result)
    
    def test_non_pdf_file(self):
        """测试处理非PDF文件"""
        result = detect_pdf_signatures(self.non_pdf_file)
        self.assertIsNone(result)
    
    @patch('pdf_signature_detector.PdfReader')
    def test_pdf_without_signatures(self, mock_pdf_reader):
        """测试处理没有签名的PDF文件"""
        # 设置模拟对象的行为
        mock_reader = MagicMock()
        mock_reader.metadata = {}
        mock_reader._root = {}
        mock_reader.pages = []
        mock_pdf_reader.return_value = mock_reader
        
        # 创建一个模拟的PDF文件路径
        mock_pdf_path = os.path.join(self.temp_dir, "mock.pdf")
        
        # 创建一个空文件作为模拟PDF
        open(mock_pdf_path, 'a').close()
        
        try:
            # 调用被测试函数
            result = detect_pdf_signatures(mock_pdf_path)
            
            # 验证结果
            self.assertIsNotNone(result)
            self.assertEqual(result['file_path'], mock_pdf_path)
            self.assertFalse(result['has_signatures'])
            self.assertEqual(result['signature_count'], 0)
            self.assertEqual(result['signatures'], [])
        finally:
            # 清理模拟文件
            if os.path.exists(mock_pdf_path):
                os.remove(mock_pdf_path)
    
    @patch('pdf_signature_detector.PdfReader')
    def test_pdf_with_signatures(self, mock_pdf_reader):
        """测试处理有签名的PDF文件"""
        # 创建模拟的签名字段对象
        mock_sig_field = MagicMock()
        mock_sig_field.get_object.return_value = {
            '/FT': '/Sig',
            '/T': 'TestSignature',
            '/V': MagicMock()
        }
        
        # 创建模拟的签名字典
        mock_sig_dict = MagicMock()
        mock_sig_dict.get_object.return_value = {
            '/Location': 'Test Location',
            '/Reason': 'Test Reason',
            '/ContactInfo': 'test@example.com'
        }
        
        # 设置模拟对象的行为
        mock_reader = MagicMock()
        mock_reader.metadata = {}
        mock_reader._root = {'/AcroForm': {'/Fields': [mock_sig_field]}}
        mock_reader.pages = []
        mock_pdf_reader.return_value = mock_reader
        
        # 设置签名字段的V属性返回模拟的签名字典
        mock_sig_field.get_object.return_value['/V'] = mock_sig_dict
        
        # 创建一个模拟的PDF文件路径
        mock_pdf_path = os.path.join(self.temp_dir, "mock_signed.pdf")
        
        # 创建一个空文件作为模拟PDF
        open(mock_pdf_path, 'a').close()
        
        try:
            # 调用被测试函数
            result = detect_pdf_signatures(mock_pdf_path)
            
            # 验证结果
            self.assertIsNotNone(result)
            self.assertTrue(result['has_signatures'])
            self.assertEqual(result['signature_count'], 1)
            self.assertEqual(len(result['signatures']), 1)
            
            # 验证签名信息
            signature = result['signatures'][0]
            self.assertEqual(signature['field_name'], 'TestSignature')
            self.assertEqual(signature['type'], 'AcroForm Signature')
            self.assertEqual(signature['location'], 'Test Location')
            self.assertEqual(signature['reason'], 'Test Reason')
            self.assertEqual(signature['contact_info'], 'test@example.com')
        finally:
            # 清理模拟文件
            if os.path.exists(mock_pdf_path):
                os.remove(mock_pdf_path)
    
    @patch('pdf_signature_detector.PdfReader')
    def test_error_handling(self, mock_pdf_reader):
        """测试错误处理功能"""
        # 设置模拟对象抛出异常
        mock_pdf_reader.side_effect = Exception("Test exception")
        
        # 创建一个模拟的PDF文件路径
        mock_pdf_path = os.path.join(self.temp_dir, "mock_error.pdf")
        
        # 创建一个空文件作为模拟PDF
        open(mock_pdf_path, 'a').close()
        
        try:
            # 调用被测试函数
            result = detect_pdf_signatures(mock_pdf_path)
            
            # 验证结果
            self.assertIsNone(result)
        finally:
            # 清理模拟文件
            if os.path.exists(mock_pdf_path):
                os.remove(mock_pdf_path)
    
    @patch('builtins.print')
    @patch('pdf_signature_detector.detect_pdf_signatures')
    def test_print_signature_info(self, mock_detect, mock_print):
        """测试打印签名信息功能"""
        # 设置模拟的检测结果
        mock_result = {
            'file_path': 'test.pdf',
            'file_size': 1024,
            'page_count': 1,
            'has_signatures': True,
            'signature_count': 1,
            'signatures': [{
                'field_name': 'TestSignature',
                'type': 'AcroForm Signature',
                'location': 'Test Location',
                'reason': 'Test Reason',
                'contact_info': 'test@example.com'
            }],
            'metadata': {}
        }
        mock_detect.return_value = mock_result
        
        # 调用被测试函数
        print_signature_info(mock_result)
        
        # 验证print函数被调用
        self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()