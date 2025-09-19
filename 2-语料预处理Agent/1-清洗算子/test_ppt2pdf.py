import os
import logging
from improved_ppt2pdf import ppt_to_pdf_with_notes

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_convert_ppts_to_pdfs(input_dir, output_dir, **kwargs):
    """
    批量将目录中的所有PowerPoint文件转换为带备注的PDF

    参数:
    input_dir (str): 包含PowerPoint文件的目录
    output_dir (str): 输出PDF文件的目录
    **kwargs: 传递给ppt_to_pdf_with_notes的其他参数

    返回:
    dict: 包含每个文件转换结果的字典
    """
    results = {}
    success_count = 0
    fail_count = 0

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取目录中的所有PowerPoint文件
    ppt_extensions = ['.pptx', '.ppt']
    ppt_files = []

    for file in os.listdir(input_dir):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in ppt_extensions:
            ppt_files.append(file)

    total_files = len(ppt_files)
    logger.info(f"找到 {total_files} 个PowerPoint文件待转换")

    for i, ppt_file in enumerate(ppt_files, start=1):
        ppt_path = os.path.join(input_dir, ppt_file)
        pdf_file = os.path.splitext(ppt_file)[0] + '_with_notes.pdf'
        pdf_path = os.path.join(output_dir, pdf_file)

        logger.info(f"转换文件 {i}/{total_files}: {ppt_file}")
        success = ppt_to_pdf_with_notes(ppt_path, pdf_path, **kwargs)

        results[ppt_file] = {
            'success': success,
            'pdf_path': pdf_path if success else None
        }

        if success:
            success_count += 1
        else:
            fail_count += 1
            logger.error(f"转换失败: {ppt_file}")

    logger.info(f"批量转换完成: {success_count} 成功, {fail_count} 失败")
    return results

if __name__ == "__main__":
    # 示例1: 单个文件转换
    logger.info("=== 示例1: 单个文件转换 ===")
    input_ppt = r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座.pptx"
    output_pdf = r"C:\Users\56401\Desktop\temp\10869-10869规划展示馆讲座_with_notes.pdf"
    
    success = ppt_to_pdf_with_notes(
        input_ppt,
        output_pdf,
        notes_font_size=20,  # 调整备注字体大小
        delete_temp=True     # 删除临时文件
    )
    
    if success:
        logger.info(f"示例1转换成功: {output_pdf}")
    else:
        logger.error("示例1转换失败")

    # 示例2: 批量转换
    logger.info("\n=== 示例2: 批量转换 ===")
    input_dir = r"C:\Users\56401\Desktop\temp\ppts"
    output_dir = r"C:\Users\56401\Desktop\temp\pdfs"
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        logger.info(f"创建示例输入目录: {input_dir}")
        logger.info("请将PowerPoint文件放入此目录后重新运行脚本")
    else:
        batch_results = batch_convert_ppts_to_pdfs(
            input_dir,
            output_dir,
            notes_font_size=18,
            delete_temp=True
        )

        # 打印批量转换结果
        logger.info("\n批量转换结果:")
        for file, result in batch_results.items():
            status = "成功" if result['success'] else "失败"
            logger.info(f"- {file}: {status}")
            if result['success']:
                logger.info(f"  输出文件: {result['pdf_path']}")

    # 示例3: 保留临时文件（用于调试）
    logger.info("\n=== 示例3: 保留临时文件（用于调试） ===")
    debug_ppt = r"C:\Users\56401\Desktop\temp\sample.pptx"
    debug_pdf = r"C:\Users\56401\Desktop\temp\sample_with_notes.pdf"
    
    if os.path.exists(debug_ppt):
        success = ppt_to_pdf_with_notes(
            debug_ppt,
            debug_pdf,
            notes_font_size=24,
            delete_temp=False  # 保留临时文件
        )
        
        if success:
            logger.info(f"示例3转换成功，临时文件已保留")
        else:
            logger.error("示例3转换失败")
    else:
        logger.info(f"示例3跳过: 未找到测试文件 {debug_ppt}")