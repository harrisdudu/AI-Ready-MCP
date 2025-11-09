# Dots.OCR 环境配置指南

## 快速安装

### 1. 创建虚拟环境
```bash
conda create -n run_ocr python=3.10
conda activate run_ocr
```

### 2. 安装基础依赖
```bash
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev
sudo apt-get install libzbar0
pip install pyzbar==0.1.9 imagehash==4.3.2 aiofiles

pip install -r requirements.txt
```

### 3. 添加badcase的收集路径权限
```bash
sudo chmod 777 /mnt/dotsocr_badcase
```


## 依赖说明

### 必需依赖
- `pillow-simd==9.5.0.post2` - 高性能图像处理
- `PyMuPDF==1.26.3` - PDF处理和图像提取
- `openai==1.90.0` - vLLM API客户端
- `tqdm==4.67.1` - 进度条显示
- `opencv-pytho==4.5.0` - 计算机视觉处理
- `numpy>=2.1.2` - 数值计算
- `requests>=2.25.0` - HTTP请求
- `aiohttp>=3.8.0` - 异步HTTP客户端
- `httpx>=0.24.0` - HTTP客户端
- `pyzbar` - JSON格式修复工具
- `ImageHash` - JSON格式修复工具

## Python版本要求
- **推荐版本**: Python 3.10

## 验证安装

运行以下命令验证环境配置：
```bash
python -c "
import sys
print(f'Python版本: {sys.version}')
try:
    import fitz
    print('✓ PyMuPDF 安装成功')
except ImportError as e:
    print(f'✗ PyMuPDF 安装失败: {e}')

try:
    import openai
    print('✓ OpenAI 安装成功')
except ImportError as e:
    print(f'✗ OpenAI 安装失败: {e}')

try:
    import PIL
    print('✓ Pillow-simd 安装成功')
except ImportError as e:
    print(f'✗ Pillow-simd 安装失败: {e}')

try:
    import cv2
    print('✓ OpenCV 安装成功')
except ImportError as e:
    print(f'✗ OpenCV 安装失败: {e}')

"
```

## 常见问题

### 1. 导入错误
如果遇到`ImportError`，请检查：
- Python版本是否符合要求
- 是否安装了所有必需的依赖
- 虚拟环境是否正确激活


### 3. 内存不足
如果遇到内存不足问题：
- 减少批处理大小（`--page_concurrency`）
- 跳过空白页（`--skip_blank_pages`）