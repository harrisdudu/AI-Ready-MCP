# DICOM文件与PACS系统数据传输工具使用说明

## 1. 概述

此工具提供了医学DICOM文件与PACS系统之间的数据传输功能，支持文件发送、查询和检索操作。工具使用Python语言开发，基于pydicom和pynetdicom库实现DICOM协议通信。

## 2. 环境要求与依赖安装

### 2.1 环境要求
- Python 3.6或更高版本
- 支持的操作系统：Windows、Linux、macOS

### 2.2 依赖安装
使用pip安装必要的依赖库：

```bash
pip install pydicom pynetdicom
```

## 3. 脚本功能介绍

脚本提供以下核心功能：

- **发送单个DICOM文件**：将指定的DICOM文件发送到PACS服务器
- **发送目录中的DICOM文件**：批量发送目录中所有DICOM文件到PACS服务器
- **查询PACS服务器**：根据患者ID、姓名、研究日期等条件查询PACS服务器中的DICOM文件
- **检索DICOM文件**：从PACS服务器检索指定研究的DICOM文件并保存到本地

## 4. 命令行参数说明

脚本支持通过命令行参数控制不同功能，基本使用格式为：

```bash
python dicom_pacs_transfer.py <command> [options]
```

### 4.1 全局参数（适用于所有命令）

- `--ip`：PACS服务器IP地址，默认为`127.0.0.1`
- `--port`：PACS服务器端口，默认为`11112`
- `--ae-title`：本地AE标题，默认为`PYNETDICOM`
- `--pacs-ae-title`：PACS服务器AE标题，默认为`PACS`

### 4.2 发送单个文件命令 (`send-file`)

```bash
python dicom_pacs_transfer.py send-file --file <dicom_file_path> [options]
```

- `--file`：DICOM文件路径（必需）

### 4.3 发送目录命令 (`send-dir`)

```bash
python dicom_pacs_transfer.py send-dir --dir <directory_path> [options]
```

- `--dir`：包含DICOM文件的目录路径（必需）

### 4.4 查询命令 (`query`)

```bash
python dicom_pacs_transfer.py query [--patient-id <id>] [--patient-name <name>] [--study-date <date>] [options]
```

- `--patient-id`：患者ID（可选）
- `--patient-name`：患者姓名（可选）
- `--study-date`：研究日期，格式为`YYYYMMDD-YYYYMMDD`（可选）

### 4.5 检索命令 (`retrieve`)

```bash
python dicom_pacs_transfer.py retrieve --study-uid <study_uid> --output-dir <output_directory> [options]
```

- `--study-uid`：研究实例UID（必需）
- `--output-dir`：保存检索到的DICOM文件的目录（必需）

## 5. 使用示例

### 5.1 发送单个DICOM文件

```bash
python dicom_pacs_transfer.py send-file --file "D:\dicom_files\image1.dcm" --ip 192.168.1.100 --port 11112
```

### 5.2 发送目录中的所有DICOM文件

```bash
python dicom_pacs_transfer.py send-dir --dir "D:\dicom_files" --ip 192.168.1.100 --port 11112 --pacs-ae-title MY_PACS
```

### 5.3 查询特定患者的DICOM文件

```bash
python dicom_pacs_transfer.py query --patient-id "P12345" --ip 192.168.1.100 --port 11112
```

### 5.4 按日期范围查询DICOM文件

```bash
python dicom_pacs_transfer.py query --study-date "20230101-20231231" --ip 192.168.1.100 --port 11112
```

### 5.5 检索特定研究的DICOM文件

```bash
python dicom_pacs_transfer.py retrieve --study-uid "1.2.3.4.5.6.7.8.9.10" --output-dir "D:\downloaded_dicoms" --ip 192.168.1.100 --port 11112
```

## 6. 配置说明

除了通过命令行参数配置外，您还可以直接修改脚本中的默认配置值：

```python
# 在脚本中找到以下部分并修改默认值
self.pacs_ip = pacs_ip  # 默认值: '127.0.0.1'
self.pacs_port = pacs_port  # 默认值: 11112
self.ae_title = ae_title  # 默认值: 'PYNETDICOM'
self.pacs_ae_title = pacs_ae_title  # 默认值: 'PACS'
self.timeout = timeout  # 默认值: 30秒
```

## 7. 日志记录

脚本会同时在控制台和日志文件中记录操作信息：
- 控制台：实时显示操作状态和结果
- 日志文件：`dicom_transfer.log`，记录详细的操作日志，包括时间戳、日志级别和消息

日志级别默认为`INFO`，您可以在脚本中修改为`DEBUG`以获取更详细的信息：

```python
logging.basicConfig(
    level=logging.DEBUG,  # 修改日志级别为DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dicom_transfer.log"),
        logging.StreamHandler()
    ]
)
```

## 8. 常见问题与解决方案

### 8.1 连接失败
- **问题**：无法连接到PACS服务器
- **解决方案**：检查PACS服务器IP地址、端口是否正确，确保PACS服务器正在运行且允许外部连接

### 8.2 DICOM文件发送失败
- **问题**：DICOM文件发送失败
- **解决方案**：确认文件是有效的DICOM格式，检查PACS服务器是否有足够的存储空间和权限

### 8.3 查询无结果
- **问题**：查询返回空结果
- **解决方案**：检查查询条件是否正确，尝试使用更宽松的查询条件

### 8.4 检索失败
- **问题**：无法检索DICOM文件
- **解决方案**：确认Study Instance UID是否正确，检查本地存储目录权限和可用空间

## 9. 在Python代码中集成使用

除了作为命令行工具使用外，您还可以在自己的Python代码中集成此工具：

```python
from dicom_pacs_transfer import DicomPacsTransfer

# 创建传输实例
transfer = DicomPacsTransfer(
    pacs_ip='192.168.1.100',
    pacs_port=11112,
    ae_title='MY_APP',
    pacs_ae_title='PACS'
)

# 发送文件
success = transfer.send_dicom_file('path/to/dicom/file.dcm')

# 查询DICOM文件
results = transfer.query_pacs({
    'QueryRetrieveLevel': 'STUDY',
    'PatientID': 'P12345'
})

# 检索DICOM文件
result = transfer.retrieve_study('1.2.3.4.5.6.7.8.9.10', 'output/directory')
```

## 10. 注意事项

- 请确保您有适当的权限访问PACS系统和处理患者数据
- 遵守相关的医疗隐私法规（如HIPAA）
- 在生产环境使用前，建议进行充分的测试
- 定期备份重要的DICOM数据

## 11. 更新与维护

- 定期更新依赖库到最新版本：`pip install --upgrade pydicom pynetdicom`
- 根据PACS系统的变更，及时调整配置参数
- 如遇到问题，请查看日志文件获取详细信息