# 消防数据清洗复盘

## git地址：

## 1、".doc", ".docx", ".ppt", ".pptx"转pdf

首先以上格式转pdf不要再linux上运行，LibreOffice在linux系统上不支持并发操作，且doc转docx失败率高

在windows上doc，.docx转pdf成功率蛮高的，.ppt和.pptx相对失败次数会多一些，对应的pdf会直接复制过去

**可以在windows上运行以下脚本**

data_clean_0.py



## 2、S3_client下载脚本初版参考

s3_client文件夹



## 3、大语言模型，多模态大模型openAI方式单点调用简单封装

### 3.1 公司内部大模型调用方法以及参考

LLM/common/openaiChat.py, LLM/commom/vlopenaiChat.py



### 3.2 线上大模型调用方法以及参考

LLM/step7_doubao_llm.py 可参考



## 4、其他工具类积累

### 4.1 给定目录，pdf文件选取前几页保存成png图片

step1_pdf_extract_img.py
