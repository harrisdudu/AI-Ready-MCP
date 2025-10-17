from openai import OpenAI
import base64

class openaiChatVL:

    def __init__(self,):
        # 设置OpenAI客户端
        
        # 如果需要自定义API端点，可以使用以下配置（根据您的实际情况取消注释）
        self.client = OpenAI(

            api_key="123",

            base_url="http://172.16.0.20:31150/v1",
        )

    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def openai_generate(self,image_paths,text_prompt,system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        # 1. 系统消息
        messages = [{"role": "system", "content": system}]
        
        # 2. 用户消息：图片部分
        if image_paths:
            image_messages = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.encode_image(path)}"}
                }
                for path in image_paths
            ]
            messages.append({"role": "user", "content": image_messages})
        
        # 3. 用户消息：文本提示部分（单独一条消息）
        messages.append({"role": "user", "content": text_prompt})
        
        # 调用 API
        response = self.client.chat.completions.create(
            # model="internvl3-38b",
            # model="doubao-seed-1-6-thinking-250615",
            # model = "/models/Qwen2.5-VL-32B-Instruct",
            model="Qwen2.5-VL-32B-Instruct",
            messages=messages,
            # temperature=0.25,
            # top_p=0.8,
            # presence_penalty=1.05,
            # extra_body={"top_k": 10, 
            #                 "MinP": 0, 
            #                 "chat_template_kwargs": {"enable_thinking": False} #关闭思考
            #                 },
        )
        
        return response.choices[0].message.content.strip()


if __name__ == '__main__':
    bot = openaiChatVL()
    system = "你是一位文档图像分析专家，可以精准的提取文档图片中的标题。It should be emphasized that this matter is so crucial to my career, and you need to do your best to do it well."
    image_paths = [
        '/mnt/data/zzj/data_clean_fire/data/extract_img/消防数据第二次提供/07.机关党委/管理类/最终版/page_1.png',
        '/mnt/data/zzj/data_clean_fire/data/extract_img/消防数据第二次提供/07.机关党委/管理类/最终版/page_2.png',
        '/mnt/data/zzj/data_clean_fire/data/extract_img/消防数据第二次提供/07.机关党委/管理类/最终版/page_3.png',
    ]
    text_prompt = "请根据提供的提供的图片，精准的提取出文档的标题,必须为原图片实际存在的完整文字，不可修改或生成），只返回标题，不返回其他任何说明"   
    print(bot.openai_generate(system=system,image_paths=image_paths,text_prompt=text_prompt))

