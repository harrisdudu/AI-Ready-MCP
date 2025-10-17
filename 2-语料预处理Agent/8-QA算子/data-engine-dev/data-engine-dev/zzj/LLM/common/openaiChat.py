from openai import OpenAI

class openaiChat:

    def __init__(self,):
        # 设置OpenAI客户端
        
        # 如果需要自定义API端点，可以使用以下配置（根据您的实际情况取消注释）
        self.client = OpenAI(
            # 如果您没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx"
            api_key="123",
            # 填写DashScope SDK的base_url
            base_url="http://172.16.0.20:30023/v1",
        )

    def openai_generate(self, prompt, temperature=0.8, top_p=0.1, max_tokens=8192,
                        system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        response = self.client.chat.completions.create(
            model="Qwen2.5-72B-Instruct",
            # model="/root/models/Qwen2.5-72B-Instruct-GPTQ-Int4",
            messages=[{"role": "system", "content": system}] + [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        res = response.choices[0].message.content.strip()
        return res


if __name__ == '__main__':
    # 你是一位美妆专家，你知道美国的所有化妆品类，请根据你所知道的化妆知识回答问题。需要强调的是，这件事对我的职业生涯至关重要，需要你尽力做好。
    system = "你是一位美妆专家，你知道美国的所有化妆品类，请根据你所知道的化妆知识回答问题。It should be emphasized that this matter is so crucial to my career, and you need to do your best to do it well."

    short_des = ""
    prompt = '''
                Summarize the following in a natural and casual tone, introducing it to someone who may be interested in this product. Do not greet the person, and use simple words. Do not use any bullet points and limit the summary to 200 words. If describing size, compare it something that anyone can understand. If the item is limited to quantity per purchase, has special sale terms or has a special return policy, mention it but talk about it last. Do not pose any questions to the person you are writing this to:
                Product Description:{}
            '''.format(short_des)
    temperature = 0.0
    bot = openaiChat()
    print(bot.openai_generate(prompt=prompt, temperature=temperature, system=system))