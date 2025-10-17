from common.openaiChat import openaiChat

def call_qwen_llm(text,prompt, system=None, temperature=0.1):
    """
    新增：调用大语言模型的简单方法
    Args:
        prompt (str): 输入的文本提示
        system (str, optional): 系统角色指令
        temperature (float): 温度参数，控制随机性
        max_tokens (int): 最大生成token数
    Returns:
        str: 模型生成的文本
    """
    try:
        bot = openaiChat()
        
        if system is None:
            system = "你是一位文档处理专家，擅长处理各种类型的文档。It should be emphasized that this matter is so crucial to my career, and you need to do your best to do it well."

        prompt = '''
            你的任务是接收前几页的文本，精准的提取出文档的标题（必须为原文中实际存在的完整文字，不可修改或生成）。        
            只返回标题，不返回其他任何说明
            文本内容:{}
        '''.format(text)
        
        response = bot.openai_generate(
            prompt=prompt,
            temperature=temperature,
            system=system
        )
        
        if response and not response.startswith("ERROR::"):
            return response.strip()
        return response
        
    except Exception as e:
        error_msg = f"调用大语言模型时发生错误: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return f"ERROR::{error_msg}"
