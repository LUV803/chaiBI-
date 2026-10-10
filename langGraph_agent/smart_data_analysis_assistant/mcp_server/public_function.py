import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 大模型客户端只建一次并复用。
# 原来是在 LLM_replay 内部每次调用都 new 一个 AsyncOpenAI（各自一条连接池，
# 高频调用下浪费连接）；但也不能提到 import 时无条件创建 —— 那样没配 key
# 的机器连 import 都会失败。所以做成"首次使用时创建 + 缓存"。
_llm_client = None
_llm_client_key = None


def _get_llm_client():
    """按需创建并缓存 AsyncOpenAI 客户端。"""
    global _llm_client, _llm_client_key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if _llm_client is None or _llm_client_key != api_key:
        _llm_client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        _llm_client_key = api_key
    return _llm_client


#定义大模型调用函数，用于处理文本类模型生成功能
async def LLM_replay(messages):
    """
    prompt_template:大模型调用的提示词模板
    message:大模型调用的用户输入
    """
    llm_client = _get_llm_client()
    result =await llm_client.chat.completions.create(
        model="deepseek-chat", #推荐使用:deepseek-chat deepseek-reasoner
        messages=[{"role":"user","content":messages}],
        max_tokens=8192,
        temperature=0)
    return result.choices[0].message.content

# result=await LLM_replay(messages="你好")
# print(result)