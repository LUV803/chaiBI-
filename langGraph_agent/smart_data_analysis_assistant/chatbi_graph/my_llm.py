#%%
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


#DEEPSEEK llm（默认使用）
llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")

# 通义千问LLM（如需切换，取消下面注释）
# llm = ChatOpenAI(
#     temperature=0,
#     model="qwen-plus-latest", #qwen-plus
#     openai_api_key=os.getenv("QWEN_API_KEY"),
#     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     extra_body={"chat_template_kwargs": {"enable_thinking": False},"parallel_tool_calls":True},
#     parallel_tool_calls=True
# )
#,"parallel_tool_calls":True
