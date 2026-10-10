"""
本文件用于向chat_api接口发送请求参数来模拟用户使用chatbi服务
"""
#%%
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# 本文件只依赖 requests / dotenv，不需要往 sys.path 里塞项目目录
# （原来写死的是某台服务器上的绝对路径，换台机器就失效）

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 键名是 server_url，不是 server_api_url —— 原来读错了名字，os.getenv 返回 None，
# 于是拼出 "http://None:9008/..."（os.getenv 读不到只会给 None，不报错）
api_host=os.getenv("server_url")
print(api_host)
api_url=f"http://{api_host}:9008/chatbi_service"
TIMEOUT = 120  # 秒。原来是 20000，requests 的单位是秒，等于 5.5 小时
input_message={
    "user_id":"ABC",
    "message":"面包和鱿鱼丝哪个贵", #你好 查询一下瓜子的单价 查询一下瓜子的单价 面包和鱿鱼丝哪个贵
    "history":[]
}
result=requests.post(api_url,json=input_message,timeout=TIMEOUT)
print("result:",result.json())

#%%
input_message={
    "user_id":"ABC",
    "message":"查询一下瓜子的单价", #你好 查询一下瓜子的单价 查询一下瓜子的单价 面包和鱿鱼丝哪个贵
    "history":[{"role":"user","content":"面包和鱿鱼丝哪个贵"},{"role":"assistant","content":"面包贵一些"}]
}
result=requests.post(api_url,json=input_message,timeout=TIMEOUT)
print("result:",result.json())

