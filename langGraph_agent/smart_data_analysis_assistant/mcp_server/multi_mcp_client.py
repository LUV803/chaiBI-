#%%
"""
一个 MCP client实现多个 MCP server连接+调用实战
pip install openai
pip install mcp
pip install matplotlib
pip install dotenv
"""
import json
from mcp.client.sse import sse_client  #sse_client
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from dotenv import load_dotenv
from contextlib import AsyncExitStack
import asyncio
from mcp import ClientSession
import os
# 原来这里往 sys.path 塞的是本文件所在目录（写死的服务器绝对路径），
# 纯属多余：本文件不导入任何同目录模块。已删除。
from pathlib import Path
# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
#
# 原来这里是 load_dotenv() + load_dotenv(dotenv_path=当前目录/".env") 两句：
# 第二句指向的是 mcp_server/.env —— 那个文件根本不存在，所以它一直是空操作；
# 真正生效的是第一句"向上查找"。两行里有一行是死的，且没人知道。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 加载 MCP 服务器的地址（从 .env 的 server_url 读）
server_url=os.getenv("server_url")
#%%
#通用MCP连接管理类
class MCPClient:
    def __init__(self):
        """初始化 MCP Streamable HTTP 客户端"""
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")  # 读取 OpenAI API Key QWEN_API_KEY
        self.base_url ="https://api.deepseek.com"   # 读取 BASE URL https://dashscope.aliyuncs.com/compatible-mode/v1
        self.model = "deepseek-chat"  # 读取 model qwen-plus
        if not self.api_key:
            raise ValueError("未找到 API KEY. 请在 .env 文件中配置 API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.sessions = {}  # 存储多个服务端会话
        self.tools_map = {}  # 工具映射：工具名称 -> (服务端 ID, 端点URL)
    async def connect_to_server(self, server_id: str, endpoint_url: str,protocal_type:str):
        """
        连接到 MCP SSE/steamble HTTP/stdio 服务器
        :param server_id: 服务端标识符
        :param endpoint_url: 服务端端点URL
        """
        if server_id in self.sessions:
            raise ValueError(f"服务端 {server_id} 已经连接")
        # 连接到sse服务器或者streamable-http服务器(这里不考虑stdio类型的服务器)
        if protocal_type=="sse":
            model_transport = await self.exit_stack.enter_async_context(
                sse_client(endpoint_url,timeout=10000,sse_read_timeout=10000)) #,sse_read_timeout=100,timeout=120
            print("stream_transport:", model_transport)
            read_stream, write_stream = model_transport  # , _ 注意sse返回只有read_steam和write_stream
            # 创建会话
        else: #如果protocal_type==streamable-http
            model_transport=await self.exit_stack.enter_async_context(
                streamablehttp_client(endpoint_url,timeout=10000,sse_read_timeout=10000)
            )
            read_stream, write_stream, transport = model_transport

        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream))
        await session.initialize()
        self.sessions[server_id] = {
            "session": session,
            "read_stream": read_stream,
            "write_stream": write_stream,
            "endpoint_url": endpoint_url,
            "protocal_type":protocal_type #协议类型
        }
        print(f"已连接到 MCP {protocal_type} 服务器: {server_id}")
        print("self.sessions:",self.sessions)
        # 更新工具映射
        response = await session.list_tools()
        print("response:",response)
        for tool in response.tools:
            self.tools_map[tool.name] = (server_id, endpoint_url)
    async def list_tools(self):
        """列出所有服务端的工具"""
        if not self.sessions:
            print("没有已连接的服务端")
            return
        print("已连接的服务端工具列表:")
        for tool_name, (server_id, _) in self.tools_map.items():
            print(f"工具: {tool_name}, 来源服务端: {server_id}")
    async def process_query(self, messages: list) -> str:
        """
        处理用户查询，支持多次工具调用
        :param messages: 消息历史列表
        :return: 最终响应内容
        """
        results = []

        # 构建统一的工具列表
        available_tools = []
        for tool_name, (server_id, _) in self.tools_map.items():
            session = self.sessions[server_id]["session"]
            response = await session.list_tools()
            for tool in response.tools:
                if tool.name == tool_name:
                    # 确保函数名符合规范（替换连字符为下划线）
                    safe_name = tool.name.replace('-', '_')
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": safe_name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
        print('整合的服务端工具列表:', available_tools)

        # 循环处理工具调用
        while True:
            # 请求模型处理
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages[-8:], #取历史8轮消息以免上下文过长
                tools=available_tools,
                tool_choice="auto",
                max_tokens=8192,
                temperature=0,
                stream=False,
                timeout=60,
                # parallel_tool_calls=True,  # 通义千问特有的控制
            )
            print("模型响应:", response)
            # 检查是否需要工具调用
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                print("工具调用请求:", tool_calls)
                # 执行每个工具调用
                for tool_call in tool_calls:
                    # 恢复原始工具名（将下划线转换回连字符）
                    original_tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    # 根据工具名称找到对应的服务端
                    server_info = self.tools_map.get(original_tool_name)
                    if not server_info:
                        raise ValueError(f"未找到工具 {original_tool_name} 对应的服务端")
                    server_id, _ = server_info
                    session = self.sessions[server_id]["session"]
                    print(f"\n调用工具 {original_tool_name} (服务端: {server_id}) 参数: {tool_args}\n")
                    tool_result = await session.call_tool(original_tool_name, tool_args)
                    print("tool_result:", tool_result)
                    # 将工具调用结果添加到消息历史中
                    results.append(tool_result.content[0].text)
                messages=([{"role": "user", "content": f"原始问题：{messages}\n工具结果：{json.dumps(results)}"}]) #将用户问题和工具得到的答案拼接在一起组成上下文
            else:
                # 如果没有工具调用，返回最终响应
                return response.choices[0].message.content

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("multi MCP客户端已启动！输入 'exit' 退出")
        while True:
            try:
                query = input("问: ").strip()
                if query.lower() == 'exit':
                    break
                response = await self.process_query([{"role": "user", "content": query}])
                print(f"AI回复: {response}")
            except Exception as e:
                print(f"发生错误: {str(e)}")
    async def clean(self):
        """清理所有资源"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()

#定义工具名和工具函数之间的对应关系
async def main():
    # 启动并初始化 MCP 客户端
    client = MCPClient()
    print(server_url)
    try:
        # 连接多个 MCP Streamable HTTP 服务器
        #连接数据库数据分析MCP服务器
        await client.connect_to_server(
            "search_db_mcp",
            f"http://{server_url}:9004/sse",  #SSE协议的固定路由 9004是注释版
            "sse"
        )
        # #连接Python代码执行图形数据分析MCP服务器
        await client.connect_to_server(
            "python_chart_mcp",
            f"http://{server_url}:9002/mcp",  #STREAMBALE HTTP协议的固定路由
            "streamable-http"
        )
        #连接机器学习数据分析MCP服务器
        await client.connect_to_server(
            "machine_learning_mcp",
            f"http://{server_url}:9003/mcp",  #STREAMBALE HTTP协议的固定路由
            "streamable-http"
        )
        # 列出可用工具
        await client.list_tools()
        # 运行交互式聊天循环
        await client.chat_loop()
    finally:
        # 清理资源
        await client.clean()
if __name__ == "__main__":
    asyncio.run(main())

#%%
#text2sql
#查询一下运动用品平均价格与食品平均价格哪个高
#手工艺品总共有多少个商品
#分析下梦彤的用户画像，从其购买的物品、职业、个人描述来分析
#对用户“梦彤”做一下产品推荐

#text2python
#写一段从1+2+..+100的python代码，并执行
#通过写一段python代码,计算一下365882*876545等于多少

#text2sql+text2python实现chart+machine_learning
#查询商品洗碗布的月销量数据，使用python语言生成代码完成绘制一张以月为维度的销量柱状图
#查询商品"洗碗布"的月销量数据，绘制一张以月为维度的销量柱状图
#查询商品"洗碗布"的月销量数据，写一段python代码并运行实现绘制一张以月为维度的销量柱状图
#查询银耳的用户评论和星级数据，并分析评论好坏与星级两者的相关性，是否是星级越高用户越满意


