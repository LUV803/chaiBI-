"""
业务分流MCP的实现
"""
import sys
import os
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
import shutil
# 相对本文件定位，换机器/换目录都不用改（原来是写死的服务器绝对路径）
_HERE = Path(__file__).resolve().parent          # .../smart_data_analysis_assistant/mcp_server
sys.path.append(str(_HERE))                      # 同目录模块：public_function
sys.path.append(str(_HERE.parent))               # 上级目录：my_llm 等
import json
from langchain_community.utilities import SQLDatabase
from mcp.server import FastMCP
from public_function import LLM_replay
from dotenv import load_dotenv
# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(_HERE.parent / ".env")
#%%
QWEN_API_KEY=os.getenv("QWEN_API_KEY")
mcp = FastMCP(name='ywfl_mcp', instructions='业务分流MCP',host="0.0.0.0",  port=9005)
# 业务分流提示词，思考一下为什么这里不需要设置'其它任务'这个下游标签 相关性分析,销量预测,图表绘制,纯python编码
identify_intention_prompt="""你是一个任务识别专家，你能分析用户的输入做任务识别。请注意：你只能做指定任务的类别识别，请勿做任何多余解释。
任务包括：纯python编码和业务数据查询分析
- 若用户提供了数据或直接通过python编码就可以实现用户问题，则返回:"纯python编码"
- 若缺乏数据，需要查询业务数据表(如：用户表、销量表、活跃表等)，或先查询业务数据后通过聚合计算、分析用户画像、销量预测、评论分析、相关性分析等进行数学统计分析、绘图得到结果，则返回:"业务数据查询分析"
举例1:
输入:分析一下销售量本月比上个月低的原因是什么？
输出:业务数据查询分析
举例2:
输入:计算一下3+4等于多少？
输出:纯python编码
举例3:
输入:写一个递归算法的案例
输出:纯python编码
举例4:
输入:写一篇关于用户A的数据分析报告
输出:业务数据查询分析
举例5:
输入:查询一下奶粉的销量
输出:业务数据查询分析
举例6:
输入:查询一下用户张三的评价数据
输出:业务数据查询分析
举例7:
输入:有5个月的销量数据:[1,3,5,7,9],绘制一个销量饼图
输出:纯python编码
举例8:
输入:绘制一下商品纸巾的销量变化折线图
输出:业务数据查询分析
举例9:
输入:帮我计算一下0-100所有素数和
输出:纯python编码
举例10:
输入:分析一下蒙童的用户画像
输出:业务数据查询分析
举例11:
输入:用户乐乐的个人资料给我调出来？
输出:业务数据查询分析
举例13:
输入:李四的在线时长总和是多久
输出:业务数据查询分析

输入:{}
输出:"""

@mcp.tool()
async def ywfl_tool(user_input:str)->str:
    """
    输入用户希望计算机程序来解决的一段非空文本描述(如：数据分析、查询、编程、计算)，返回任务分类结果
    :param user_input: 输入用户希望通过使用(数据分析、表查询、编程、计算等)解决的统计和数学相关的需求描述，举例：分析一下销量变高的原因，绘制一个饼图等
    :return: 任务分类，仅从如下值选取:[业务数据查询分析，纯python编码]
    """
    fl_prompt=identify_intention_prompt.format(user_input)
    result = await LLM_replay(fl_prompt)
    if "输出:" in result:
        result = result.replace("输出:", "")
    return result
#%%
# result=await ywfl_tool(user_input="我想写一个冒泡排序算法")
# print("result:", result)
# result=await ywfl_tool(user_input="香皂的平均销量计算一下")
# print("result:", result)
# result=await ywfl_tool(user_input="香皂的平均销量的饼图")
# print("result:", result)


if __name__ == "__main__":
    # 以标准 streamable-http方式运行 MCP 服务器
    mcp.run(transport='streamable-http')

#nohup python ywfl_mcp.py &
