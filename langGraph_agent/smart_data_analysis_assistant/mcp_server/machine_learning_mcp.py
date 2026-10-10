#%%
"""
机器学习方法的MCP
pip install scipy
pip install pandas
pip install dotenv
工具一:analysis_product_reviews_tool: 从评论中分析用户情感,挖掘商品的满意度和评价
工具二:reviews_stars_correlation_test_function,分析”某商品“满意度和星级评分之间的相关性(检验用户是不是乱点星级功能)
工具三:sales_predict_tool销量预测工具
...
"""
import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from mcp.server import FastMCP
from dotenv import load_dotenv
import os
import json
from openai import AsyncOpenAI #OPENAI的异步客户端 OPENAI
from public_function import LLM_replay
import asyncio
# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# print(os.getenv("DEEPSEEK_API_KEY"))
#%%
mcp = FastMCP(name='machine_learning_mcp', instructions='机器学习建模MCP',host="0.0.0.0",  port=9003)
#提取用户情感类型的函数
async def extract_function(one_review):
    extract_prompt=f"""你是一个用户商品评价分类大师，你能根据用户评论，对其表达的使用体验进行基于第三人称的提取。
    请注意:
    - "是否满意"取值范围:["满意","不满意","中性"]
    - 输出格式:[商品名称:XXX|是否满意:不满意|原因:XXXX] 若用户没有说原因,原因填写'无'
    举例1:
    输入:鸡腿有效期都过了
    输出:[商品名称:鸡腿|是否满意:不满意|原因:有效期过了]
    举例2:
    输入:奶茶片太干了，我家狗子吃了一直口渴
    输出:[商品名称:奶茶片|是否满意:不满意|原因:太干，狗吃了一直口渴]
    举例3:
    输入:旺仔牛奶真棒，这是我买过最棒的了
    输出:[商品名称:旺仔牛奶|是否满意:满意|原因:买过最棒的]
    举例4:
    输入:百事可乐中规中矩吧，价格也还行
    输出:[商品名称:百事可乐|是否满意:中性|原因:品质中规中矩，价格尚可]
    
    
    输入:{one_review}
    输出:"""
    result=await LLM_replay(extract_prompt)
    if "输出:" in result:
        result=result.replace("输出:","")
    print("result:",result)
    return result

#%%
def format_output(ItemName, product_data):
    satisfaction = product_data['满意度分布']
    proportion = product_data['满意度比例']
    comments = product_data['原因'].replace('评论如下:', '')
    output = f"商品“{ItemName}”总评论数{product_data['总评论数']}。"
    output += f"其中{satisfaction['满意']}人满意({proportion['满意']})，"
    output += f"{satisfaction['不满意']}人不满意({proportion['不满意']})，"
    output += f"{satisfaction['中性']}人持中性意见({proportion['中性']})，"
    output += f"评价如下：{comments}"
    return output
#%%
#工具一:挖掘商品的满意度和评价
@mcp.tool()
async def analysis_product_reviews_tool(reviews_list) -> str:
    """
    输入某商品的购买评论列表, 分析该商品的用户满意度与评价
    :param reviews_list:包含商品名的用户完整的评论数据列表,如:'["黑芝麻很好吃呀，下次还会买"，"黑芝麻会回购，太新鲜了点", "太喜欢这款黑芝麻了，爱了爱了",...]'
    :return: 包含三类满意度统计的分析结果
    """
    whole_review_list = {}  # 数据结构:
    # {
    #   "商品名称": {
    #       "满意度分布": {"满意":3, "中性":2, "不满意":1},
    #       "满意度比例": {"满意":"50%", "中性":"33%", "不满意":"17%"},
    #       "原因": ["原因1","原因2"],
    #       "总评论数": 6
    #   }
    # }
    # 明确定义三种满意度分类，提高LLM输出的容错性
    SATISFIED = ["满意", "非常满意", "五星"]
    NEUTRAL = ["中性", "一般", "还行"]
    DISSATISFIED = ["不满意", "差评", "糟糕"]
    try:
        if type(reviews_list)==str:
            reviews_list=json.loads(reviews_list)
    except:
        return "评论数据解析错误"
    if len(reviews_list)==0:
        return "该商品没有足够的评论数据"
    for review in reviews_list:
        try:
            # 提取评论信息
            review_result = await extract_function(review) #并行调用LLM的模式,多线程调用
            review_result = review_result.strip("[]") #去掉首位的[]字符串
            review_items = review_result.split("|")
            # 解析字段
            ItemName = None #商品名称
            satisfaction = None #满意度
            reason = None #原因
            for item in review_items:
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "商品名称":
                    ItemName = value
                elif key == "是否满意":
                    satisfaction = value
                elif key == "原因":
                    reason = value
            if not all([ItemName, satisfaction, reason]):
                continue
            # 初始化商品记录
            if ItemName not in whole_review_list:
                whole_review_list[ItemName] = {
                    "满意度分布": {"满意": 0, "中性": 0, "不满意": 0},
                    "原因": [],
                    "总评论数": 0
                }
            # 分类统计满意度
            if satisfaction in SATISFIED:
                whole_review_list[ItemName]["满意度分布"]["满意"] += 1
            elif satisfaction in NEUTRAL:
                whole_review_list[ItemName]["满意度分布"]["中性"] += 1
            elif satisfaction in DISSATISFIED:
                whole_review_list[ItemName]["满意度分布"]["不满意"] += 1
            # 添加原因和总数统计
            whole_review_list[ItemName]["原因"].append(reason)
            whole_review_list[ItemName]["总评论数"] += 1
        except Exception as e:
            print(f"评论解析错误: {e}, 评论内容: {review}")
            continue

    # 计算比例和整理数据
    for product, data in whole_review_list.items():
        total = data["总评论数"]
        # 计算百分比（保留整数）
        data["满意度比例"] = {
            "满意": f"{data['满意度分布']['满意'] / total:.0%}",
            "中性": f"{data['满意度分布']['中性'] / total:.0%}",
            "不满意": f"{data['满意度分布']['不满意'] / total:.0%}"
        }
        # 将原因列表合并为字符串
        data["原因"] ="评论如下:"+"|".join(data["原因"])
    #将所有商品字典格式拼接成一个字符串进行输出:
    total_str=""
    for product, data in whole_review_list.items():
        one_product_str=format_output(product, data)
        print("one_product_str:",one_product_str)
        total_str+=one_product_str+"\n"
    return total_str

# async def main():
#     result=await analysis_product_reviews_tool('["这个鲍鱼味道真美味，是我吃过最好吃的了","鲍鱼很新鲜，一看就是很好的食材","这鲍鱼难吃死了","西湖醋鱼很好吃","西湖藕粉难吃死了"]')
#     print(result)
#
# import asyncio
# result=asyncio.run(main())
# print(result)
#%%
#工具二:分析”某商品“满意度和星级评分之间的相关性(检验用户是不是乱点星级功能)
"""
相关系数介绍:
线性相关系数（Pearson correlation） 接近 -1，表示 list1 和 list2 呈负相关（即 "满意" 对应较低数值，"不满意" 对应较高数值）。如果接近1则代表list1和list2呈现正相关
P值（p_value） 衡量统计显著性，如果 p_value < 0.05，则认为相关性显著。
注意:list2和list1中所有元素必须至少有一个值不相同,否则无法计算
相关系数和样本量有关系，小样本下即便相关性很强(相关系数很大)，有可能不显著(p-value高)
"""
from scipy.stats import pearsonr
async def reviews_stars_correlation_test_function(ItemName,star_list,satisfaction_list):
    """
    :param ItemName:商品名称
    :param star_list: 星级列表
    :param satisfaction_list:满意度列表
    :return:相关系数
    """
    # 前置校验：pearsonr 对退化输入（空/长度不等/常量序列）会直接抛 ValueError，
    # 那样模型只会看到一个工具异常，这里提前拦下并给出可读的原因
    if len(star_list) != len(satisfaction_list):
        return f"数据有问题，无法计算相关度: 星级({len(star_list)}条)与满意度({len(satisfaction_list)}条)数量不一致"
    if len(star_list) < 2:
        return f"商品{ItemName}的有效评论不足 2 条，无法计算相关性"
    if len(set(star_list)) < 2:
        return f"商品{ItemName}的所有星级评分都相同（{star_list[0]}星），无法计算相关性"
    if len(set(satisfaction_list)) < 2:
        return f"商品{ItemName}的所有评论情感判定都相同（{satisfaction_list[0]}），无法计算相关性"

    numeric_satisfaction_list=[] #数值型满意、不满意
    for k in satisfaction_list:
        if k=="满意":
            numeric_satisfaction_list.append(2)
        elif k=="中性":
            numeric_satisfaction_list.append(1)
        else: #如果不满意
            numeric_satisfaction_list.append(0)
    #计算皮尔逊相关系数
    corr, p_value = pearsonr(star_list, numeric_satisfaction_list)
    corr=round(corr,3)
    if p_value<=0.05 and corr>0.7:
        return f"商品{ItemName}用户满意度与星级评分呈现强正相关，相关系数：{corr},显著性水平:{p_value:.3f}"
    elif p_value<=0.05 and corr<-0.7:
        return f"商品{ItemName}用户满意度与星级评分呈现强负相关，相关系数：{corr},显著性水平:{p_value:.3f}"
    elif p_value>0.05 and corr>0.7:
        return f"商品{ItemName}用户满意度与星级评分相关性呈强正相关但不显著，相关系数：{corr},显著性水平:{p_value:.3f}"
    elif p_value>0.05 and corr<-0.7:
        return f"商品{ItemName}用户满意度与星级评分相关性呈强负相关但不显著，相关系数：{corr},显著性水平:{p_value:.3f}"
    else:
        return f"商品{ItemName}用户满意度与星级评分相关性不强，相关系数：{corr},显著性水平:{p_value:.3f}"

# async def main():
#     return_value=await reviews_stars_correlation_test_function("银耳",star_list=[5,4,5,3],satisfaction_list=["满意","满意","满意","不满意"])
#     print(return_value)
# asyncio.run(main())
#%%
def _to_list(value, field_name):
    """把模型传来的参数解析成 list。

    原实现用 eval() 解析，等于给"模型输出"开了任意代码执行的入口；
    ast.literal_eval 只解析字面量（数字/字符串/列表/字典等），不执行任何代码。
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 类型不支持: {type(value).__name__}")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} 解析结果不是列表: {type(parsed).__name__}")
    return parsed


#%%
@mcp.tool()
async def reviews_stars_correlation_test_tool(ItemName,reviews,stars):
    #LLM提取用户的满意度(或者从存储的表中读取用户的满意度数据亦可)
    """
    对指定商品名进行用户满意度与星级评分相关性分析，返回相关性描述
    :param ItemName:商品名称
    :param reviews:查询到的包含商品名的完整评论数据列表list格式,例如:["黑芝麻很好吃呀，下次还会买"，"黑芝麻会回购，太新鲜了点", "太喜欢这款黑芝麻了，爱了爱了",...]
    :param stars:查询到的商品星级评分列表list格式,例如:[5,4,4,4,5,...]
    :return:星级评分与满意度之间的相关性描述
    """
    #先将'["很好吃，会回购","一般般，下次不买了","非常难吃，不会回购"]'从字符串格式转成list格式
    try:
        reviews = _to_list(reviews, "reviews")
        stars = _to_list(stars, "stars")
    except Exception as e:
        return f"数据有问题，无法计算相关度,错误原因:{e}"
    #仅仅选出包含商品名称的list作为分析数据集
    review_item_data_list=[]
    stars_item_data_list=[]
    for star,review in zip(stars,reviews):
        review_item_data_list.append(review)
        stars_item_data_list.append(star)
    whole_review_list=[]
    whole_star_list=[]
    #必须多于两个评论才能计算相关性
    if len(review_item_data_list)<2:
        return "该产品没有足够的用户评论数据"
    #对评价列表进行基于LLM的信息抽取
    for review,star in zip(review_item_data_list,stars_item_data_list):
        try:
            review_result = await extract_function(review)
            review_result = review_result.replace("[","").replace("]","")  # 去掉首位的[]字符串
            review_items = review_result.split("|")
            # 解析字段（split(":", 1)：原因里带冒号时不能把后半段丢掉）
            for item in review_items:
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                # 只将提取到的满意度的stars进行后续分析建模
                if key == "是否满意":
                    satisfaction = value
                    whole_review_list.append(satisfaction)
                    whole_star_list.append(star)
        except Exception as e:
            print(f"评论解析错误: {e}, 评论内容: {review}")
            continue
    print("whole_star_list:",whole_star_list,"whole_review_list:",whole_review_list)
    return_corr_message=await reviews_stars_correlation_test_function(ItemName,star_list=whole_star_list,satisfaction_list=whole_review_list)
    return return_corr_message
# ItemName='银耳'
# reviews=str(['银耳无硫熏，孕妇吃着放心！', '银耳颜色发黄，怀疑硫磺熏过！', '银耳宣传“有机”实际无认证，虚假宣传！', '这款银耳泡发率超高，一小朵就能煮一大锅胶质满满的银耳羹！', '煮银耳时加点桂花，香气扑鼻，幸福感爆棚！', '银耳煮完胶质满满，感觉皮肤变好了！', '复购的银耳品质下降，不再买了！', '比超市买的银耳更干净，几乎无杂质，洗一遍就能煮！', '银耳包装密封性好，干燥无杂质，推荐！', '泡发率高的银耳，一小朵能煮一大锅！', '银耳价格虚高，比菜市场贵一倍！', '银耳碎渣少，朵形完整，一看就是精选好货！', '这款银耳质量稳定，每次炖甜品都满意！', '煮完银耳汤黏稠度刚好，口感细腻，喜欢！', '银耳价格实惠，学生党也能天天吃！', '这次银耳有霉味，不敢吃，差评！', '物流慢，银耳到货时包装全破了！', '银耳炖牛奶超好喝，比奶茶健康多了，孩子也喜欢！', '回购N次的银耳，品质稳定，出胶快，性价比之王！', '银耳炖桃胶、皂角米绝配，养颜圣品，女生必囤！', '客服态度超好，银耳问题解答超耐心，购物体验满分！', '包装漏气，银耳受潮结块了！', '银耳分量缩水，和描述不符！', '即食银耳超方便，早上加热2分钟就能喝，营养又省时！', '银耳无硫熏，颜色自然，孕妇和宝宝都能放心吃！', '卖家送的银耳食谱超实用，解锁了超多新吃法！', '包装密封性超好，银耳干燥不返潮，储存方便！', '银耳羹美容养颜，觉得很有营养！', '银耳搭配红枣枸杞绝了，营养又美味！', '银耳品质好，比线下店新鲜，会长期买！', '银耳碎渣太多，泡发后全是小颗粒！', '煮完银耳汤发酸，质量有问题！', '银耳杂质多，洗了三遍还有沙子！', '客服态度差，银耳问题不解决！', '银耳胶质丰富，长期吃感觉皮肤更水润了！', '银耳煮后胶质浓稠，冷藏后像果冻一样，超赞！', '复购多次的银耳，出胶快，性价比超高！', '即食银耳方便快捷，早餐必备！', '银耳煮一小时都不出胶，垃圾食品！', '银耳炖不烂，口感像树皮！', '银耳搭配红枣、枸杞炖煮，香甜滋补，秋冬必备！', '银耳低卡又饱腹，减肥期当代餐超合适！', '银耳泡发后煮粥特别软糯，全家都爱喝！', '银耳分量足，价格比超市便宜，会继续买！', '银耳炖桃胶口感顺滑，愿意复购！', '即食银耳太甜，香精味重，不喜欢！', '银耳口感细腻软糯，炖汤后入口即化，全家人都超爱！', '独立小包装银耳，干净卫生，一次一包刚刚好！', '银耳搭配雪梨炖煮，润肺止咳，换季必备！'])
# stars=str([5, 2, 2, 5, 5, 5, 1, 5, 5, 5, 3, 4, 5, 5, 5, 2, 2, 5, 5, 4, 5, 2, 2, 4, 5, 5, 5, 5, 4, 5, 3, 0, 2, 2, 5, 5, 5, 4, 2, 2, 5, 5, 4, 5, 5, 2, 4, 5, 5])
#
# async def main():
#     return_corr_message=await reviews_stars_correlation_test_tool(ItemName="银耳",reviews=reviews,stars=stars)
#     print(return_corr_message)
# asyncio.run(main())
#%%
# 工具三:进行时间序列建模预测
import numpy as np
import pandas as pd
async def exponential_moving_average(series, alpha, forecast_periods=1):
    """
    输入一个序列数据，计算指数移动平均并预测未来值
    参数:
    series: pandas.Series，原始时间序列数据
    alpha: float，平滑因子，范围(0,1]
    forecast_periods: int，预测未来的期数，默认为1
    返回:
    forecast: pandas.Series，包含原始数据的EMA和平滑后的预测值
    """
    # 计算EMA
    ema = series.ewm(alpha=alpha, adjust=False).mean()
    print("ema:",ema)
    # 获取最后一个EMA值作为预测基础
    last_ema = list(ema)[-1]
    return last_ema

@mcp.tool()
async def sales_predict_tool(ItemName,sales_data_list):
    """
    输入某商品的销量数据列表,预测该商品下一期的销量
    :param ItemName:商品名称
    :param sales_data_list:历史销量数据list,格式如:[100,200,300,...]
    :return:销量预测结果
    """
    # alpha 是平滑因子：越大越看重近期数据，记忆窗口约为 1/alpha 期。
    # 取 0.5 => 约 2 期的窗口，明显偏向最近几个月；若希望更多参考长期历史（本案例是
    # 电商商品，销量受季节/节假日影响较强），应把 alpha 调小（例如 0.2）。
    alpha = 0.5
    if not sales_data_list:
        return f"商品{ItemName}没有可用于预测的销量数据"
    try:
        # 用 float64：float16 只有 3 位有效数字，几百上千的销量会被直接截断
        series = pd.Series(np.asarray(sales_data_list, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        return f"商品{ItemName}的销量数据无法解析为数值序列: {exc}"
    forecast =await exponential_moving_average(series, alpha, forecast_periods=1)
    print("销量预测结果forecast:",forecast)
    str_history=str(sales_data_list) #历史销量数据
    return f"商品{ItemName}历史销量数据为:{str_history},经过指数平滑移动平均预测下一个月的销量为:{forecast}"

# async def main():
#     forecast_message=await sales_predict_tool(ItemName="西瓜",sales_data_list=[100,200,300,500,300,200])
#     print("指数平滑预测销量:",forecast_message)
# asyncio.run(main())



#%%
if __name__ == "__main__":
    # 以标准 streamable-http方式运行 MCP 服务器
    mcp.run(transport='streamable-http')

#nohup python machine_learning_mcp.py &