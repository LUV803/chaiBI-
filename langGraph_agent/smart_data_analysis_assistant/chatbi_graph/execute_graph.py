"""
命令行方式执行整个工作流（最快的联调入口）。

修复记录：
  原实现里 `event_message_list = whole_history` 是**别名不是拷贝**，
  随后循环中的 `event_message_list.append(...)` 直接改写了 whole_history，
  再叠加下面那条 dict 形式的 append，同一条 AI 回复会以两种形式各进一次历史，
  下一轮就把重复消息全喂给模型。现在每轮用 list(history) 传副本，
  并且只把本轮最终回复写回历史。
"""
import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.append(str(Path(__file__).resolve().parent))

from build_graph import make_graph  # noqa: E402

FALLBACK_REPLY = "（本轮没有产生可展示的回复，请换个说法再试一次）"


async def run_graph():
    """执行工作流。"""
    async with make_graph() as graph:
        history = []  # 纯文本多轮历史，格式与 chat_api 的 history 一致
        while True:
            user_input = input("用户：").strip()
            if user_input.lower() in ("q", "exit", "quit"):
                print("对话结束，拜拜！")
                break
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})
            print("本轮输入消息:", history)

            reply = None
            # 传副本：避免 stream 过程中被就地修改
            async for event in graph.astream({"messages": list(history)}, stream_mode="values"):
                last_message = event["messages"][-1]
                last_message.pretty_print()
                if isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls:
                    reply = last_message.content

            history.append({"role": "assistant", "content": reply or FALLBACK_REPLY})
            if reply is None:
                print(FALLBACK_REPLY)


if __name__ == '__main__':
    asyncio.run(run_graph())

# 测试用例：
# 查询健身手套价格是多少
# 查询运动类商品有多少
# 查询用户王一珂的在线平均时长为多久
# 运动用品平均价格与食品平均价格哪个高
# 抽纸近12个月的总销量和洗手液的总销量哪个更高
# 分析一下王一珂的用户画像，通过其购买的物品、职业和个人描述的数据来分析
# 王一珂和梦彤的消费习惯有什么不同
# 查询一下保鲜袋历史12个月的销量，基于此预测一下未来的销量
# 我想写一段冒泡排序 写一段python代码，实现冒泡排序
# 查询商品洗碗布的月销量数据，绘制一张以月为维度的销量柱状图
# 查询银耳的用户评论和星级数据，并分析评论好坏与星级两者的相关性，是否是星级越高用户越满意
#
# 多轮对话测试案例:
# 查询用户王一珂的用户id / 她的在线平均时长为多久 / 分析一下她的用户画像，从其购买的物品、职业、个人描述来分析
# 抽纸单价是多少？/ 查询一下它过去12个月的销量 / 请帮我再预测一下它下个月的销量 / 根据其12个月的销量与预测的一个月的销量共计13个月绘制一张销量折线图
