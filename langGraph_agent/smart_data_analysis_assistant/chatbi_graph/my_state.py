from typing import TypedDict, Annotated,List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
class BIState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # 手动管理消息
# {"messages": [AIMessages(content=xxxx),【AI,,,】,【HUMNA,...】]}

# class BIState(TypedDict):
#      messages: [List[AnyMessage]] # 手动管理消息 , add_messages