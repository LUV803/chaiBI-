"""
构建 ChatBI 的 LangGraph 工作流。

图结构：
    START -> call_identify_intention -> identify_intention_tool_node
              ├─ "纯python编码"      -> call_python_coder ⇄ python_run_tool_node   （循环）
              ├─ "业务数据查询分析"   -> call_list_tables -> list_tables_tool -> data_analysis_agent -> END
              └─ 其它(自由对话/兜底)  -> END

本次修改修复的问题：
  1. 工具发现：原来是一长串 if/elif 给变量赋值，没有 else 分支 ——
     MCP 少返回一个工具就是 NameError；现在改为按名字建字典 + 启动时校验必需工具。
  2. 变量遮蔽：原来 `list_tables_tool = ToolNode([list_tables_tool])` 把工具对象
     覆盖成了节点对象，节点名改成 list_tables_node，不再遮蔽。
  3. 分流判断：原来用 `text == "纯python编码"` 精确相等，模型多一个空格就落空，
     整图直接 END 且用户什么都收不到；现在做归一化 + 子串匹配。
  4. 流程图绘制：原来在 make_graph() 里每次都调 draw_mermaid_png()，
     而它会请求 mermaid.ink 在线渲染 —— 每个 HTTP 请求都走一次外网并写文件，
     断网时直接让服务起不来；现在默认关闭，需要时用 DRAW_GRAPH_PNG=1 开。
  5. 不再从 tools_node 导入提示词 —— 那个模块 import 时就会连数据库。
  6. python 分支原来是单程线 `call_python_coder -> python_run_tool_node -> END`：
     脚本执行完图就结束了，没有任何节点再生成面向用户的回答 ——
     脚本跑通、用户拿到空白。改为与 data 分支对齐的循环 + 轮数上限，
     详见 should_continue_python() 的注释（2026-10-10）。
"""
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

sys.path.append(str(Path(__file__).resolve().parent))

from my_llm import llm  # noqa: E402
from prompts import generate_query_system_prompt  # noqa: E402

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# MCP 服务器地址（.env 里的 server_url）
server_url = os.getenv("server_url", "127.0.0.1")
AGENT_DEBUG = os.getenv("AGENT_DEBUG", "0") == "1"

# python 分支最多"写码→执行"几轮。给上限是因为它现在是循环边：
# 模型如果反复觉得脚本没写对，会一直重写；靠 langgraph 的 recursion_limit
# 兜底会报 GraphRecursionError（对用户是报错），这里主动收敛成"带着已有结果回答"。
MAX_PYTHON_ROUNDS = int(os.getenv("MAX_PYTHON_ROUNDS", "3"))

mcp_server_config = {
    # 数据库查询MCP（SSE）
    "search_db_mcp": {
        "url": f"http://{server_url}:9004/sse",
        "transport": "sse",
        "timeout": 60,
        "sse_read_timeout": 60,
    },
    # 机器学习MCP
    "machine_learning_mcp": {
        "url": f"http://{server_url}:9003/mcp",
        "transport": "streamable_http",
        "timeout": 120,  # 机器学习耗时较长
        "sse_read_timeout": 120,
    },
    # 生成并执行 python 代码的 MCP
    "python_chart_mcp": {
        "url": f"http://{server_url}:9002/mcp",
        "transport": "streamable_http",
        "timeout": 120,
        "sse_read_timeout": 120,
    },
    # 业务分流MCP
    "ywfl_mcp": {
        "url": f"http://{server_url}:9005/mcp",
        "transport": "streamable_http",
        "timeout": 60,
        "sse_read_timeout": 60,
    },
}

# 图里真正要用到的工具，缺一个就明确报错，而不是等到运行时 NameError
REQUIRED_TOOLS = {
    "ywfl_tool",                              # 业务分流
    "list_tables_tool",                       # 表结构
    "db_sql_tool",                            # 执行 SQL
    "run_python_script_tool",                 # 执行 python / 绘图
    "translate_to_python_plot_script",        # 需求 -> 绘图代码
    "reviews_stars_correlation_test_tool",    # 满意度与星级相关性
    "analysis_product_reviews_tool",          # 评论情感分析
    "sales_predict_tool",                     # 销量预测
}

LABEL_PYTHON = "纯python编码"
LABEL_DATA = "业务数据查询分析"

_NORMALIZE = re.compile(r"[\s。.，,、;；:：!！?？\"'“”‘’\[\]【】()（）]")


def _normalize(text: str) -> str:
    return _NORMALIZE.sub("", text or "").lower()


class BIState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _tool_message_text(message) -> str:
    """把消息内容拍平成纯文本。

    MCP 工具返回的 ToolMessage.content 可能是结构化列表
    [{'type':'text','text':'...'}]，必须拼成纯文本才能做判断。
    """
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content else ""


def should_continue_python(state: BIState) -> Literal["python_run_tool_node", "__end__"]:
    """python 分支的循环条件。

    修复前这里是写死的两条边：call_python_coder → python_run_tool_node → END。
    后果不是"偶尔少答一句"，而是**整条分支端到端拿不到答案**：
    脚本执行完，图停在一条 ToolMessage 上就结束了，没有任何节点再生成面向用户的回答，
    界面上是空白。举一个跟安全无关的复现：
        问「用 python 算一下 1 到 100 的累加和，输出结果」
        消息序列 = Human → AI(调 ywfl) → Tool(分流) → AI(调 run_python) → Tool(脚本跑完) → END
        最终回复 = ''（长度 0）
    模型把脚本写对了、跑通了、print 了完整结果，只是那句话没人接。

    改法与 data_analysis_agent 那条分支对齐 —— 让 ToolNode 执行完回答 LLM，
    由 LLM 决定"还要再算一步"还是"可以给结论了"。
    """
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    # 循环上限：数历史里已经执行过几次 run_python_script_tool。
    # 用消息本身计数，不用给 BIState 加字段 —— 这是最小改动，
    # 也让"跑了多少次"在最终消息列表里就看得见（排查时不用另找地方）。
    executed = sum(
        1 for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name == "run_python_script_tool"
    )
    if executed >= MAX_PYTHON_ROUNDS:
        print(f"python 分支达到上限 {MAX_PYTHON_ROUNDS} 轮，强制结束")
        return END
    return "python_run_tool_node"


def should_continue_ywfl(state: BIState) -> Literal["call_python_coder", "call_list_tables", "__end__"]:
    """业务分流后的动态路由。"""
    last_message = state["messages"][-1]
    text = _tool_message_text(last_message)
    normalized = _normalize(text)
    print(f"ywfl分流文本内容: {text!r} -> {normalized!r}")

    if LABEL_PYTHON in normalized:
        return "call_python_coder"
    if LABEL_DATA in normalized:
        return "call_list_tables"
    # 既不是两个分流标签，说明模型在走自由对话（正常回复），结束本轮流。
    # 万一这里是因为模型输出跑偏而落空，chat_api / execute_graph 侧还有兜底回复。
    print(f"ywfl 未匹配到分流标签，按自由对话结束本轮: {text!r}")
    return END


@asynccontextmanager
async def make_graph():
    """定义并编译工作流（作为异步上下文管理器，保持 MCP 会话存活）。"""
    client = MultiServerMCPClient(mcp_server_config)
    tools = []
    # 并行建立 4 个 MCP 会话，保持 client 与 server 之间的连接不断
    async with client.session("python_chart_mcp") as python_chart_session, \
            client.session("ywfl_mcp") as ywfl_session, \
            client.session("machine_learning_mcp") as machine_learning_session, \
            client.session("search_db_mcp") as search_db_session:
        python_chart_server_tools = await load_mcp_tools(python_chart_session)
        ywfl_server_tools = await load_mcp_tools(ywfl_session)
        machine_learning_server_tools = await load_mcp_tools(machine_learning_session)
        search_db_server_tools = await load_mcp_tools(search_db_session)
        for group in (ywfl_server_tools, machine_learning_server_tools,
                      python_chart_server_tools, search_db_server_tools):
            tools.extend(group)

        tools_by_name = {tool.name: tool for tool in tools}
        print(f"已加载 MCP 工具 {len(tools_by_name)} 个: {sorted(tools_by_name)}")
        missing = REQUIRED_TOOLS - tools_by_name.keys()
        if missing:
            raise RuntimeError(
                f"缺少必需的 MCP 工具 {sorted(missing)}；"
                f"请确认 9002~9005 四个 MCP 服务都已启动（当前已加载: {sorted(tools_by_name)}）"
            )

        ywfl_tool = tools_by_name["ywfl_tool"]
        list_tables_tool = tools_by_name["list_tables_tool"]
        db_sql_tool = tools_by_name["db_sql_tool"]
        run_python_script_tool = tools_by_name["run_python_script_tool"]
        translate_to_python_plot_script_tool = tools_by_name["translate_to_python_plot_script"]
        reviews_stars_correlation_test_tool = tools_by_name["reviews_stars_correlation_test_tool"]
        analysis_product_reviews_tool = tools_by_name["analysis_product_reviews_tool"]
        sales_predict_tool = tools_by_name["sales_predict_tool"]

        def call_identify_intention(state: BIState):
            """业务分流节点。"""
            identify_system_message = [SystemMessage(content="""你是一个对话智能助手,你具有语言技能和工具调用技能。
            若用户希望做数据分析、查询业务数据、机器学习建模、做计算、绘制统计图表、写代码等工作，你需要调用`ywfl_tool`工具来实现下游任务分流，
            若用户做纯咨询则使用对话技能完成自由对话""")]
            print("多轮对话开始节点BISTATE消息列表检查点:", state['messages'])
            # 不强制调用 ywfl_tool：纯咨询时允许模型直接自由回复
            llm_with_tools = llm.bind_tools([ywfl_tool])
            ywfl_result = llm_with_tools.invoke(identify_system_message + state['messages'])
            print("业务分流模块参数提取:", ywfl_result)
            return {"messages": [ywfl_result]}

        def call_python_coder(state: BIState):
            """PYTHON直接写程序 + 自动执行节点。"""
            llm_with_tools = llm.bind_tools([run_python_script_tool])
            python_coder_result = llm_with_tools.invoke(state['messages'], parallel_tool_calls=True)
            print("python写程序+执行结果:", python_coder_result)
            return {'messages': [python_coder_result]}

        def call_list_tables(state: BIState):
            """获取数据库表结构信息节点。"""
            tool_call = {
                "name": "list_tables_tool",
                "args": {},
                "id": "call_list_tables",
                "type": "tool_call",
            }
            return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        identify_intention_tool_node = ToolNode([ywfl_tool], name="identify_intention_tool_node")
        # 注意节点名与工具名区分开：原来的写法把工具变量覆盖成了 ToolNode 对象。
        # ToolNode 的 name 必须与下面 add_node 注册名、add_edge 里引用的名字三者一致，
        # 否则编译期报 "Found edge starting at unknown node"。
        list_tables_node = ToolNode([list_tables_tool], name="list_tables_node")
        python_run_tool_node = ToolNode([run_python_script_tool], name="python_run_tool_node")

        # 数据分析和机器学习、绘图的一个编译好的 ReAct 子图
        data_analysis_agent = create_react_agent(
            model=llm,
            tools=[db_sql_tool,
                   run_python_script_tool,
                   reviews_stars_correlation_test_tool,
                   translate_to_python_plot_script_tool,
                   analysis_product_reviews_tool,
                   sales_predict_tool],
            prompt=generate_query_system_prompt,
            name="data_analysis_agent",
            debug=AGENT_DEBUG,
        )

        workflow = StateGraph(BIState)
        workflow.add_node(call_python_coder)
        workflow.add_node(python_run_tool_node)
        workflow.add_node(call_identify_intention)
        workflow.add_node(identify_intention_tool_node)
        workflow.add_node(call_list_tables)
        workflow.add_node(list_tables_node)
        workflow.add_node(data_analysis_agent)

        workflow.add_edge(START, "call_identify_intention")
        workflow.add_edge("call_identify_intention", "identify_intention_tool_node")
        workflow.add_conditional_edges("identify_intention_tool_node", should_continue_ywfl)
        # python 分支：LLM 写码 → ToolNode 执行 → 回到 LLM 看结果/给结论。
        # 修复前是 add_edge("python_run_tool_node", END)，整条分支不产出回答（见 should_continue_python 注释）。
        workflow.add_conditional_edges("call_python_coder", should_continue_python)
        workflow.add_edge("python_run_tool_node", "call_python_coder")
        workflow.add_edge("call_list_tables", "list_tables_node")
        workflow.add_edge("list_tables_node", "data_analysis_agent")
        workflow.add_edge("data_analysis_agent", END)

        print("正在创建LangGraph图....")
        graph = workflow.compile()  # 如需持久化多轮记忆，可传 checkpointer=MemorySaver()

        # 流程图是构建期产物，默认不再每次绘制（draw_mermaid_png 需要访问 mermaid.ink）
        if os.getenv("DRAW_GRAPH_PNG", "0") == "1":
            try:
                png = graph.get_graph().draw_mermaid_png()
                out_path = Path(__file__).resolve().parent / "build_graph.png"
                out_path.write_bytes(png)
                print(f"流程图已保存: {out_path}")
            except Exception as exc:
                print(f"绘制流程图失败（不影响运行）: {type(exc).__name__}: {exc}")

        yield graph
