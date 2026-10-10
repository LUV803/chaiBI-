"""
ChatBI 回归测试。

为什么需要它
------------
提示词、SQL 工具、沙箱这几处改动，影响面都不局限于"我改的那一个问题"。
改完之后如果只手工试一两个问题，很容易把别的地方打坏而不知道
（本项目就真发生过：给 SQL 工具换执行方式时，把 `LIKE '%关键字%'`
这条最常见的写法弄坏了，而当时手工试的几个问题恰好都没用到 LIKE）。

于是把踩过的坑固化成用例，改完跑一遍就知道有没有回归。

分两层
------
  离线层（默认就跑）：不需要大模型 —— MCP 工具是否齐全、SQL 只读是否还拦得住、
                      LIKE 是否可用、沙箱是否还关得住、图是否能构建、分流归一化。
                      免费、确定、几秒钟出结果。
  大模型层（--with-llm）：需要调用模型，会消耗 API 额度。检查的是"行为"而不是
                      "字符串"：模型有没有在维度不存在时拒绝作答、有没有把
                      LIMIT 行数当总数、有没有复述筛选条件。

用法
----
    python regression_test.py              # 只跑离线层
    python regression_test.py --with-llm   # 两层都跑

退出码：全部通过为 0，有失败为 1（可直接用于 CI）。

前置条件：9002~9005 四个 MCP 服务已启动。
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK，中文输出会乱码/报错
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402

from build_graph import (  # noqa: E402
    REQUIRED_TOOLS,
    _normalize,
    make_graph,
    mcp_server_config,
    should_continue_ywfl,
)

# --------------------------------------------------------------------------- #
# 断言与计分
# --------------------------------------------------------------------------- #
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"    {detail}" if detail and not ok else ""))
    return ok


def _flatten(content) -> str:
    """把 ToolMessage 的内容拍平成纯文本（可能是 [{'type':'text','text':...}]）。"""
    if isinstance(content, list):
        return "".join(
            i.get("text", "") for i in content
            if isinstance(i, dict) and i.get("type") == "text"
        )
    return str(content or "")


async def call_tool(tool, **kwargs) -> str:
    return _flatten(await tool.ainvoke(kwargs))


# --------------------------------------------------------------------------- #
# 离线层
# --------------------------------------------------------------------------- #
async def test_mcp_inventory() -> None:
    """四个 MCP 服务都要活着，且图里用到的 8 个工具一个都不能少。"""
    print("\n[离线] MCP 工具清单")
    client = MultiServerMCPClient(mcp_server_config)
    found: set[str] = set()
    for name in mcp_server_config:
        try:
            async with client.session(name) as session:
                tools = await load_mcp_tools(session)
                found |= {t.name for t in tools}
            check(f"服务可达: {name}", True)
        except BaseException as exc:  # noqa: BLE001  anyio 会把异常包成 ExceptionGroup
            leaf = exc
            while hasattr(leaf, "exceptions"):
                leaf = leaf.exceptions[0]
            check(f"服务可达: {name}", False, f"{type(leaf).__name__}: {leaf}")

    missing = REQUIRED_TOOLS - found
    check("图所需的 8 个工具齐全", not missing, f"缺失: {sorted(missing)}" if missing else "")


async def test_sql_tool() -> None:
    """SQL 工具：既要拦得住写操作，也不能误伤正常查询。"""
    print("\n[离线] SQL 工具")
    client = MultiServerMCPClient(mcp_server_config)
    async with client.session("search_db_mcp") as session:
        db = {t.name: t for t in await load_mcp_tools(session)}["db_sql_tool"]

        # —— 回归重点：LIKE 曾经因为 % 被当成参数占位符而整体报错 ——
        out = await call_tool(db, query="SELECT count(*) FROM product WHERE product_name LIKE '%运动%'")
        check("LIKE '%关键字%' 可用（曾经整体报错）",
              "错误" not in out and "行数据" in out, out[:120])

        out = await call_tool(db, query='SELECT count(*)::int FROM product')
        check(":: 类型转换可用", "错误" not in out, out[:120])

        out = await call_tool(db, query='SELECT "1月销量" FROM sales LIMIT 2')
        check("中文列名双引号可用", "错误" not in out, out[:120])

        # —— 0 行是合法结果，不能被当成错误（旧实现会误报，导致模型反复"修正"对 SQL）——
        out = await call_tool(db, query="SELECT * FROM product WHERE 1=0")
        check("0 行结果表述为'共 0 行数据'而非报错",
              "共 0 行数据" in out, out[:120])

        # —— 只读防线 ——
        out = await call_tool(db, query="DELETE FROM product")
        check("拒绝 DELETE", "错误" in out, out[:120])

        out = await call_tool(db, query="SELECT 1; INSERT INTO product VALUES (1)")
        check("拒绝多语句拼接", "错误" in out, out[:120])


async def test_sandbox() -> None:
    """python 沙箱：越权要被拦，正常绘图不能受影响。"""
    print("\n[离线] Python 沙箱")
    client = MultiServerMCPClient(mcp_server_config)
    async with client.session("python_chart_mcp") as session:
        run = {t.name: t for t in await load_mcp_tools(session)}["run_python_script_tool"]

        out = await call_tool(run, script_content="import os\nos.system('echo pwned')")
        check("拦截 import os", "错误" in out or "拦截" in out, out[:120])

        out = await call_tool(
            run,
            script_content=(
                "print(open(r'D:/分享文件/推荐的项目/ChatBI数据分析助手/"
                "langGraph_agent/smart_data_analysis_assistant/.env').read()[:50])"
            ),
        )
        check("拦截读取项目 .env", "错误" in out or "拦截" in out, out[:120])

        out = await call_tool(
            run,
            script_content=(
                "import matplotlib\nmatplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "plt.rcParams['font.sans-serif']=['SimHei']\n"
                "plt.figure(); plt.bar(['1月','2月'],[3,5]); plt.title('销量')\n"
                "plt.savefig('t.png', dpi=60)\nprint('ok')\n"
            ),
        )
        check("正常绘图仍然可用（不能被沙箱误伤）",
              ".png" in out and "错误" not in out, out[:160])


async def test_graph_and_routing() -> None:
    """图能构建、拓扑没变、分流归一化正确。"""
    print("\n[离线] 图构建与分流路由")
    expected_nodes = {
        "call_python_coder", "python_run_tool_node", "call_identify_intention",
        "identify_intention_tool_node", "call_list_tables", "list_tables_node",
        "data_analysis_agent",
    }
    try:
        async with make_graph() as graph:
            g = graph.get_graph()
            nodes = set(g.nodes)
            # 边也要验：只验节点集合是拦不住"某条分支接错了线"的。
            # 实测教训（见 03-踩坑记录.md 故事 11）：python 分支曾经是
            #   call_python_coder → python_run_tool_node → END
            # 节点一个不缺，但这行 END 让整条分支永远不产出回答 ——
            # 脚本跑通、用户拿到空白，而这项"拓扑检查"全绿放行。
            raw_edges = g.edges
            edges = {
                (e.source, e.target) if hasattr(e, "source") else (e[0], e[1])
                for e in raw_edges
            }
        check("图构建成功", True)
        check("节点集合符合预期", expected_nodes <= nodes,
              f"缺少: {sorted(expected_nodes - nodes)}")
        # python 分支必须是循环：工具执行完回到 LLM，由 LLM 给出面向用户的回答。
        # 写成 END 就等于"算完了没人说话"。
        check("python 分支执行后回到 LLM（而不是直接结束）",
              ("python_run_tool_node", "call_python_coder") in edges,
              f"python_run_tool_node 的出边: "
              f"{sorted(t for s, t in edges if s == 'python_run_tool_node')}")
        # 反向：确认它没有同时挂着一条到 END 的边（两条并存行为取决于条件函数，容易看错）
        check("python_run_tool_node 没有直连结束节点",
              ("python_run_tool_node", "__end__") not in edges)
    except BaseException as exc:  # noqa: BLE001
        leaf = exc
        while hasattr(leaf, "exceptions"):
            leaf = leaf.exceptions[0]
        check("图构建成功", False, f"{type(leaf).__name__}: {leaf}")

    # 分流标签的归一化：模型多一个空格/标点/书名号都要能匹配上
    cases = [
        ("纯python编码", "call_python_coder"),
        ("纯Python编码。", "call_python_coder"),
        ("  纯 python 编码  ", "call_python_coder"),
        ("业务数据查询分析", "call_list_tables"),
        ("【业务数据查询分析】", "call_list_tables"),
        ("业务数据查询分析。", "call_list_tables"),
        ("好的，我来帮你查询", "__end__"),   # 自由对话 -> 结束本轮
        ("", "__end__"),
    ]
    for text, expected in cases:
        state = {"messages": [ToolMessage(content=text, tool_call_id="t")]}
        check(f"分流归一化 {text!r}", should_continue_ywfl(state) == expected)

    # 结构化 content（MCP 工具的真实返回形态）也要能拍平
    structured = [{"type": "text", "text": "业务数据查询分析"}]
    state = {"messages": [ToolMessage(content=structured, tool_call_id="t")]}
    check("分流识别结构化 content", should_continue_ywfl(state) == "call_list_tables")


# --------------------------------------------------------------------------- #
# 大模型层
# --------------------------------------------------------------------------- #
# 每个用例声明"回复里必须出现什么"和"绝不能出现什么"。
# 不去比对完整字符串 —— 模型的措辞每次都不一样，那样只会天天误报。
LLM_CASES = [
    {
        "tag": "维度不存在时必须拒绝作答",
        "question": "查询运动类商品有多少",
        # 必须点出维度缺失（第一组任选其一），并指明缺的是"品类"类维度（第二组任选其一）
        "contain_any": [["不存在", "没有", "无法统计"], ["品类", "分类", "类别"]],
        # 绝不能给出一个品类数量 —— 尤其是它自己用商品名模糊匹配凑出来的那个 1
        "not_match": [
            r"运动类商品共有\s*\d+",
            r"共有\s*1\s*[款个种]",
            r"运动类商品.{0,6}\d+\s*[款个种]",
        ],
    },
    {
        "tag": "总数统计必须去重，且不能用 LIMIT 行数当总数",
        "question": "查询商品一共有多少种",
        "contain_any": [["196"]],           # 当前数据集下 product_name 去重后的真实值
        "not_match": [r"共\s*30\s*[条款个种]"],   # 30 是查询行数上限，不是总数
    },
    {
        "tag": "正常查询要复述筛选条件",
        "question": "查询健身手套价格是多少",
        "contain_any": [["没有", "不存在", "0 行"], ["product_name", "商品名称"]],
        "not_match": [],
    },
]


async def _ask(graph, question: str) -> tuple[str, list[str]]:
    """跑一轮问答，返回（最终回复, 子图内部各步 SQL 工具返回）。

    注意 subgraphs=True：data_analysis_agent 是子图，不加这个参数的话
    它内部的 db_sql_tool 调用**不会**出现在事件流里，出问题时完全看不到
    模型到底跑了什么 SQL。
    """
    reply, steps = None, []
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        subgraphs=True,
    ):
        event = chunk[1] if isinstance(chunk, tuple) else chunk
        msg = event.get("messages", [None])[-1]
        if isinstance(msg, ToolMessage) and ("行数据" in str(msg.content) or "错误" in str(msg.content)):
            steps.append(_flatten(msg.content))
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            reply = msg.content
    return reply or "", steps


async def test_llm_behavior() -> None:
    print("\n[大模型] 行为用例（会消耗 API 额度）")
    async with make_graph() as graph:
        for case in LLM_CASES:
            reply, steps = await _ask(graph, case["question"])
            print(f"\n  ── {case['tag']}")
            print(f"     问: {case['question']}")
            print(f"     内部执行 SQL 步数: {len(steps)}")
            print(f"     答: {reply[:150].replace(chr(10), ' ')}...")

            for group in case["contain_any"]:
                check(f"{case['tag']} / 含 {group[0]}…", any(k in reply for k in group))

            for pattern in case["not_match"]:
                hit = re.search(pattern, reply)
                check(f"{case['tag']} / 不出现 {pattern}", hit is None,
                      f"命中了: {hit.group(0)!r}" if hit else "")


# --------------------------------------------------------------------------- #
async def main() -> int:
    parser = argparse.ArgumentParser(description="ChatBI 回归测试")
    parser.add_argument("--with-llm", action="store_true",
                        help="额外运行需要大模型的用例（会消耗 API 额度）")
    args = parser.parse_args()

    print("=" * 70)
    print("ChatBI 回归测试")
    print("=" * 70)

    await test_mcp_inventory()
    await test_sql_tool()
    await test_sandbox()
    await test_graph_and_routing()

    if args.with_llm:
        await test_llm_behavior()
    else:
        print("\n[大模型] 已跳过（加 --with-llm 运行）")

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [(n, d) for n, ok, d in _RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"结果: {passed}/{len(_RESULTS)} 通过")
    if failed:
        print("\n失败项:")
        for name, detail in failed:
            print(f"  - {name}" + (f"    {detail}" if detail else ""))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
