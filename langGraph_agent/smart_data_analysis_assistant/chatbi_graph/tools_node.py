"""
【保留作参考，主干不再使用】本地 SQLDatabaseToolkit 路线。

背景：这是"在 LangGraph 进程里直接连数据库、用 langchain_community 的 SQL 工具"
的做法。当前主干已经改成通过 search_db_mcp（9004）暴露 list_tables_tool / db_sql_tool，
图里也不再使用 call_get_schema / get_schema_node（build_graph.py 从未注册过这两个节点），
所以这里只保留下面的构建函数，方便对照学习两种实现方式。

⚠️ 原来的写法有两个坑，已修掉：
  1. 模块 import 时就 `SQLDatabase.from_uri(...)` + `SQLDatabaseToolkit(...)` ——
     "只想 import 一段提示词"也得有活的数据库，且失败即 import 失败；
     现在改成调用函数时才连接（lazy）。
  2. 第二个连接串用的是裸 `postgresql://`，SQLAlchemy 2.1 起默认走 psycopg3，
     没装 psycopg 就 ModuleNotFoundError（本项目实际踩过这个坑），
     现在显式写 `postgresql+psycopg2://`。

提示词已经挪到 prompts.py（纯数据、零副作用）。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def build_local_db(include_tables=None, sample_rows_in_table_info=3):
    """按需创建本地数据库连接对象。

    :param include_tables: 只包含指定表；None 表示全部
    """
    from langchain_community.utilities import SQLDatabase

    user = os.getenv("user")
    password = os.getenv("password")
    db_host = os.getenv("db_host")
    db_port = os.getenv("db_port", 5432)
    dbname = os.getenv("dbname")
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{user}:{password}@{db_host}:{db_port}/{dbname}",
        include_tables=include_tables,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )
    print("成功连接到数据库! 可用表:", db.get_usable_table_names())
    return db


def build_local_sql_toolkit(db, llm):
    """按需构建 SQLDatabaseToolkit（含 sql_db_query / sql_db_schema / sql_db_list_tables 等）。"""
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit, toolkit.get_tools()


if __name__ == "__main__":
    # 手动实验用：python tools_node.py
    from langgraph.prebuilt import ToolNode

    from my_llm import llm
    from prompts import generate_query_system_prompt  # noqa: F401  （对照用）

    _db = build_local_db()
    _toolkit, _tools = build_local_sql_toolkit(_db, llm)
    _schema_tool = next(t for t in _tools if t.name == "sql_db_schema")
    _schema_node = ToolNode([_schema_tool], name="get_schema_node")
    print("本地 SQL 工具:", [t.name for t in _tools])
