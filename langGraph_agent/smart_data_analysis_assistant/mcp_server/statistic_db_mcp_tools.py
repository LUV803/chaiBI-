"""
TEXT2SQL 数据库查询 MCP:
  list_tables_tool: 获取数据表结构信息（含表和字段注释）
  db_sql_tool:      执行 SQL 查询并返回结果

⚠️ 关于写保护（重要）
原实现直接 `db.run_no_throw(query)`，而连接用的是 postgres 超级用户账号，
唯一防线只有 system prompt 里的"不要执行 DML"——提示词不是安全边界。
现在做了三层防护：
  1. SQL 文本层：只允许 SELECT/WITH/TABLE/VALUES/SHOW/EXPLAIN 开头，
     剥离注释与字符串字面量后扫描写类关键字（含 CTE 里藏 INSERT/DELETE 的情况），
     并拒绝多语句拼接
  2. 连接层：会话默认 default_transaction_read_only=on —— 由 PostgreSQL 自己拒绝写操作，
     即使文本层被绕过也写不进去（不依赖账号权限，不用改建库脚本）
  3. 资源层：statement_timeout 限制单条 SQL 执行时长；返回行数上限，避免刷爆上下文
生产环境仍建议再把连接账号换成只有 SELECT 权限的角色（双保险）。

运行：python statistic_db_mcp_tools.py   (SSE, 端口 9004)
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import FastMCP
from sqlalchemy import create_engine

sys.path.append(str(Path(__file__).resolve().parent))

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

db_host = os.getenv("db_host", "127.0.0.1")
db_port = os.getenv("db_port", 5432)
user = os.getenv("user", "postgres")
password = os.getenv("password", "123456")
dbname = os.getenv("dbname", "sales_chat")

STATEMENT_TIMEOUT_MS = int(os.getenv("SQL_STATEMENT_TIMEOUT_MS", "15000"))
MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))

mcp = FastMCP(name="search_db_mcp", instructions="数据库查询MCP", host="0.0.0.0", port=9004)

# 显式写 psycopg2 驱动：裸 postgresql:// 在 SQLAlchemy 2.1 起默认走 psycopg3，
# 换个环境就会 ModuleNotFoundError（本项目踩过这个坑）
DATABASE_URI = f"postgresql+psycopg2://{user}:{password}@{db_host}:{db_port}/{dbname}"
engine = create_engine(
    DATABASE_URI,
    pool_pre_ping=True,
    connect_args={
        "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"
                   f" -c default_transaction_read_only=on"
    },
)


# --------------------------------------------------------------------------- #
# SQL 只读校验
# --------------------------------------------------------------------------- #
ALLOWED_FIRST_KEYWORDS = {"select", "with", "table", "values", "show", "explain"}
FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "grant",
    "revoke", "copy", "vacuum", "reindex", "refresh", "comment", "call", "do",
    "merge", "into", "lock", "cluster", "security", "reassign", "discard",
    "listen", "notify", "prepare", "deallocate", "execute", "import", "begin",
    "commit", "rollback", "savepoint",
}

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\$\$.*?\$\$", re.S)
_IDENTIFIER = re.compile(r'"(?:[^"]|"")*"')
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _strip_noise(sql: str) -> str:
    """剥离注释、字符串字面量、双引号标识符，只留下可判定的 SQL 结构。"""
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    sql = _STRING_LITERAL.sub("''", sql)
    sql = _IDENTIFIER.sub('""', sql)
    return sql


def check_read_only(query: str) -> str | None:
    """校验通过返回 None，否则返回拒绝原因。"""
    if not query or not query.strip():
        return "SQL 语句为空"

    stripped = _strip_noise(query).strip()
    # 多语句拼接：去掉空段后只剩一段才允许（结尾多一个分号是允许的）
    parts = [p for p in stripped.split(";") if p.strip()]
    if len(parts) > 1:
        return "不允许一次提交多条 SQL 语句"
    body = parts[0].strip() if parts else ""
    if not body:
        return "SQL 语句为空"

    words = [w.lower() for w in _WORD.findall(body)]
    if not words:
        return "SQL 语句无法识别"
    if words[0] not in ALLOWED_FIRST_KEYWORDS:
        return f"只允许执行只读查询，不允许以 {words[0].upper()} 开头的语句"

    hit = FORBIDDEN_KEYWORDS.intersection(words)
    if hit:
        return f"只允许执行只读查询，语句中不允许出现: {', '.join(sorted(hit)).upper()}"

    # PostgreSQL 事务级只读兜底提示（真正拦截在连接层）
    return None


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def get_table_comments():
    """获取所有表及其字段的注释信息（只读，走 information_schema）"""
    query = """
    SELECT
        t.table_name,
        obj_description(('"' || t.table_schema || '"."' || t.table_name || '"')::regclass, 'pg_class') as table_comment,
        c.column_name,
        c.data_type,
        pg_catalog.col_description(('"' || t.table_schema || '"."' || t.table_name || '"')::regclass::oid, c.ordinal_position) as column_comment
    FROM
        information_schema.tables t
    JOIN
        information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
    WHERE
        t.table_schema NOT IN ('pg_catalog', 'information_schema')
        AND t.table_type = 'BASE TABLE'
    ORDER BY
        t.table_name, c.ordinal_position
    """
    with engine.connect() as conn:
        # 用 mappings() 拿到字典型结果行；不再依赖 db._execute 这类私有 API
        rows = conn.exec_driver_sql(query).mappings().all()

    tables = {}
    for row in rows:
        table_name = row["table_name"]
        if table_name not in tables:
            tables[table_name] = {
                "comment": row["table_comment"] or "无表注释",
                "columns": [],
            }
        tables[table_name]["columns"].append({
            "name": row["column_name"],
            "type": row["data_type"],
            "comment": row["column_comment"] or "无列注释",
        })
    return tables


@mcp.tool()
async def list_tables_tool() -> str:
    """
    输入是个空字符串, 返回数据库中的所有表及其结构信息，包括表和字段的注释
    :return: 数据库中的所有表及其结构信息的格式化字符串
    """
    try:
        tables = get_table_comments()
    except Exception as exc:
        return f"获取表结构失败: {type(exc).__name__}: {exc}"

    blocks = []
    for table_name, info in tables.items():
        header = f"表名: {table_name}  [注释: {info['comment']}]"
        columns = [
            f"  - {c['name']} ({c['type']})  [注释: {c['comment']}]"
            for c in info["columns"]
        ]
        blocks.append(header + "\n" + "\n".join(columns))
    return "\n\n".join(blocks) if blocks else "数据库中没有任何表"


@mcp.tool()
def db_sql_tool(query: str) -> str:
    """
    执行SQL查询并返回结果。如果查询不正确，将返回错误信息;如果返回错误，请重写查询语句，检查后重试。
    :param query: 非空的要执行的SQL查询语句（只允许 SELECT 等只读语句）
    :return:str: 查询结果（0 行也会明确说明是"无数据"而不是错误）或错误信息
    """
    reject = check_read_only(query)
    if reject:
        return f"错误: {reject}"

    try:
        with engine.connect() as conn:
            # 走原生 DBAPI 游标，绕开 SQLAlchemy 的参数处理层。
            #
            # 为什么不用 conn.exec_driver_sql：
            #   psycopg2 的 paramstyle 是 pyformat，SQL 文本里的 `%` 会被当成参数
            #   占位符，于是 `WHERE name LIKE '%关键字%'` 直接抛
            #     TypeError: immutabledict is not a sequence
            #   而 LIKE 模糊匹配恰恰是 Text2SQL 最常用的写法之一。
            #   （把 % 转义成 %% 也能过，但规则更绕，不如直接绕开这层。）
            # 选原生游标还顺带保住了 exec_driver_sql 原本的好处：
            #   `::` 类型转换、双引号包裹的中文标识符都不会被误当作绑定参数。
            # 只读 / statement_timeout 由连接层 connect_args 的 options 保证，
            # 这条路同样生效（DELETE 依旧会被 PostgreSQL 拒绝）。
            cursor = conn.connection.cursor()
            try:
                cursor.execute(query)
                columns = [d[0] for d in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(MAX_ROWS) if columns else []
            finally:
                cursor.close()
    except Exception as exc:
        # 把数据库原始报错回传给模型，它才能据此改写 SQL（这是 Text2SQL 自愈的关键）
        return f"错误: SQL 执行失败 -> {type(exc).__name__}: {exc}"

    if not columns:
        return "查询执行成功，该语句不返回结果集"

    # 0 行是合法结果，不能当成错误（旧实现会误报成"查询失败"，
    # 导致模型反复"修正"一条本来就是对的 SQL）
    if not rows:
        return f"查询执行成功，共 0 行数据（字段: {', '.join(columns)}）"

    body = str([tuple(r) for r in rows])
    suffix = ""
    if len(rows) >= MAX_ROWS:
        suffix = f"\n（结果已截断，最多返回 {MAX_ROWS} 行；如需更精确的结果请增加筛选条件或聚合）"
    return f"字段: {', '.join(columns)}\n查到 {len(rows)} 行数据: {body}{suffix}"


if __name__ == "__main__":
    # 以标准 sse方式运行 MCP 服务器
    mcp.run(transport="sse")

# nohup python statistic_db_mcp_tools.py &
