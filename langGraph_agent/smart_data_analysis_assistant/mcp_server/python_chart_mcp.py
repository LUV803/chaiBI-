"""
Python 绘图 / 代码执行 MCP。

工具1 run_python_script_tool：执行一段 Python 脚本，回传文本结果或生成的图片路径。
工具2 translate_to_python_plot_script：把「绘图需求 + 数据描述」翻译成可执行的 matplotlib 代码。

⚠️ 关于执行安全（重要）
用户/模型给出的代码**不在本进程内执行**，而是交给 py_sandbox_runner.py 以独立子进程运行：
    - 父进程负责超时强杀（默认 20s，PY_SANDBOX_TIMEOUT 可调），死循环不会挂死 MCP 服务
    - 子进程用 sys.addaudithook 拦截进程创建 / 系统命令 / 网络 / 越权文件读写
    - 每次执行独立工作目录，图片统一落到输出目录，不再共享全局 matplotlib 状态
这解决了原实现中 `exec(script_content, globals().copy())` 带来的任意代码执行、
事件循环阻塞、以及并发请求共用 plt 全局状态三个问题。
但它仍是"加固"而非容器级隔离（同 OS 用户、同文件系统命名空间），
生产环境请把 runner 放进 Docker（--network=none --read-only --cpus/--memory 限额）里执行。
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import FastMCP

sys.path.append(str(Path(__file__).resolve().parent))
from public_function import LLM_replay  # noqa: E402

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MCP_NAME = "python_chart_mcp"
RUNNER_PATH = Path(__file__).resolve().parent / "py_sandbox_runner.py"

# 图片输出目录（与 chatbi_graph/output 保持一致，前端/文档引用的是这个目录）
IMAGE_STORAGE_DIR = Path(__file__).parent.parent / "chatbi_graph" / "output"
IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# matplotlib 字体缓存的持久目录：放在输出目录下，跨请求复用，避免每次重扫字体
MPL_CONFIG_DIR = IMAGE_STORAGE_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# 每次执行的临时工作目录（系统临时区，执行完即删）
SANDBOX_ROOT = Path(tempfile.gettempdir()) / "chatbi_py_sandbox"
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)

EXEC_TIMEOUT = float(os.getenv("PY_SANDBOX_TIMEOUT", "20"))

# 只读禁区：项目目录（含唯一的 .env、project_data/*.xlsx）与用户主目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DENY_READ_ROOTS = [str(PROJECT_ROOT), str(Path.home())]

mcp = FastMCP(name=MCP_NAME, instructions="Python绘图工具MCP", host="0.0.0.0", port=9002)


def _run_in_sandbox(script_content: str):
    """在子进程中执行用户代码（同步函数，由 asyncio.to_thread 调度，不阻塞事件循环）。"""
    workdir = Path(tempfile.mkdtemp(prefix="exec_", dir=str(SANDBOX_ROOT)))
    script_path = workdir / "user_script.py"
    config_path = workdir / "sandbox_config.json"
    try:
        script_path.write_text(script_content, encoding="utf-8")
        config_path.write_text(
            json.dumps({
                "image_dir": str(IMAGE_STORAGE_DIR),
                "mpl_config_dir": str(MPL_CONFIG_DIR),
                "deny_read": DENY_READ_ROOTS,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        # 精简环境变量：只保留解释器运行必需项，避免继承宿主的密钥等敏感变量
        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),   # Windows 下 python 启动必需
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        }
        proc = subprocess.run(
            [sys.executable, str(RUNNER_PATH), str(script_path), str(workdir), str(config_path)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workdir),
            env=child_env,
            timeout=EXEC_TIMEOUT,
        )
        return _parse_result(proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "variables": {}, "images": [],
                "error": f"代码执行超过 {EXEC_TIMEOUT:.0f} 秒已被强制终止（疑似死循环或长时间阻塞）"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_result(stdout: str, stderr: str, returncode: int):
    """从子进程 stdout 中解析标记行。"""
    for line in reversed(stdout.splitlines()):
        if line.startswith("__PY_RESULT__:"):
            try:
                return json.loads(line[len("__PY_RESULT__:"):])
            except json.JSONDecodeError:
                break
    detail = (stderr or stdout or "").strip().splitlines()
    tail = "\n".join(detail[-5:]) if detail else "无输出"
    return {"ok": False, "stdout": "", "variables": {}, "images": [],
            "error": f"沙箱子进程异常退出(returncode={returncode}):\n{tail}"}


def _format_result(result: dict) -> str:
    """把沙箱结果整理成给模型看的文本（保持与旧版一致的措辞，避免影响下游 prompt 行为）。"""
    lines = []
    if result.get("images"):
        for path in result["images"]:
            lines.append(f"根据您的要求,生成的统计图路径:{path}")
    if result.get("stdout", "").strip():
        lines.append("程序输出:\n" + result["stdout"].strip())
    if result.get("variables"):
        lines.append("代码执行产生的新变量:\n" + json.dumps(result["variables"], ensure_ascii=False))
    if result.get("notes"):
        lines.append("环境提示: " + result["notes"])
    if result.get("error"):
        lines.append("执行过程中发生错误: " + result["error"])

    if not lines:
        return "程序执行得到结果:\n代码执行完成，未产生新的变量或图片"
    if result.get("error"):
        return "\n".join(lines)
    return "程序执行得到结果:\n" + "\n".join(lines)


@mcp.tool()
async def run_python_script_tool(script_content: str) -> str:
    """
    执行用户提供的Python代码，并返回执行结果或生成的图片路径
    :param script_content: 需要执行的Python代码字符串
    :return: 代码运行的最终结果或生成的图片路径信息描述
    """
    if not isinstance(script_content, str) or not script_content.strip():
        return "程序执行得到结果:\n执行过程中发生错误: 脚本内容为空"
    # to_thread + subprocess timeout：既不阻塞事件循环，也能真正杀死卡死的子进程
    result = await asyncio.to_thread(_run_in_sandbox, script_content)
    return _format_result(result)


# 说明：为什么除了上面的执行工具，还要单独提供一个"生成绘图代码"的工具？
# 因为不同的用户诉求下模型的推理路径不同：
#   A. 直接绘图：模型倾向于找一个"能生成图片的工具"，而不是先写代码再执行
#   B. 查数 + 绘图：模型会先写 python 绘图代码，再执行——这时用 run_python_script_tool
# 所以提供一个"需求 -> python 绘图代码"的工具，能有效避免模型在 A 场景下卡住。
@mcp.tool()
async def translate_to_python_plot_script(graph_demand: str, data_desc: str) -> str:
    """
    根据绘图需求，将提供数据转成python matplotlib库写的绘图代码,如果当前数据不足以生成绘图代码请勿调用
    :param graph_demand:绘图的需求，如：绘制一张折线图,出具一张饼图
    :param data_desc:可以支持实现绘制统计图的数据描述
    :return:一段可以执行的python绘图代码
    """
    prompt = f"""你是一个python绘图代码生成大师,你善于根据当前用户提供的数据，依据用户的绘图需求生成可执行的python绘图代码，必须通过matplotlib库实现。
    注意：(1)若没有提供合适的数据完成绘图代码生成，请返回:我无法生成绘图代码 (2) 请勿生成需求要求外的代码，请勿生成非绘图代码。(3) 你的任务仅仅是生成可执行python代码，请勿做任何分析或解释。
    举例1:
    输入:
    用户需求:请根据销量数据生成一张销量折线图
    数据:12个月的销量数据[30,50,70,50,80,55,30,50,70,50,80,55]
    输出:import matplotlib.pyplot as plt
import numpy as np
sales_data = [30, 50, 70, 50, 80, 55, 30, 50, 70, 50, 80, 55]
months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.plot(months, sales_data, marker='o', color='#4CAF50', linewidth=2, markersize=8)
plt.title('2023年每月销量趋势', fontsize=16, pad=20)
plt.xlabel('月份', fontsize=12)
plt.ylabel('销量(件)', fontsize=12)
plt.ylim(0, max(sales_data) + 10)
for x, y in zip(months, sales_data):
    plt.text(x, y+2, str(y), ha='center', va='bottom', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
    举例2:
    输入:
    用户需求:请根据订单数据绘制一张每个产品的销售分布饼图
    数据:需要查询订单表的数据
    输出:我无法生成绘图代码


    输入:
    用户需求:{graph_demand}
    数据:{data_desc}
    输出:"""
    code_result = await LLM_replay(messages=prompt)
    if "无法生成" in code_result:
        return "数据不够，需要从数据库中查询获取更多数据"
    return code_result.strip()


if __name__ == "__main__":
    # 以标准 streamable-http方式运行 MCP 服务器
    mcp.run(transport="streamable-http")

# nohup python python_chart_mcp.py &
