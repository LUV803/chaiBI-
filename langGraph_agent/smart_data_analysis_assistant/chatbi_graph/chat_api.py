"""
本文件用于对项目进行API封装部署。

修复记录：
  1. 原来每个请求都调一次 `make_graph()`：重连 4 个 MCP、重新 list_tools、
     还额外请求 mermaid.ink 画一次流程图。现在用 FastAPI lifespan 在启动时
     建立一次并复用（图本身无 checkpointer，是每请求独立的，可安全共享）。
  2. 原来在流结束后可能 return None（例如分流标签没匹配上、整图直接走到 END），
     FastAPI 会返回 200 + null，前端看到空白。现在统一返回兜底文案。
  3. CORS 原来同时开 allow_origins=["*"] 和 allow_credentials=True，
     这个组合浏览器会直接拒绝；改为由环境变量白名单控制，默认放开但关闭凭证。
  4. 图片只返回本地路径、前端取不到；现在把 /images 静态目录挂出来，
     并在响应里额外给出 images 列表。

环境变量：
  CHATBI_API_KEY     设置后要求请求头带 X-API-Key，留空则不校验（默认）
  CHATBI_CORS_ORIGINS  逗号分隔的允许来源，默认 *
"""
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))

from build_graph import make_graph  # noqa: E402

# 项目唯一的 .env 在上一级目录（smart_data_analysis_assistant/.env）
# 显式定位：不依赖 cwd，也不用 find_dotenv() 的向上查找
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FALLBACK_REPLY ="抱歉，这一轮我没能得出结果（可能是问题描述不够明确，或分流标签解析异常），请换个说法再试一次。"
API_KEY = os.getenv("CHATBI_API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("CHATBI_CORS_ORIGINS", "*").split(",") if o.strip()]

# 图片输出目录：与 python_chart_mcp 的 IMAGE_STORAGE_DIR 保持一致
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s\"'，。；;]*?\.png")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建一次图（顺带把 4 个 MCP 会话保持住），关闭时自动释放。"""
    async with make_graph() as graph:
        app.state.graph = graph
        print("LangGraph 已就绪，服务开始接收请求")
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # 与 allow_origins=["*"] 不能同时为 True
    allow_methods=["*"],
    allow_headers=["*"],
)
# 生成的图表通过 /images/xxx.png 直接访问，前端不用去读后端本地路径
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")


class UserInput(BaseModel):
    user_id: str
    message: str
    history: list[dict] = []


@app.post("/chatbi_service")
async def chatbi_server(user_input: UserInput, request: Request, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效的 X-API-Key")

    user_id = user_input.user_id
    user_message = user_input.message
    history = list(user_input.history)  # 拷贝，避免就地修改调用方传入的列表
    history.append({"role": "user", "content": user_message})
    print(f"用户Id:{user_id},本轮输入:{user_message},历史记录:{history}")

    graph = request.app.state.graph
    reply = None
    async for event in graph.astream({"messages": history}, stream_mode="values"):
        messages = event.get("messages")
        if not messages:
            continue
        message = messages[-1]
        message.pretty_print()
        # 只要最后一条是"有内容且无工具调用"的 AIMessage，就是本轮给用户的回复
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            reply = message.content

    if reply is None:
        print("本轮未产生可展示的回复，返回兜底文案")
        return {"message": FALLBACK_REPLY, "images": []}

    print("本轮回复:", reply)
    images = [p.replace("\\", "/") for p in IMAGE_PATH_RE.findall(reply)]
    return {"message": reply, "images": images}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9008)

# nohup uvicorn chat_api:app --host 0.0.0.0 --port 9008 --workers 1 &
