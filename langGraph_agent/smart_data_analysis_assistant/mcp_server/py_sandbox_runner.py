"""
Python 代码执行沙箱 —— 子进程侧 runner。

由 python_chart_mcp.run_python_script_tool 以独立子进程方式拉起：
    python py_sandbox_runner.py <user_script_path> <workdir> <config_json>

用户代码的 stdout 会被重定向捕获，结果通过 stdout 的标记行回传给父进程：
    __PY_RESULT__:{"ok":true,"stdout":"...","variables":{...},"images":[...],"error":null}

加固层次（由外到内）：
  1. 独立子进程：用户代码崩溃/死循环不会影响 MCP 服务进程；父进程负责超时强杀
  2. 精简环境变量 + cwd / MPLCONFIGDIR 限定到工作目录
  3. sys.addaudithook 审计钩子（真正的执行层拦截，装上后无法卸载）：
     - 写 / 删除 / 改名：只允许本次执行的工作目录与 matplotlib 缓存目录
     - 读：默认放行，但禁止项目目录与用户主目录（.env、密钥、业务数据所在处）
     - 阻断进程创建、系统命令、网络、动态库加载
  4. 受限 __builtins__：用户脚本自己的 import 走白名单，__import__/eval/exec/compile 等被移除

⚠️ 这是"加固"，不是容器级隔离：仍然是同一个 OS 用户、同一个文件系统命名空间，
   也无法防御 CPython 解释器本身的漏洞。生产环境应把本 runner 放进容器执行，例如：
     docker run --rm --network=none --read-only --cpus=1 -m 512m -v <workdir>:/sandbox python:3.11 ...
   （最小权限的 Docker 隔离 + 只读数据库账号，才是这套系统的正确终点。）
"""
import builtins
import contextlib
import io
import json
import os
import sys
import traceback

MAX_CAPTURE = 200_000  # 捕获的用户 stdout 上限（字符），防止刷爆内存
MAX_VARS = 20          # 回传的新变量个数上限
MAX_VAR_LEN = 200      # 单个变量字符串化后的长度上限

RESULT_MARKER = "__PY_RESULT__:"


# --------------------------------------------------------------------------- #
# 路径工具
# --------------------------------------------------------------------------- #
def _norm(path):
    """把任意路径写法归一化为绝对真实路径；失败返回 None。"""
    try:
        if isinstance(path, int):
            return None
        return os.path.realpath(os.path.abspath(os.fspath(path)))
    except Exception:
        return None


def _under(path, roots):
    if path is None:
        return False
    for root in roots:
        if not root:
            continue
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:  # 不同盘符（Windows）
            continue
    return False


# --------------------------------------------------------------------------- #
# 审计钩子
# --------------------------------------------------------------------------- #
# 直接拒绝的审计事件：进程创建、系统命令、网络、动态库
# 注意：不要拦 os.putenv/os.unsetenv —— matplotlib 等库内部会写环境变量，
#       拦了就等于把绘图功能一起废掉；子进程本来就是一次性的，改环境变量无害。
DENY_EVENTS = frozenset({
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.kill",
    "os.startfile",
    "subprocess.Popen",
    # ctypes：只拦"真正发起原生调用"的事件。dlopen/dlsym 必须放行 ——
    # winreg 等标准库内部要用 ctypes 读 kernel32 的 GetLastError，
    # 而 import ctypes 已经在导入层被禁止，用户拿不到自己的 ctypes 对象。
    "ctypes.call_function",
    "socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.sendto", "socket.sendmsg",
    "urllib.Request", "http.client.connect",
    "ftplib.connect", "smtplib.connect", "telnetlib.Telnet",
    "webbrowser.open",
    # 注册表：只读放行（matplotlib 靠它枚举已安装字体），写一律拦
    "winreg.CreateKey", "winreg.CreateKeyEx", "winreg.SetValue",
    "winreg.DeleteKey", "winreg.DeleteValue", "winreg.SaveKey", "winreg.LoadKey",
    "shutil.rmtree", "shutil.move", "shutil.copyfile", "shutil.copymode",
    "sys.addaudithook",
})

# 需要按路径判定的写类事件：事件名 -> 参数里代表路径的下标
WRITE_EVENTS = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.unlink": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.link": (0, 1),
    "os.symlink": (0, 1),
}


def _make_audit_hook(write_roots, deny_read_roots):
    """生成审计钩子。write_roots 白名单，deny_read_roots 黑名单。"""

    def _hook(event, args):
        if event in DENY_EVENTS:
            raise PermissionError(f"沙箱拦截：禁止调用 {event}")

        if event == "open":
            # args = (path, mode, flags)
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if isinstance(path, int):
                return  # 已经打开的 fd，不重复判定
            target = _norm(path)
            if target is None:
                return
            mode = mode or ""
            writing = any(c in mode for c in "wax+") or bool(
                isinstance(flags, int) and (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            )
            if writing:
                if not _under(target, write_roots):
                    raise PermissionError(f"沙箱拦截：不允许写入工作目录之外的文件 {target}")
            else:
                if _under(target, deny_read_roots) and not _under(target, write_roots):
                    raise PermissionError(f"沙箱拦截：不允许读取该路径 {target}")
            return

        idxs = WRITE_EVENTS.get(event)
        if idxs:
            for i in idxs:
                if i >= len(args):
                    continue
                target = _norm(args[i])
                if target is not None and not _under(target, write_roots):
                    raise PermissionError(f"沙箱拦截：{event} 越权路径 {target}")

    return _hook


# --------------------------------------------------------------------------- #
# 受限 builtins
# --------------------------------------------------------------------------- #
IMPORT_ALLOWLIST = {
    "matplotlib", "numpy", "pandas", "math", "cmath", "statistics", "random",
    "json", "re", "datetime", "decimal", "fractions", "itertools", "functools",
    "collections", "operator", "string", "textwrap", "typing", "warnings",
    "calendar", "uuid", "heapq", "bisect", "copy", "enum", "dataclasses",
}
FUNC_REMOVED = {"eval", "exec", "compile", "input", "breakpoint", "exit", "quit",
                "globals", "locals", "vars", "help", "open"}


def _build_safe_builtins():
    """构造用户脚本可见的 builtins。

    注意：这里移除的 `open`/`globals` 等只影响**用户脚本自身**的顶层命名空间解析；
    被 import 进来的库（matplotlib/numpy）内部仍使用真正的 builtins，
    所以文件读写的最终防线是上面的审计钩子，而不是这里。
    `open` 被移除后，用户脚本无法直接读写文件；绘图落盘由 runner 统一负责。
    """
    safe = {k: v for k, v in vars(builtins).items() if k not in FUNC_REMOVED and k != "__import__"}
    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level != 0 or root not in IMPORT_ALLOWLIST:
            raise PermissionError(f"沙箱拦截：不允许导入模块 {name}")
        return real_import(name, globals, locals, fromlist, level)

    safe["__import__"] = _guarded_import
    return safe


# --------------------------------------------------------------------------- #
# 输出捕获
# --------------------------------------------------------------------------- #
class _LimitedWriter(io.TextIOBase):
    """限制捕获体积；超限直接中断执行，避免 print 刷爆内存。"""

    def __init__(self, limit):
        self._buf = io.StringIO()
        self._limit = limit

    def write(self, s):
        if self._buf.tell() + len(s) > self._limit:
            raise RuntimeError("输出内容过大，已中断执行")
        return self._buf.write(s)

    def flush(self):
        pass

    def getvalue(self):
        return self._buf.getvalue()


# --------------------------------------------------------------------------- #
# 绘图支持
# --------------------------------------------------------------------------- #
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC",
    "Source Han Sans CN", "Droid Sans Fallback",
)


def _setup_matplotlib():
    """预导入 matplotlib（Agg 后端，不会弹窗阻塞），并尽量挂上中文字体。

    返回 (环境变量字典, 失败原因)。失败原因会回传给父进程，便于排查
    "脚本没报错但也没出图"这类问题，而不是静默吞掉。
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {}, f"matplotlib 预加载失败: {type(exc).__name__}: {exc}"
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for family in CJK_FONT_CANDIDATES:
            if family in available:
                matplotlib.rcParams["font.sans-serif"] = [family] + list(
                    matplotlib.rcParams.get("font.sans-serif", [])
                )
                break
        matplotlib.rcParams["axes.unicode_minus"] = False  # 负号不显示成方块
    except Exception:
        pass

    env = {"plt": plt}
    try:
        import numpy as np
        env["np"] = np
    except Exception:
        pass
    try:
        import pandas as pd
        env["pd"] = pd
    except Exception:
        pass
    return env, None


def _current_pyplot():
    """取当前进程真正的 pyplot 模块。

    不能只依赖预加载时拿到的那份引用：用户脚本自己 `import matplotlib.pyplot`
    同样可行，此时图是画在 sys.modules 里那份模块上的。
    """
    module = sys.modules.get("matplotlib.pyplot")
    if module is None:
        return None
    try:
        module.get_fignums()
    except Exception:
        return None
    return module


def _save_new_images(plt, workdir, image_dir):
    """把脚本产生的图片落到 image_dir，返回 (路径列表, 失败原因列表)。"""
    import uuid as _uuid
    saved, errors = [], []
    for num in plt.get_fignums():
        try:
            fig = plt.figure(num)
            path = os.path.join(image_dir, f"figure_{_uuid.uuid4().hex}.png")
            fig.savefig(path, format="png", bbox_inches="tight")
            saved.append(path)
        except Exception as exc:
            errors.append(f"图 {num} 保存失败: {type(exc).__name__}: {exc}")
    try:
        plt.close("all")
    except Exception:
        pass
    # 脚本自己 savefig 到 cwd（= workdir）的情况，一并收走
    try:
        for name in sorted(os.listdir(workdir)):
            if name.lower().endswith(".png"):
                src = os.path.join(workdir, name)
                if src in saved:
                    continue
                dst = os.path.join(image_dir, f"figure_{os.path.splitext(name)[0][-12:]}.png")
                os.replace(src, dst)
                saved.append(dst)
    except Exception as exc:
        errors.append(f"回收工作目录图片失败: {type(exc).__name__}: {exc}")
    return saved, errors


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _emit(payload):
    sys.__stdout__.write(RESULT_MARKER + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main():
    script_path, workdir, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    image_dir = os.path.realpath(config["image_dir"])
    workdir = os.path.realpath(workdir)
    # 写白名单只有三处：本次工作目录、matplotlib 缓存目录、图片输出目录
    write_roots = [workdir, os.path.realpath(config["mpl_config_dir"]), image_dir]
    deny_read_roots = [r for r in (os.path.realpath(p) for p in config.get("deny_read", [])) if r]

    # 把临时目录也重定向进工作目录：库（matplotlib/pandas 等）写临时文件时
    # 落在白名单内，否则会被"只允许写工作目录"的规则误伤。
    tmp_dir = os.path.join(workdir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    os.chdir(workdir)
    os.environ["MPLCONFIGDIR"] = config["mpl_config_dir"]
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = tmp_dir
    os.environ["HOME"] = workdir          # 让库把家目录也当成工作目录
    os.environ["USERPROFILE"] = workdir

    sys.addaudithook(_make_audit_hook(write_roots, deny_read_roots))

    with open(script_path, encoding="utf-8") as fh:
        source = fh.read()

    env = {"__name__": "__main__", "__builtins__": _build_safe_builtins()}
    preset, setup_error = _setup_matplotlib()
    env.update(preset)
    plt = None

    captured = _LimitedWriter(MAX_CAPTURE)
    before = set(env)
    payload = {"ok": False, "stdout": "", "variables": {}, "images": [], "error": None}

    try:
        with contextlib.redirect_stdout(captured):
            try:
                value = eval(compile(source, "<user_script>", "eval"), env)
                if value is not None:
                    print(value)
            except SyntaxError:
                exec(compile(source, "<user_script>", "exec"), env)
        payload["ok"] = True
    except BaseException as exc:  # noqa: BLE001 —— 用户代码什么都可能抛，必须全兜住
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        payload["error"] = err
        tb = traceback.format_exc(limit=2)
        payload["traceback"] = tb.strip().splitlines()[-3:]

    img_errors = []
    plt = _current_pyplot()
    if plt is not None:
        try:
            payload["images"], img_errors = _save_new_images(plt, workdir, image_dir)
        except Exception as exc:
            payload["images"] = []
            img_errors = [f"图片收集失败: {type(exc).__name__}: {exc}"]

    notes = [n for n in (setup_error, *img_errors) if n]
    if not notes and not payload["images"] and "matplotlib" in source:
        notes.append("脚本引用了 matplotlib 但没有产生任何图形（是否漏了 plt.show()/savefig？）")
    if notes:
        payload["notes"] = "；".join(notes)

    created = [k for k in env if k not in before and not k.startswith("_")]
    for name in created[:MAX_VARS]:
        try:
            text = str(env[name])
        except Exception:
            text = f"<无法字符串化的对象: {type(env[name]).__name__}>"
        payload["variables"][name] = text[:MAX_VAR_LEN]

    payload["stdout"] = captured.getvalue()
    _emit(payload)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # runner 自身异常也要以标记行回传，父进程才能识别
        _emit({
            "ok": False, "stdout": "", "variables": {}, "images": [],
            "error": f"沙箱运行器异常: {type(exc).__name__}: {exc}",
        })
