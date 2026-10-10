# inspect_schema.py —— 读取4个数据表的列结构和样例，输出到文件避免控制台乱码
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "project_data"
files = {
    "product":      BASE / "商品表.xlsx",
    "user":         BASE / "用户表.xlsx",
    "sales":        BASE / "销量表.xlsx",
    "user_active":  BASE / "用户活跃表.xlsx",
}

lines = []
for name, path in files.items():
    df = pd.read_excel(path)
    lines.append(f"===== {name}（{len(df)} 行） =====")
    lines.append("列名: " + ", ".join(str(c) for c in df.columns))
    lines.append("样例(前3行):")
    lines.append(df.head(3).to_string())
    lines.append("")

out = Path(__file__).resolve().parent / "schema_report.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("done ->", out)
