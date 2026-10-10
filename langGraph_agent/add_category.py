# add_category.py —— 给 product 表加 category 列，并用 DeepSeek 对商品归类后写回
# 运行前确认：chatbi 环境已装 pandas / sqlalchemy / psycopg2-binary / openai / python-dotenv
import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from openai import OpenAI

# 1. 从 smart_data_analysis_assistant/.env 读取数据库配置和 DeepSeek key
#    相对本文件定位，换目录不用改（原来是写死的 D:\分享文件\...）
ENV_PATH = Path(__file__).resolve().parent / "smart_data_analysis_assistant" / ".env"
load_dotenv(ENV_PATH)

# 2. 连接数据库
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('user')}:{os.getenv('password')}"
    f"@{os.getenv('db_host')}:{os.getenv('db_port')}/{os.getenv('dbname')}"
)

# 3. 给 product 表加 category 列（已存在则跳过）
with engine.begin() as conn:
    conn.exec_driver_sql("ALTER TABLE product ADD COLUMN IF NOT EXISTS category VARCHAR(20)")
print("[1/4] category 列已添加")

# 4. 读取全部商品（product_no 是唯一键，用于写回）
df = pd.read_sql("SELECT product_no, product_name FROM product", engine)
print(f"[2/4] 读取到 {len(df)} 个商品")

# 5. 用 DeepSeek 分批归类
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

CATEGORIES = "运动, 食品, 日用品, 其他"  # 类别集合，可自行调整

def classify_batch(names_batch):
    """让 LLM 把一批商品名归类，返回 {商品名: 类别} 字典"""
    prompt = f"""你是商品分类专家。请把下面每个商品分到以下类别之一（只能选一个）：
类别：{CATEGORIES}

商品列表：
{"\n".join(names_batch)}

只输出一个 JSON 对象，键为商品名、值为类别。例如：
{{"篮球":"运动", "大米":"食品", "洗衣液":"日用品"}}"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

# 分批调用，避免单次请求太长
mapping = {}
BATCH = 50
for i in range(0, len(df), BATCH):
    batch = df["product_name"][i:i + BATCH].tolist()
    for attempt in range(3):  # 最多重试3次
        try:
            mapping.update(classify_batch(batch))
            break
        except Exception as e:
            print(f"  批次 {i}-{i+BATCH} 失败，重试({attempt+1}/3): {e}")
print(f"[3/4] 已归类 {len(mapping)} 个商品")

# 6. 写回 category 列（未归类的默认"其他"）
updated = 0
with engine.begin() as conn:
    for _, row in df.iterrows():
        cat = mapping.get(row["product_name"], "其他")
        conn.exec_driver_sql(
            "UPDATE product SET category=%s WHERE product_no=%s",
            (cat, row["product_no"]),
        )
        updated += 1
print(f"[4/4] 已更新 {updated} 行。完成！")

# 7. 展示归类结果统计
stats = pd.read_sql("SELECT category, COUNT(*) AS cnt FROM product GROUP BY category", engine)
print(stats.to_string(index=False))
