import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# 相对本文件定位 project_data/，换目录不用改
DATA_DIR = Path(__file__).resolve().parent / "project_data"

# 数据库配置统一走 .env（原来把连接串连同密码明文写在这里）
load_dotenv(Path(__file__).resolve().parent / "smart_data_analysis_assistant" / ".env")
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('user')}:{os.getenv('password')}"
    f"@{os.getenv('db_host')}:{os.getenv('db_port')}/{os.getenv('dbname')}"
)

files = {
    "product":      DATA_DIR / "商品表.xlsx",
    "user":         DATA_DIR / "用户表.xlsx",
    "sales":        DATA_DIR / "销量表.xlsx",
    "user_active":  DATA_DIR / "用户活跃表.xlsx",
}
for table, path in files.items():
    df = pd.read_excel(path)
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"导入 {table}: {len(df)} 行")