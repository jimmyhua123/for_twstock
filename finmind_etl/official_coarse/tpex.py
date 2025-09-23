from __future__ import annotations
import pandas as pd
from .utils import Http, Cache, parse_date_auto

BASE = "https://www.tpex.org.tw"
HEADERS = {"Referer": "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/"}

def fetch_stock_day_month_csv(stock_id: str, roc_yyy: int, mm: int, sleep_ms: int = 250) -> pd.DataFrame:
    url = f"{BASE}/web/stock/aftertrading/daily_trading_info/st43_result.php"
    params = {"l":"zh-tw","d":f"{roc_yyy:03d}/{mm:02d}","stkno": stock_id}
    key = f"TPEX_ST43::{stock_id}::{roc_yyy:03d}{mm:02d}"
    c = Cache(); hit = c.load(key)
    if hit is not None: return hit
    r = Http().get(url, params=params, headers=HEADERS, sleep_ms=sleep_ms)
    try:
        tables = pd.read_html(r.text, flavor="bs4")
    except Exception:
        tables = []
    if not tables:
        return pd.DataFrame(columns=["date","stock_id","open","high","low","close","volume"])
    t = tables[0]
    cols = {c:str(c) for c in t.columns}
    t.columns = list(cols.values())
    def to_num(x):
        try: return float(str(x).replace(",",""))
        except: return None
    out=[]
    for _, row in t.iterrows():
        d = parse_date_auto(row[t.columns[0]])
        if pd.isna(d): continue
        out.append({
            "date": d.date(), "stock_id": stock_id,
            "open": to_num(row[t.columns[3]]),
            "high": to_num(row[t.columns[4]]),
            "low":  to_num(row[t.columns[5]]),
            "close":to_num(row[t.columns[6]]),
            "volume": int(float(str(row[t.columns[1]]).replace(",","") or 0)),
        })
    df = pd.DataFrame(out); c.save(key, df); return df

def fetch_inst_daily_placeholder(*args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(columns=["date","stock_id","inst_net"])
