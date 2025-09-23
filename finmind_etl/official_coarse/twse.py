from __future__ import annotations
import pandas as pd, json
from .utils import Http, Cache, parse_date_auto

BASE = "https://www.twse.com.tw"
HEADERS = {"Referer": "https://www.twse.com.tw/zh/trading/exchange/MI_INDEX.html"}

def _safe_json(r):
    try:
        return r.json()
    except Exception:
        return {"stat": "FAIL", "data": []}

def fetch_stock_day_month(stock_id: str, yyyymm: str, sleep_ms: int = 250) -> pd.DataFrame:
    url = f"{BASE}/exchangeReport/STOCK_DAY"
    params = {"response":"json","date":yyyymm+"01","stockNo":stock_id}
    key = f"TWSE_STOCK_DAY::{stock_id}::{yyyymm}"
    c = Cache(); hit = c.load(key)
    if hit is not None: return hit
    for _ in range(3):
        r = Http().get(url, params=params, headers=HEADERS, sleep_ms=sleep_ms)
        js = _safe_json(r)
        if js.get("stat") not in (None, "OK") and not js.get("data"):
            # 伺服器回覆失敗，稍後重試
            continue
        rows = []
        for row in js.get("data", []):
            try:
                d = parse_date_auto(row[0])
                rows.append({
                    "date": d.date(), "stock_id": stock_id,
                    "open": float(str(row[3]).replace(",","")),
                    "high": float(str(row[4]).replace(",","")),
                    "low":  float(str(row[5]).replace(",","")),
                    "close":float(str(row[6]).replace(",","")),
                    "volume": int(str(row[1]).replace(",","")),
                })
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows); c.save(key, df); return df
    # 最終保底：回空表但不丟例外，讓 pipeline 繼續
    return pd.DataFrame(columns=["date","stock_id","open","high","low","close","volume"])

def fetch_t86_date(yyyymmdd: str, sleep_ms: int = 150) -> pd.DataFrame:
    url = f"{BASE}/fund/T86"
    params = {"response":"json","date":yyyymmdd,"selectType":"ALL"}
    key = f"TWSE_T86::{yyyymmdd}"
    c = Cache(); hit = c.load(key)
    if hit is not None: return hit
    r = Http().get(url, params=params, headers=HEADERS, sleep_ms=sleep_ms)
    js = _safe_json(r)
    data = js.get("data", [])
    rows=[]
    for row in data:
        try:
            stock_id = str(row[0]).strip()
            total_net = int(str(row[-1]).replace(",",""))
            rows.append({"date": pd.to_datetime(yyyymmdd).date(),"stock_id": stock_id,"inst_net": total_net})
        except Exception:
            continue
    df = pd.DataFrame(rows); c.save(key, df); return df
