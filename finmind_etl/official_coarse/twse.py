from __future__ import annotations
import pandas as pd
from .utils import Http, Cache, parse_date_auto

BASE = "https://www.twse.com.tw"

def fetch_stock_day_month(stock_id: str, yyyymm: str) -> pd.DataFrame:
    # /exchangeReport/STOCK_DAY?response=json&date=YYYYMM01&stockNo=2330
    url = f"{BASE}/exchangeReport/STOCK_DAY"
    params = {"response":"json","date":yyyymm+"01","stockNo":stock_id}
    key = f"TWSE_STOCK_DAY::{stock_id}::{yyyymm}"
    c = Cache(); hit = c.load(key)
    if hit is not None: return hit
    r = Http().get(url, params=params).json()
    rows = []
    for row in r.get("data", []):
        # [日期, 成交股數, 成交金額, 開盤價, 最高, 最低, 收盤, 漲跌, 成交筆數]
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
    df = pd.DataFrame(rows)
    c.save(key, df)
    return df

def fetch_t86_date(yyyymmdd: str) -> pd.DataFrame:
    # /fund/T86?response=json&date=YYYYMMDD&selectType=ALL
    url = f"{BASE}/fund/T86"
    params = {"response":"json","date":yyyymmdd,"selectType":"ALL"}
    key = f"TWSE_T86::{yyyymmdd}"
    c = Cache(); hit = c.load(key)
    if hit is not None: return hit
    r = Http().get(url, params=params).json()
    data = r.get("data", [])
    # 官方欄位順序可能會變，盡力用索引+名稱兼容
    # 常見欄：[證券代號, 證券名稱, 外陸資買進股數, 外陸資賣出股數, 外陸資買賣超股數, 投信買進股數, 投信賣出股數, 投信買賣超股數, 自營商買進股數(自行買賣), 自營商賣出股數(自行買賣), 自營商買賣超股數(自行買賣), 自營商買進股數(避險), 自營商賣出股數(避險), 自營商買賣超股數(避險), 三大法人買賣超股數]
    rows=[]
    for row in data:
        try:
            stock_id = str(row[0]).strip()
            total_net = int(str(row[-1]).replace(",",""))
            rows.append({"date": pd.to_datetime(yyyymmdd).date(),
                         "stock_id": stock_id,
                         "inst_net": total_net})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    c.save(key, df)
    return df
