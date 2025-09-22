from __future__ import annotations
import time, json, hashlib, pathlib, datetime as dt
from typing import Dict, Any
import pandas as pd
import requests

class Http:
    def __init__(self, retry=3, backoff=1.5, timeout=30):
        self.retry, self.backoff, self.timeout = retry, backoff, timeout
    def get(self, url, params=None, headers=None):
        for i in range(self.retry):
            r = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(self.backoff ** (i+1)); continue
            r.raise_for_status()
        r.raise_for_status()

class Cache:
    def __init__(self, root: str = "official_raw/_cache"):
        self.root = pathlib.Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, key: str) -> pathlib.Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.root / f"{h}.parquet"
    def load(self, key: str):
        p = self._path(key)
        if p.exists():
            try: return pd.read_parquet(p)
            except Exception: return None
        return None
    def save(self, key: str, df: pd.DataFrame):
        p = self._path(key); df.to_parquet(p, index=False)

def parse_date_auto(s: str) -> pd.Timestamp:
    s = str(s).strip().replace("/", "-")
    # 嘗試民國年 113-09-21 → 2024-09-21
    try:
        parts = s.split("-")
        if len(parts[0]) in (2,3):  # 民國年
            y = int(parts[0]) + 1911
            s2 = f"{y}-{parts[1]}-{parts[2]}"
            return pd.to_datetime(s2)
    except Exception:
        pass
    return pd.to_datetime(s, errors="coerce")

def rolling_rsi(close: pd.Series, period=14) -> pd.Series:
    r = close.diff()
    up = r.clip(lower=0).rolling(period).mean()
    dn = (-r.clip(upper=0)).rolling(period).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))

def finalize_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    # 期望欄：date, stock_id, stock_name, open, high, low, close, volume
    df = df.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"])
    # 基礎衍生
    df["ret_1d"] = df.groupby("stock_id")["close"].pct_change()
    df["ret_5d"] = df.groupby("stock_id")["close"].pct_change(5)
    df["ret_20d"] = df.groupby("stock_id")["close"].pct_change(20)
    df["high_20d"] = df.groupby("stock_id")["high"].transform(lambda s: s.rolling(20).max())
    df["breakout_20d"] = df["close"] / df["high_20d"] - 1.0
    df["volatility_20d"] = df.groupby("stock_id")["ret_1d"].transform(lambda s: s.rolling(20).std())
    # 成交量相對強度（用量比代替真 turnover）
    ma20v = df.groupby("stock_id")["volume"].transform(lambda s: s.rolling(20).mean())
    df["volume_ratio_20d"] = (df["volume"] / ma20v).replace([pd.NA, pd.NaT], 0)
    # RSI
    df["rsi_14"] = df.groupby("stock_id")["close"].transform(rolling_rsi)
    return df

def add_industry(df: pd.DataFrame, universe_csv: str) -> pd.DataFrame:
    u = pd.read_csv(universe_csv, dtype={"stock_id":str})
    # 接受 industry 或 industry_category 欄位
    if "industry" not in u.columns and "industry_category" in u.columns:
        u["industry"] = u["industry_category"]
    u = u[["stock_id","stock_name","industry"]].drop_duplicates("stock_id")
    return df.merge(u, on="stock_id", how="left")
