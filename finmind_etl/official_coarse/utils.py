from __future__ import annotations
import time, json, hashlib, pathlib, datetime as dt, random
from typing import Dict, Any
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util import Retry
except Exception:
    from urllib3.util.retry import Retry

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/html,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

class Http:
    def __init__(self, retry=5, backoff=1.3, timeout=30, base_headers: Dict[str,str] | None = None):
        self.timeout = timeout
        self.backoff = backoff
        self.s = requests.Session()
        self.s.headers.update(DEFAULT_HEADERS | (base_headers or {}))
        r = Retry(
            total=retry,
            connect=retry,
            read=retry,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        ad = HTTPAdapter(max_retries=r, pool_connections=16, pool_maxsize=32)
        self.s.mount("https://", ad)
        self.s.mount("http://", ad)

    def get(self, url, params=None, headers=None, sleep_ms: int = 0):
        # 隨機抖動以減輕同時命中
        if sleep_ms:
            time.sleep((sleep_ms + random.randint(0, sleep_ms//2)) / 1000.0)
        h = (headers or {})
        for i in range(8):  # 額外手動 retry 幾次，涵蓋某些非 5xx 的斷線
            try:
                r = self.s.get(url, params=params, headers=h, timeout=self.timeout, allow_redirects=True)
                # 某些情況返回 200 但內容無效，交給呼叫端判斷
                return r
            except requests.exceptions.RequestException:
                time.sleep((self.backoff ** (i+1)) * 0.5)
                continue
        # 最後再丟一次讓上層接住
        return self.s.get(url, params=params, headers=h, timeout=self.timeout, allow_redirects=True)

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
    try:
        parts = s.split("-")
        if len(parts[0]) in (2,3):
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
    df = df.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"])
    df["ret_1d"] = df.groupby("stock_id")["close"].pct_change()
    df["ret_5d"] = df.groupby("stock_id")["close"].pct_change(5)
    df["ret_20d"] = df.groupby("stock_id")["close"].pct_change(20)
    df["high_20d"] = df.groupby("stock_id")["high"].transform(lambda s: s.rolling(20).max())
    df["breakout_20d"] = df["close"] / df["high_20d"] - 1.0
    df["volatility_20d"] = df.groupby("stock_id")["ret_1d"].transform(lambda s: s.rolling(20).std())
    ma20v = df.groupby("stock_id")["volume"].transform(lambda s: s.rolling(20).mean())
    df["volume_ratio_20d"] = (df["volume"] / ma20v).replace([pd.NA, pd.NaT], 0)
    df["rsi_14"] = df.groupby("stock_id")["close"].transform(rolling_rsi)
    return df

def add_industry(df: pd.DataFrame, universe_csv: str) -> pd.DataFrame:
    u = pd.read_csv(universe_csv, dtype={"stock_id":str})
    if "industry" not in u.columns and "industry_category" in u.columns:
        u["industry"] = u["industry_category"]
    u = u[["stock_id","stock_name","industry"]].drop_duplicates("stock_id")
    return df.merge(u, on="stock_id", how="left")
