from __future__ import annotations
import math, datetime as dt
from pathlib import Path
import pandas as pd
from .utils import finalize_daily_panel, add_industry
from . import twse, tpex

def _month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while True:
        cur = dt.date(y, m, 1)
        if cur > end.replace(day=1): break
        yield f"{y}{m:02d}"
        m += 1
        if m > 12: y += 1; m = 1

def build_from_official(universe_csv: str, since: str, until: str, out_features: str, sleep_ms: int = 250):
    uni = pd.read_csv(universe_csv, dtype={"stock_id":str})
    if "market" in uni.columns:
        is_otc = uni["market"].astype(str).str.contains("櫃|OTC|TPEx", case=False, regex=True)
    else:
        is_otc = pd.Series(False, index=uni.index)
    start = pd.to_datetime(since).date()
    end   = pd.to_datetime(until).date()

    frames=[]
    for idx, r in uni.iterrows():
        sid = str(r["stock_id"])
        try:
            if is_otc.loc[idx]:
                roc_start = start.year - 1911; roc_end = end.year - 1911
                for y in range(roc_start, roc_end+1):
                    for m in range(1, 13):
                        ym = dt.date(y+1911, m, 1)
                        if ym < start or ym > end: continue
                        df = tpex.fetch_stock_day_month_csv(sid, y, m, sleep_ms=sleep_ms)
                        frames.append(df)
            else:
                for ym in _month_iter(start, end):
                    df = twse.fetch_stock_day_month(sid, ym, sleep_ms=sleep_ms)
                    frames.append(df)
        except Exception:
            # 記錄錯誤但不中斷；該股票缺資料將在後面自然被濾掉
            continue

    px = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date","stock_id","open","high","low","close","volume"])

    # T86 （上市）
    t86_frames=[]
    cur = start
    while cur <= end:
        ymd = cur.strftime("%Y%m%d")
        try:
            t86_frames.append(twse.fetch_t86_date(ymd, sleep_ms=100))
        except Exception:
            pass
        cur += dt.timedelta(days=1)
    t86 = pd.concat(t86_frames, ignore_index=True) if t86_frames else pd.DataFrame(columns=["date","stock_id","inst_net"])

    df = px.copy()
    if df.empty:
        Path(out_features).parent.mkdir(parents=True, exist_ok=True)
        # 即便空，也產生檔案讓流程不中斷
        pd.DataFrame(columns=["stock_id","ret_5d","ret_20d"]).to_csv(out_features, index=False, encoding="utf-8")
        return out_features

    df = finalize_daily_panel(df)
    df = df.merge(t86, on=["date","stock_id"], how="left")
    g = df.groupby("stock_id", group_keys=False)
    df["inst_net_5d"] = g["inst_net"].apply(lambda s: s.rolling(5).sum())
    df["vol_5d"]      = g["volume"].apply(lambda s: s.rolling(5).sum())
    df["inst_net_buy_5d_ratio"] = df["inst_net_5d"] / df["vol_5d"].replace(0, pd.NA)
    df["inst_consistency_20d"]  = g["inst_net"].apply(lambda s: (s.fillna(0)>0).rolling(20).mean())

    snap = df[df["date"] == pd.to_datetime(until).date()].copy()
    keep = [
        "stock_id","open","high","low","close","volume",
        "ret_5d","ret_20d","rsi_14","breakout_20d","volatility_20d","volume_ratio_20d",
        "inst_net_buy_5d_ratio","inst_consistency_20d"
    ]
    snap = snap[[c for c in keep if c in snap.columns]]
    snap = add_industry(snap, universe_csv)

    Path(out_features).parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(out_features, index=False, encoding="utf-8")
    return out_features
