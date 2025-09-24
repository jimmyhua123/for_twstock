# finmind_features_scoring.py
# ------------------------------------------------------------
# 讀取標準化後的 CSV（與可選 raw PER JSON），計算四大面向特徵與分數
# 預設輸出最新交易日的快照（scores_watchlist.csv / scores_breakdown.csv / features_snapshot.csv）
# 可選 --full-daily 輸出整段期間 features_daily.csv
# 所有 CSV 以 encoding="utf-8-sig" 輸出
# ------------------------------------------------------------
import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from datetime import datetime

def log(msg: str) -> None:
    print(msg, flush=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"[警告] 找不到檔案：{path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # NEW: 大小寫無關對齊到 stock_id
    if "stock_id" not in df.columns:
        # 先找「名稱其實就是 stock_id（大小寫/底線不同）」的欄
        for c in list(df.columns):
            if c.strip().lower().replace("-", "").replace("_", "") == "stockid":
                df = df.rename(columns={c: "stock_id"})
                break

    # 若沒有 stock_id 再嘗試常見別名
    if "stock_id" not in df.columns:
        for alt in ("stock_code", "code", "symbol", "security_id", "證券代號", "證券代碼"):
            if alt in df.columns:
                df = df.rename(columns={alt: "stock_id"})
                break

    # 若 index 名稱是 stock_id（或其別名），也攤回欄
    idx_name = getattr(df.index, "name", None)
    if "stock_id" not in df.columns and idx_name and idx_name.strip().lower() in (
        "stock_id", "stockcode", "code", "symbol", "security_id", "證券代號", "證券代碼"
    ):
        df = df.reset_index().rename(columns={idx_name: "stock_id"})

    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()

    return df

# def read_csv(path: Path) -> pd.DataFrame:
#     if not path.exists():
#         log(f"[警告] 找不到檔案：{path}")
#         return pd.DataFrame()
#     # 關鍵：輸出是 utf-8-sig，讀取也要用 utf-8-sig；並去掉欄名可能殘留的 BOM
#     df = pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")
#     df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
#     if "stock_id" in df.columns:
#         df["stock_id"] = df["stock_id"].astype(str).str.strip()
#     return df

def read_per_from_raw(raw_dir: Path) -> pd.DataFrame:
    """讀 raw/TaiwanStockPER.json，回傳每檔最新 pe/pb（自動兼容欄名：PE/PER/PE_ratio、PB/PBR/PB_ratio）。"""
    src = raw_dir / "TaiwanStockPER.json"
    if not src.exists():
        return pd.DataFrame()
    with open(src, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", []) if isinstance(obj, dict) else obj
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # 統一 stock_id 型別
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()

    # 可能的欄名對應（大小寫/不同寫法）
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"pe", "per", "pe_ratio", "peratio"}:
            rename_map[c] = "pe"
        if lc in {"pb", "pbr", "pb_ratio"}:
            rename_map[c] = "pb"
    if rename_map:
        df = df.rename(columns=rename_map)

    # 只保留需要欄（若沒抓到也會留空）
    keep = ["stock_id"] + [c for c in ("pe","pb","dividend_yield") if c in df.columns]

    df = df[keep].copy()

    # 取每檔最新一筆（若 PER 檔有 date 欄位）
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["stock_id", "date"]).drop_duplicates("stock_id", keep="last")

    # 轉數值
    for col in ["pe", "pb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



# --------------------------
# 技術面工具：SMA / RSI / VOL
# --------------------------
def sma(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=1).mean()

def rolling_vol(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=2).std()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def pct_rank_with_fallback(df: pd.DataFrame, col: str, by: str = "industry_category", min_group: int = 5) -> pd.Series:
     """
     先在產業(by)內做百分位（0~100）；若該產業樣本數 < min_group，則回退用「全體樣本」的百分位。
     """
     s = df[col]
     # 全體百分位
     global_rank = s.rank(pct=True) * 100.0 if s.notna().any() else s
     if by not in df.columns:
         return global_rank
     # 產業內百分位
     def _rank(ss: pd.Series):
         return ss.rank(pct=True) * 100.0 if ss.notna().any() else ss
     by_rank = df.groupby(by, dropna=False)[col].transform(_rank)
     by_size = df.groupby(by, dropna=False)[col].transform("size")
     out = by_rank.copy()
     need_fallback = (by_size < min_group) | by_size.isna()
     out[need_fallback] = global_rank[need_fallback]
     return out

def invert_score(s: pd.Series) -> pd.Series:
    """把高越好轉為低越好（或反之）：score -> 100 - score。"""
    return 100.0 - s


# --------------------------
# 主流程
# --------------------------
def build_features(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    # 基礎表
    dim = read_csv(clean_dir / "dim_security.csv")
    price = read_csv(clean_dir / "fact_price_daily.csv")
    mkt = read_csv(clean_dir / "fact_market_return.csv")
    inst = read_csv(clean_dir / "fact_institutional_flow_daily.csv")
    share = read_csv(clean_dir / "fact_foreign_shareholding_daily.csv")
    margin = read_csv(clean_dir / "fact_margin_short_daily.csv")
    rev = read_csv(clean_dir / "fact_revenue_monthly.csv")


    # 估值（可選）
    per_df = read_per_from_raw(raw_dir) if raw_dir else pd.DataFrame()

    # 讀完之後馬上做「stock_id → str」正規化，避免 dtype 不一致
    for df in [dim, price, mkt, inst, share, margin, rev]:
        if not df.empty and "stock_id" in df.columns:
            df["stock_id"] = df["stock_id"].astype(str).str.strip()



    # 價格表：保證 (stock_id, date) 唯一
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")

    # 估值（PER/PBR）同樣轉字串
    if not per_df.empty and "stock_id" in per_df.columns:
        per_df["stock_id"] = per_df["stock_id"].astype(str).str.strip()

    # 檢查必要
    if price.empty:
        raise SystemExit("[錯誤] 找不到 fact_price_daily.csv，請先執行清理/標準化腳本。")

    # 日期轉型
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values(["stock_id", "date"])

    # 技術面：SMA / RSI / VOL / 量能倍數
    for win in (20, 60, 120):
        price[f"sma{win}"] = price.groupby("stock_id")["close"].transform(lambda s: sma(s, win))
        price[f"sma{win}_gap_pct"] = (price["close"] / price[f"sma{win}"] - 1.0) * 100.0

    # RSI(14)
    price["rsi14"] = price.groupby("stock_id")["close"].transform(lambda s: rsi(s, 14))

    # 波動（20D）
    price["rolling_vol_20d"] = price.groupby("stock_id")["ret_1d"].transform(lambda s: rolling_vol(s, 20))

    # 量能倍數（vs 20D 成交量均值）
    if "Trading_Volume" in price.columns:
        price["vol_mult_20d"] = price.groupby("stock_id")["Trading_Volume"].transform(
            lambda s: s / (s.rolling(20, min_periods=1).mean().replace(0, np.nan))
        )

    # 市場相對（excess_ret_20d：個股 20D - TAIEX 20D）
    if not mkt.empty:
        mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
        taiex = mkt[mkt["stock_id"].astype(str) == "TAIEX"].copy()
        if "mkt_ret_20d" in taiex.columns:
            taiex = taiex.rename(columns={"mkt_ret_20d": "taiex_ret_20d"})
        elif "ret_20d" in taiex.columns:
            taiex = taiex.rename(columns={"ret_20d": "taiex_ret_20d"})
        else:
            taiex = pd.DataFrame(columns=["date", "taiex_ret_20d"])
        if not taiex.empty and "taiex_ret_20d" in taiex.columns:
            price = price.merge(taiex[["date", "taiex_ret_20d"]], on="date", how="left")
            if "ret_20d" in price.columns:
                price["excess_ret_20d"] = price["ret_20d"] - price["taiex_ret_20d"]

    # 籌碼面：三大法人 5D/20D 淨買賣合計（若存在）
    chip = pd.DataFrame()
    if not inst.empty:
        inst["date"] = pd.to_datetime(inst["date"], errors="coerce")
        # 這裡的 inst 已經是「寬表」（清理腳本輸出），不需要再 pivot，一律先排序+去重
        inst = inst.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")
        net5_cols  = [c for c in inst.columns if c.startswith("net_") and c.endswith("_5d")]
        net20_cols = [c for c in inst.columns if c.startswith("net_") and c.endswith("_20d")]
        keep_cols = ["stock_id","date"] + net5_cols + net20_cols
        chip = inst[keep_cols].copy()
        chip["chip_net5_sum"]  = chip[net5_cols].sum(axis=1, skipna=True) if net5_cols else np.nan
        chip["chip_net20_sum"] = chip[net20_cols].sum(axis=1, skipna=True) if net20_cols else np.nan

    # 外資持股比變化（5D/20D）
    share_feat = pd.DataFrame()
    if not share.empty:
        share["date"] = pd.to_datetime(share["date"], errors="coerce")
        share = share.sort_values(["stock_id", "date"])
        ratio_cols = [c for c in share.columns if "ratio" in c.lower()]
        if ratio_cols:
            ratio_col = ratio_cols[0]
            share_feat = share[["stock_id", "date", ratio_col]].copy()
            share_feat["foreign_ratio_5d_chg"] = share_feat.groupby("stock_id")[ratio_col].diff(5)
            share_feat["foreign_ratio_20d_chg"] = share_feat.groupby("stock_id")[ratio_col].diff(20)
            share_feat = share_feat.rename(columns={ratio_col: "foreign_ratio"})
            # 保證 (stock_id,date) 唯一
            share_feat = share_feat.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")

    # 融資/融券變化（若在清理步驟已計 delta）
    margin_feat = pd.DataFrame()
    if not margin.empty:
        margin["date"] = pd.to_datetime(margin["date"], errors="coerce")
        margin = margin.sort_values(["stock_id", "date"])
        # 找出所有 *_TodayBalance_delta 欄位
        delta_cols = [c for c in margin.columns if c.endswith("TodayBalance_delta")]
        if delta_cols:
            # 先做同一天「橫向加總」→ 每檔每日的綜合變動量
            tmp = margin[["stock_id", "date"] + delta_cols].copy()
            # 確保為數值型（避免字串導致相加變成拼接）
            for c in delta_cols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            tmp["margin_delta_row_sum"] = tmp[delta_cols].sum(axis=1, skipna=True)
            # 依股票做 5/20 日「縱向滾動和」
            tmp = tmp.sort_values(["stock_id", "date"])
            tmp["margin_delta_5d_sum"] = (
                tmp.groupby("stock_id")["margin_delta_row_sum"]
                   .transform(lambda s: s.rolling(5, min_periods=1).sum())
            )
            tmp["margin_delta_20d_sum"] = (
                tmp.groupby("stock_id")["margin_delta_row_sum"]
                   .transform(lambda s: s.rolling(20, min_periods=1).sum())
            )

            margin_feat = tmp[["stock_id", "date", "margin_delta_row_sum",
                               "margin_delta_5d_sum", "margin_delta_20d_sum"]]
            # 保證 (stock_id,date) 唯一
            margin_feat = margin_feat.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")
        else:
            margin_feat = pd.DataFrame()


    # 合併特徵
    feats = price.copy()
    # 強保：左表 feats 必須要有 stock_id
    if "stock_id" not in feats.columns:
        # 若 index 叫 stock_id / code... 就攤回欄
        idx_name = getattr(feats.index, "name", None)
        if idx_name and idx_name.strip().lower() in ("stock_id","stockcode","code","symbol","security_id","證券代號","證券代碼"):
            feats = feats.reset_index().rename(columns={idx_name: "stock_id"})
        else:
            # 在欄位裡再找一次（大小寫/底線無關）
            for c in list(feats.columns):
                if c.strip().lower().replace("-", "").replace("_", "") in ("stockid","stockcode","code","symbol","securityid"):
                    feats = feats.rename(columns={c: "stock_id"})
                    break
    if "stock_id" in feats.columns:
        feats["stock_id"] = feats["stock_id"].astype(str).str.strip()
    else:
        log("[嚴重] 價格左表（feats）仍缺少 stock_id，後續將無法分檔計算；請檢查 clean/fact_price_daily.csv 欄名。")

    for df_ in [chip, share_feat, margin_feat]:
        if not df_.empty:
            feats = feats.merge(df_, on=["stock_id", "date"], how="left")
            # 立刻壓成唯一鍵，避免一對多放大
            feats = feats.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")

    # --- 基本面：月營收 YoY / MoM（右對齊：逐檔 as-of 合併）---
    if not rev.empty:
        if "date" in rev.columns:
            rev["announce_date"] = pd.to_datetime(rev["date"], errors="coerce")
        rev_small = (
            rev[["stock_id", "announce_date", "revenue_yoy", "revenue_mom"]]
            .dropna(subset=["stock_id", "announce_date"])
            .assign(stock_id=lambda d: d["stock_id"].astype(str).str.strip())
            .sort_values(["stock_id", "announce_date"])
            .drop_duplicates(["stock_id", "announce_date"], keep="last")
        )

        feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
        feats = (
            feats.dropna(subset=["stock_id", "date"])
                .assign(stock_id=lambda d: d["stock_id"].astype(str).str.strip())
                .sort_values(["stock_id", "date"])
        )

        merged_parts = []
        for sid, left_g in feats.groupby("stock_id", sort=False):
            left_g = left_g.sort_values("date")
            right_g = rev_small[rev_small["stock_id"] == sid].sort_values("announce_date")
            if right_g.empty:
                lg = left_g.copy()
                lg["revenue_yoy"] = np.nan
                lg["revenue_mom"] = np.nan
                merged_parts.append(lg)
                continue

            # 關鍵：避免 merge_asof 產生 stock_id_x / stock_id_y
            right_g = right_g.drop(columns=["stock_id"])

            mg = pd.merge_asof(
                left_g,
                right_g,
                left_on="date",
                right_on="announce_date",
                direction="backward",
                allow_exact_matches=True
            ).drop(columns=["announce_date"], errors="ignore")

            merged_parts.append(mg)

        feats = pd.concat(merged_parts, ignore_index=True)

        # --- 重要：併 PER 前，先確保左表 feats 一定有 'stock_id' 欄（不是放在 index 也不是別名）---
        if "stock_id" not in feats.columns:
            # 若 index 名就叫 stock_id，攤回欄
            idx_name = getattr(feats.index, "name", None)
            if idx_name in ("stock_id",):
                feats = feats.reset_index()
            else:
                # 嘗試把常見別名改名為 stock_id
                for alt in ("code", "stock_code", "symbol", "security_id", "證券代號", "證券代碼"):
                    if alt in feats.columns:
                        feats = feats.rename(columns={alt: "stock_id"})
                        break
        # 還是沒有就直接跳過估值合併（避免炸掉）
        if "stock_id" not in feats.columns:
            log("[警告] 左表 feats 缺少 'stock_id' 欄，跳過 PER/PBR 併入。")
            feats["pe"] = feats.get("pe", np.nan)
            feats["pb"] = feats.get("pb", np.nan)
            # 若你也要殖利率：
            if "dividend_yield" not in feats.columns:
                feats["dividend_yield"] = np.nan
    
        # 估值：PE / PB（用 raw 最新值）
        # 這段是安全版：只有左右表都有 'stock_id' 才併；否則補 NaN
        per_cols = []
        if not per_df.empty:
            if "stock_id" in per_df.columns:
                per_df["stock_id"] = per_df["stock_id"].astype(str).str.strip()
            per_cols = [c for c in ["pe", "pb", "dividend_yield"] if c in per_df.columns]

        # 只有當 feats、per_df 都有 'stock_id' 且有可併欄位時才併
        if ("stock_id" in feats.columns) and (not per_df.empty) and ("stock_id" in per_df.columns) and per_cols:
            feats = feats.merge(per_df[["stock_id"] + per_cols], on="stock_id", how="left")
            for col in per_cols:
                feats[col] = feats.groupby("stock_id")[col].ffill()
        else:
            # 沒法併就保證欄位存在，避免後面引用出錯
            for c in ("pe", "pb", "dividend_yield"):
                if c not in feats.columns:
                    feats[c] = np.nan


        # 附加維度：產業
        if not dim.empty and ("stock_id" in dim.columns) and ("stock_id" in feats.columns):
            cols = ["stock_id"]
            if "stock_name" in dim.columns:
                cols.append("stock_name")
            if "industry_category" in dim.columns:
                cols.append("industry_category")
            feats = feats.merge(dim[cols], on="stock_id", how="left")
        else:
            # 沒法併就補空欄，避免後面引用出錯
            for c in ("stock_name", "industry_category"):
                if c not in feats.columns:
                    feats[c] = np.nan


        feats = feats.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")
        return feats


# def score_snapshot(feats: pd.DataFrame, asof: Optional[str], out_dir: Path,
#                    w_tech=0.3, w_chip=0.3, w_fund=0.3, w_risk=0.1,
#                    focus_ids: Optional[list] = None) -> None:
def score_snapshot(feats: pd.DataFrame, asof: Optional[str], out_dir: Path,
                   w_tech=0.3, w_chip=0.3, w_fund=0.3, w_risk=0.1,
                   focus_ids: Optional[list] = None,
                   snapshot_suffix: str = "") -> None:    
    """以 as-of 日期做同業標準化打分，輸出三個檔：scores_watchlist / breakdown / features_snapshot"""
    # 取評分日期
    if asof:
        asof_date = pd.to_datetime(asof)
    else:
        asof_date = feats["date"].max()

    snap = feats[feats["date"] == asof_date].copy()
    if snap.empty:
        raise SystemExit(f"[錯誤] 找不到評分日期的資料：{asof_date.date()}")
    # 關鍵：同一天同一檔只留一列
    snap = snap.sort_values(["stock_id","date"]).drop_duplicates(["stock_id","date"], keep="last")
    snap = snap.drop_duplicates("stock_id", keep="last")
    # 風險 winsorize（避免極端值把分數打到 0）
    if "rolling_vol_20d" in snap.columns:
        q1 = snap["rolling_vol_20d"].quantile(0.01)
        q99 = snap["rolling_vol_20d"].quantile(0.99)
        snap["rolling_vol_20d_w"] = snap["rolling_vol_20d"].clip(q1, q99)

    # ---- 技術面候選特徵 ----
    tech_cols = []
    for c in ["ret_20d", "excess_ret_20d", "sma20_gap_pct", "sma60_gap_pct", "sma120_gap_pct", "rsi14"]:
        if c in snap.columns:
            tech_cols.append(c)
    if "vol_mult_20d" in snap.columns:
        # 量能倍數：高通常代表關注度上升，視為加分（也可改為介於 1~2 最佳的邏輯）
        tech_cols.append("vol_mult_20d")

    # ---- 籌碼面候選特徵 ----
    chip_cols = []
    for c in ["chip_net5_sum", "chip_net20_sum", "foreign_ratio_5d_chg", "foreign_ratio_20d_chg",
              "margin_delta_5d_sum", "margin_delta_20d_sum"]:
        if c in snap.columns:
            chip_cols.append(c)

    # ---- 基本面候選特徵 ----
    fund_cols = []
    for c in ["revenue_yoy", "revenue_mom", "pe", "pb"]:
        if c in snap.columns:
            fund_cols.append(c)

    # ---- 風險候選特徵（低越好 → 轉換成高分）----
    risk_cols = []
    # 先用 winsorize 後的欄位；沒有就退回原欄
    if "rolling_vol_20d_w" in snap.columns:
        risk_cols.append("rolling_vol_20d_w")
    elif "rolling_vol_20d" in snap.columns:
        risk_cols.append("rolling_vol_20d")
    # 跳空幅度（若有 open/prev close 可擴充；此版先不強制）
    # if {"open"}.issubset(snap.columns):
    #     snap["gap_pct"] = (snap["open"] - snap.groupby("stock_id")["close"].shift(1)) / snap.groupby("stock_id")["close"].shift(1)
    #     risk_cols.append("gap_pct")

    # 產業
    if "industry_category" not in snap.columns:
        snap["industry_category"] = "UNKNOWN"

    # 做分數：在產業內做百分位（0~100）
    def add_scores(cols: List[str], prefix: str, invert: Optional[List[str]] = None, min_group: int = 5):
        invert = invert or []
        for c in cols:
            if c not in snap.columns:
                continue
            s = pct_rank_with_fallback(snap, c, by="industry_category", min_group=min_group)
            if c in invert:
                s = invert_score(s)
            snap[f"score_{prefix}_{c}"] = s

    # 技術面分數：全部「高越好」，RSI 接近 50 最佳的處理要複雜些，此版先視為中性「高越好」
    add_scores(tech_cols, "tech")

    # 籌碼面分數：全部「高越好」（買超/持股比上升/融資變化正向等）
    add_scores(chip_cols, "chip")

    # 基本面分數：revenue_yoy/mom「高越好」，估值 PE/PB「低越好」
    invert_list = [c for c in fund_cols if c in ("pe", "pb")]
    add_scores(fund_cols, "fund", invert=invert_list)

    # 風險面分數：rolling_vol_20d「低越好」→ 轉換
    add_scores(risk_cols, "risk", invert=risk_cols)

    # 聚合四大面向
    def avg_of(prefix: str) -> pd.Series:
        cols = [c for c in snap.columns if c.startswith(f"score_{prefix}_")]
        if not cols:
            return pd.Series(np.nan, index=snap.index)
        return snap[cols].mean(axis=1)

    snap["score_tech"] = avg_of("tech")
    snap["score_chip"] = avg_of("chip")
    snap["score_fund"] = avg_of("fund")
    snap["score_risk"] = avg_of("risk")

    # 總分（加權）
    weights = (w_tech, w_chip, w_fund, w_risk)
    snap["score_total"] = (
        w_tech * snap["score_tech"].fillna(0) +
        w_chip * snap["score_chip"].fillna(0) +
        w_fund * snap["score_fund"].fillna(0) +
        w_risk * snap["score_risk"].fillna(0)
    )

    # 附上名稱
    cols_name = []
    for c in ["stock_name", "industry_category"]:
        if c in snap.columns:
            cols_name.append(c)

    # 觀察清單（重點欄位）
    watch_cols = ["stock_id"] + cols_name + [
        "close", "ret_5d", "ret_20d", "excess_ret_20d",
        "chip_net5_sum", "chip_net20_sum", "foreign_ratio_5d_chg", "revenue_yoy",
        "pe", "pb",
        "score_tech", "score_chip", "score_fund", "score_risk", "score_total"
    ]
    watch_cols = [c for c in watch_cols if c in snap.columns]
    watch = snap[watch_cols].sort_values("score_total", ascending=False)
    watch.insert(1, "date", asof_date.normalize())
    # 若提供 focus 清單：以該清單 reindex（缺檔也保留一列，便於比對）
    if focus_ids:
        focus_ids = [str(s).strip() for s in focus_ids if str(s).strip()]
        # 若 watch 缺少某些 id，就補一列空資料
        watch = (watch.set_index("stock_id")
                       .reindex(focus_ids)  # 這步會自動補缺檔為 NaN
                       .reset_index()
                       .rename(columns={"index":"stock_id"}))
        # 重新把非 NaN 的排前面（維持 focus 清單順序）
        # 不額外排序，保留清單原本序

    # 拆解分數（debug 用）
    breakdown_cols = ["stock_id"] + cols_name + [c for c in snap.columns if c.startswith("score_")]
    breakdown = snap[breakdown_cols]

    # 完整特徵快照
    snapshot = snap.copy()

    # 輸出（BOM）
    # 檔名加上時間流水碼：YYYYMMDD_HHMMSS（評分日 + 產報時間）
    stamp = f"{asof_date.strftime('%Y%m%d')}"
    print(f"[評分] as-of = {asof_date.date()}，檔名時間碼 = {stamp}")
    # 輸出（BOM）
    out_watch   = out_dir / f"scores_watchlist_{stamp}.csv"
    out_break   = out_dir / f"scores_breakdown_{stamp}.csv"
    # out_snapshot= out_dir / f"features_snapshot_{stamp}.csv"
    tag         = f"_{snapshot_suffix}" if snapshot_suffix else ""
    out_snapshot= out_dir / f"features_snapshot{tag}_{stamp}.csv"    

    watch.to_csv(out_watch, index=False, encoding="utf-8-sig")
    breakdown.to_csv(out_break, index=False, encoding="utf-8-sig")
    snapshot.to_csv(out_snapshot, index=False, encoding="utf-8-sig")

    log(f"[輸出] {out_watch.name}")
    log(f"[輸出] {out_break.name}")
    log(f"[輸出] {out_snapshot.name}")


def export_full_daily(feats: pd.DataFrame, out_dir: Path) -> None:
    out_daily = out_dir / "features_daily.csv"
    feats.to_csv(out_daily, index=False, encoding="utf-8-sig")
    log(f"[輸出] {out_daily.name}（全期間特徵）")


def main():
    ap = argparse.ArgumentParser(description="四大面向特徵與評分")
    ap.add_argument("--clean-dir", type=str, default="finmind_out", help="清理/標準化後 CSV 目錄（預設：finmind_out）")
    ap.add_argument("--raw-dir", type=str, default="finmind_raw", help="raw JSON 目錄（用於讀取 TaiwanStockPER.json）")
    ap.add_argument("--out-dir", type=str, default="finmind_scores", help="輸出目錄（預設：finmind_scores）")
    ap.add_argument("--asof", type=str, default=None, help="評分日期 YYYY-MM-DD（不給則取最新交易日）")
    # ap.add_argument("--full-daily", action="store_true", help="輸出整段期間的每日特徵（檔案較大）")
    ap.add_argument("--full-daily", action="store_true", help="輸出整段期間的每日特徵（檔案較大）")
    ap.add_argument("--outfile-suffix", type=str, default="fine",
                    help="features 快照檔名尾碼（預設 fine → features_snapshot_fine_YYYYMMDD.csv；留空則不加尾碼）")
    ap.add_argument("--w-tech", type=float, default=0.3, help="技術面權重（預設 0.3）")
    ap.add_argument("--w-chip", type=float, default=0.3, help="籌碼面權重（預設 0.3）")
    ap.add_argument("--w-fund", type=float, default=0.3, help="基本面權重（預設 0.3）")
    ap.add_argument("--w-risk", type=float, default=0.1, help="風險面權重（預設 0.1）")
    ap.add_argument("--focus-ids", type=str, default=None,help="以逗號分隔的股票代碼清單，快照輸出只排序這些並補齊缺檔")
    ap.add_argument("--focus-file", type=str, default=None,help="文字檔，每行一個股票代碼；與 --focus-ids 合併去重")
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = build_features(clean_dir, raw_dir)
    # 解析 focus 清單（--focus-ids 與 --focus-file 合併）
    focus_ids = []
    if args.focus_ids:
        focus_ids += [s.strip() for s in args.focus_ids.split(",") if s.strip()]
    if args.focus_file:
        p = Path(args.focus_file)
        if p.exists():
            focus_ids += [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    # 去重並保持原順序
    seen = set(); ordered_focus = []
    for s in focus_ids:
        if s not in seen:
            ordered_focus.append(s); seen.add(s)

    # score_snapshot(feats, args.asof, out_dir, args.w_tech, args.w_chip, args.w_fund, args.w_risk,
    #                focus_ids=ordered_focus if ordered_focus else None)
    score_snapshot(
        feats, args.asof, out_dir,
        args.w_tech, args.w_chip, args.w_fund, args.w_risk,
        focus_ids=ordered_focus if ordered_focus else None,
        snapshot_suffix=(args.outfile_suffix or "")
    )

    if args.full_daily:
        export_full_daily(feats, out_dir)


if __name__ == "__main__":
    main()
