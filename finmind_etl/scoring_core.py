from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

# ---- helpers ----

def _get_industry_col(df: pd.DataFrame, cfg: dict) -> str:
    col = cfg.get("industry_col", "industry")
    if col not in df.columns:
        # 常見別名：industry_category
        if "industry" in df.columns:
            return "industry"
        if "industry_category" in df.columns:
            # 建一個別名 column，讓後續 by="industry" 能使用
            df["industry"] = df["industry_category"]
            return "industry"
        # fallback
        df["industry"] = "UNKNOWN"
        return "industry"
    # 若設定是其他名稱，複製一份到 industry 以簡化後續 groupby
    if col != "industry":
        df["industry"] = df[col]
    return "industry"


def _mean_ignore_all_nan(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    # 全 NaN → 結果 NaN；否則平均非 NaN 值
    s = df[cols].mean(axis=1, skipna=True)
    all_nan_mask = df[cols].isna().all(axis=1)
    s[all_nan_mask] = np.nan
    return s


def build_four_pillars_from_prepercentile(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = cfg.get("features", {})
    out = df.copy()
    for pillar in ("tech","chip","fund","risk"):
        cols = feats.get(pillar, [])
        out[f"{pillar}_score"] = _mean_ignore_all_nan(out, cols)
    return out


def dynamic_weighted_total(df: pd.DataFrame, weights: Dict[str,float]) -> pd.Series:
    # 對每一列：只對非 NaN 面向做加權，並把權重正規化到 1
    pillars = ["tech","chip","fund","risk"]
    w = np.array([weights.get(p, 0.0) for p in pillars], dtype=float)
    mat = np.vstack([df[f"{p}_score"].to_numpy(dtype=float) for p in pillars])  # shape (4, N)
    nan_mask = np.isnan(mat)
    w_broadcast = w[:, None] * (~nan_mask)
    denom = w_broadcast.sum(axis=0)
    numer = np.nansum(mat * w[:, None], axis=0)
    out = np.full_like(numer, np.nan)
    nonzero = denom > 0
    out[nonzero] = numer[nonzero] / denom[nonzero]
    return pd.Series(out, index=df.index)

# ---- optional raw→percentile 路徑（保留既有流程）----

def mad(x: pd.Series) -> float:
    med = x.median()
    return (x - med).abs().median() or 1e-9


def winsorize(df: pd.DataFrame, cols: List[str], lo=0.01, hi=0.99) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        ql = df[c].quantile(lo)
        qh = df[c].quantile(hi)
        df[c] = df[c].clip(ql, qh)
    return df


def industry_neutralize(df: pd.DataFrame, cols: List[str], ind_col: str = "industry") -> pd.DataFrame:
    g = df.groupby(ind_col)
    for c in cols:
        if c not in df.columns:
            continue
        med = g[c].transform("median")
        m = g[c].transform(mad)
        df[c + "_adj"] = (df[c] - med) / m
    return df


def to_percentile(df: pd.DataFrame, cols: List[str], by: str | None = None) -> pd.DataFrame:
    if by:
        g = df.groupby(by)
        for c in cols:
            if c not in df.columns:
                continue
            df[c + "_pct"] = g[c].rank(pct=True).mul(100)
    else:
        for c in cols:
            if c not in df.columns:
                continue
            df[c + "_pct"] = df[c].rank(pct=True).mul(100)
    return df


def build_four_pillars_from_raw(df: pd.DataFrame, cfg: dict, universe: str) -> pd.DataFrame:
    # 依舊使用原流程：winsorize -> (可選) industry neutralize -> percentile -> 平均
    feats_cfg = cfg.get("features", {})
    all_cols = sum([v for v in feats_cfg.values()], [])
    out = df.copy()
    norm = cfg.get("normalization", {})
    lo, hi = (norm.get("winsorize_pct", [0.01, 0.99]) + [0,0])[:2]
    out = winsorize(out, all_cols, lo, hi)

    ind = _get_industry_col(out, cfg)
    if universe == "market_neutralized_by_industry":
        out = industry_neutralize(out, all_cols, ind_col=ind)
        base_cols = [c + "_adj" for c in all_cols]
        out = to_percentile(out, base_cols, by=None)
        pct_cols = [c + "_adj_pct" for c in all_cols]
    elif universe == "industry":
        out = to_percentile(out, all_cols, by=ind)
        pct_cols = [c + "_pct" for c in all_cols]
    else:
        out = to_percentile(out, all_cols, by=None)
        pct_cols = [c + "_pct" for c in all_cols]

    # 以 _pct 欄位平均成面向分數
    for pillar in ("tech","chip","fund","risk"):
        cols = [c + ("_adj_pct" if universe == "market_neutralized_by_industry" else "_pct") for c in feats_cfg.get(pillar, [])]
        cols = [c for c in cols if c in out.columns]
        out[f"{pillar}_score"] = _mean_ignore_all_nan(out, cols)
    return out

# ---- 統一入口 ----

def build_scores(df: pd.DataFrame, cfg: dict, universe: str = "market_neutralized_by_industry") -> pd.DataFrame:
    out = df.copy()
    _get_industry_col(out, cfg)  # 確保有 industry 列
    use_pre = bool(cfg.get("use_prepercentiled", False))
    if use_pre:
        out = build_four_pillars_from_prepercentile(out, cfg)
    else:
        out = build_four_pillars_from_raw(out, cfg, universe)
    out["score_total"] = dynamic_weighted_total(out, cfg.get("weights", {}))
    return out
