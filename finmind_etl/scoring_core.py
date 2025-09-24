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
    arr = df[cols].to_numpy(dtype=float)
    cnt = np.sum(~np.isnan(arr), axis=1)
    s = np.nansum(arr, axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    mask = cnt > 0
    out[mask] = s[mask] / cnt[mask]
    return pd.Series(out, index=df.index)


def combine_pillars(df: pd.DataFrame, pillars: Dict[str, List[str]], weights: Dict[str, float]) -> pd.DataFrame:
    pillar_scores = {}
    ordered: List[str] = []
    for name, cols in pillars.items():
        pillar_scores[f"{name}_score"] = _mean_ignore_all_nan(df, cols)
        ordered.append(name)
    out = pd.DataFrame(pillar_scores, index=df.index)
    if not ordered:
        out["score_total"] = np.nan
        return out
    score_matrix = np.vstack([
        out.get(f"{name}_score", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        for name in ordered
    ])
    weight_vec = np.array([weights.get(name, 0.0) for name in ordered], dtype=float)
    valid_mask = ~np.isnan(score_matrix)
    weight_adj = weight_vec[:, None] * valid_mask
    denom = weight_adj.sum(axis=0)
    numer = np.nansum(score_matrix * weight_vec[:, None], axis=0)
    total = np.full(df.shape[0], np.nan, dtype=float)
    ok = denom > 0
    total[ok] = numer[ok] / denom[ok]
    out["score_total"] = total
    return out


def build_four_pillars_from_prepercentile(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = cfg.get("features", {})
    out = df.copy()
    scores = combine_pillars(out, feats, cfg.get("weights", {}))
    return pd.concat([out, scores], axis=1)

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

    suffix = "_adj_pct" if universe == "market_neutralized_by_industry" else "_pct"
    pillar_pct = {
        pillar: [f"{c}{suffix}" for c in feats_cfg.get(pillar, []) if f"{c}{suffix}" in out.columns]
        for pillar in ("tech", "chip", "fund", "risk")
    }
    scores = combine_pillars(out, pillar_pct, cfg.get("weights", {}))
    return pd.concat([out, scores], axis=1)

# ---- 統一入口 ----

def build_scores(df: pd.DataFrame, cfg: dict, universe: str = "market_neutralized_by_industry") -> pd.DataFrame:
    out = df.copy()
    _get_industry_col(out, cfg)  # 確保有 industry 列
    use_pre = bool(cfg.get("use_prepercentiled", False))
    if use_pre:
        out = build_four_pillars_from_prepercentile(out, cfg)
    else:
        out = build_four_pillars_from_raw(out, cfg, universe)
    return out
