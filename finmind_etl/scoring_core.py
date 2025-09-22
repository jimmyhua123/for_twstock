from __future__ import annotations

import numpy as np
import pandas as pd


def mad(x: pd.Series) -> float:
    med = x.median()
    return (x - med).abs().median() or 1e-9


def industry_neutralize(df: pd.DataFrame, cols: list[str], ind_col: str = "industry") -> pd.DataFrame:
    if ind_col not in df.columns:
        return df
    g = df.groupby(ind_col)
    for c in cols:
        if c not in df.columns:
            continue
        med = g[c].transform("median")
        m = g[c].transform(mad).replace(0, 1e-9)
        df[c + "_adj"] = (df[c] - med) / m
    return df


def winsorize(df: pd.DataFrame, cols: list[str], lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        ql = df[c].quantile(lo)
        qh = df[c].quantile(hi)
        df[c] = df[c].clip(ql, qh)
    return df


def to_percentile(df: pd.DataFrame, cols: list[str], by: str | None = None) -> pd.DataFrame:
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df
    if by and by in df.columns:
        g = df.groupby(by)
        for c in valid_cols:
            df[c + "_pct"] = g[c].rank(pct=True).mul(100)
    else:
        for c in valid_cols:
            df[c + "_pct"] = df[c].rank(pct=True).mul(100)
    return df


def combine_four_pillars(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = config.get("weights", {})
    feats = config.get("features", {})
    for pillar in ("tech", "chip", "fund", "risk"):
        cols = [f + "_pct" for f in feats.get(pillar, []) if f + "_pct" in df.columns]
        if cols:
            df[pillar + "_score"] = df[cols].mean(axis=1)
        else:
            df[pillar + "_score"] = np.nan
    df["score_total"] = (
        df.get("tech_score", np.nan) * weights.get("tech", 0)
        + df.get("chip_score", np.nan) * weights.get("chip", 0)
        + df.get("fund_score", np.nan) * weights.get("fund", 0)
        + df.get("risk_score", np.nan) * weights.get("risk", 0)
    )
    return df
