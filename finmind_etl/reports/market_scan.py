from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..scoring_core import combine_four_pillars, industry_neutralize, to_percentile, winsorize


def _flatten_features(features_config: dict) -> list[str]:
    cols: list[str] = []
    for values in features_config.values():
        cols.extend(values)
    return cols


def _ensure_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "industry" not in df.columns and "industry_category" in df.columns:
        df["industry"] = df["industry_category"]
    if "stock_name" not in df.columns and "name" in df.columns:
        df["stock_name"] = df["name"]
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    if "industry" not in df.columns:
        df["industry"] = ""
    if "stock_name" not in df.columns:
        df["stock_name"] = ""
    return df


def _rename_adj_percentiles(df: pd.DataFrame, feature_cols: Iterable[str]) -> None:
    for col in feature_cols:
        pct_col = f"{col}_adj_pct"
        if pct_col in df.columns:
            df[f"{col}_pct"] = df[pct_col]


def run_market_scan(features_df: pd.DataFrame, config: dict, universe: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = features_df.copy()
    feats = _ensure_basic_columns(feats)

    feature_cols = _flatten_features(config.get("features", {}))
    existing_cols = [c for c in feature_cols if c in feats.columns]

    norm_cfg = config.get("normalization", {})
    if norm_cfg and existing_cols:
        winsor_lo, winsor_hi = norm_cfg.get("winsorize_pct", [0.01, 0.99])
        feats = winsorize(feats, existing_cols, winsor_lo, winsor_hi)

    if universe == "market_neutralized_by_industry":
        feats = industry_neutralize(feats, existing_cols, ind_col="industry")
        adj_cols = [f"{c}_adj" for c in existing_cols if f"{c}_adj" in feats.columns]
        feats = to_percentile(feats, adj_cols, by=None)
        _rename_adj_percentiles(feats, existing_cols)
    elif universe == "industry":
        feats = to_percentile(feats, existing_cols, by="industry")
    else:
        feats = to_percentile(feats, existing_cols, by=None)

    feats = combine_four_pillars(feats, config)

    keep_cols = [
        "stock_id",
        "stock_name",
        "industry",
        "score_total",
        "tech_score",
        "chip_score",
        "fund_score",
        "risk_score",
    ]
    available_cols = [c for c in keep_cols if c in feats.columns]
    if "score_total" not in available_cols and "score_total" in feats.columns:
        available_cols.append("score_total")
    result_df = feats[available_cols].sort_values("score_total", ascending=False)

    csv_path = out / "market_scan_scores.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = out / "market_scan_report.md"
    lines = ["# Market Scan (Top 50)\n"]
    top_rows = result_df.head(50)
    for _, row in top_rows.iterrows():
        stock_id = row.get("stock_id", "")
        stock_name = row.get("stock_name", "")
        industry = row.get("industry", "")
        total = row.get("score_total")
        display = "NA" if pd.isna(total) else int(round(float(total)))
        lines.append(f"- {stock_id} {stock_name} [{industry}] → {display}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"csv": str(csv_path), "md": str(md_path)}
