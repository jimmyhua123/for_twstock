from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..scoring_core import combine_four_pillars, to_percentile, winsorize


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


def run_watchlist_report(features_df: pd.DataFrame, config: dict, out_dir: str) -> dict:
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

    csv_path = out / "watchlist_scores.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = out / "watchlist_report.md"
    lines = ["# Watchlist Deep Report\n"]
    for _, row in result_df.iterrows():
        stock_id = row.get("stock_id", "")
        stock_name = row.get("stock_name", "")
        industry = row.get("industry", "")
        total = row.get("score_total")
        tech = row.get("tech_score")
        chip = row.get("chip_score")
        fund = row.get("fund_score")
        risk = row.get("risk_score")
        total_display = "NA" if pd.isna(total) else int(round(float(total)))
        tech_display = "NA" if pd.isna(tech) else f"{float(tech):.1f}"
        chip_display = "NA" if pd.isna(chip) else f"{float(chip):.1f}"
        fund_display = "NA" if pd.isna(fund) else f"{float(fund):.1f}"
        risk_display = "NA" if pd.isna(risk) else f"{float(risk):.1f}"
        lines.append(f"## {stock_id} {stock_name}  (Industry: {industry})")
        lines.append(
            "總分：{total}，技術/籌碼/基本/風險：{tech}/{chip}/{fund}/{risk}\n".format(
                total=total_display,
                tech=tech_display,
                chip=chip_display,
                fund=fund_display,
                risk=risk_display,
            )
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"csv": str(csv_path), "md": str(md_path)}
