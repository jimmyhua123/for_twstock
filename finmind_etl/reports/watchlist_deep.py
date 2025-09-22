from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..scoring_core import build_scores

KEEP = ["stock_id","stock_name","industry","score_total","tech_score","chip_score","fund_score","risk_score"]

def run_watchlist_report(features_df: pd.DataFrame, config: dict, out_dir: str) -> dict:
    outdir = Path(out_dir); outdir.mkdir(parents=True, exist_ok=True)
    uni = config.get("universe", "watchlist")
    scored = build_scores(features_df.copy(), config, universe=uni)
    csv_path = outdir / "watchlist_scores.csv"
    cols = [c for c in KEEP if c in scored.columns]
    scored.loc[:, cols].sort_values("score_total", ascending=False).to_csv(csv_path, index=False, encoding="utf-8")
    md_path = outdir / "watchlist_report.md"
    lines = ["# Watchlist Deep Report\n"]
    for _, r in scored.loc[:, cols].sort_values("score_total", ascending=False).iterrows():
        lines.append(f"## {r.get('stock_id','?')} {r.get('stock_name','?')}  (Industry: {r.get('industry','?')})")
        lines.append(f"總分：{int(round(r.get('score_total', float('nan')) or 0,0))}，技術/籌碼/基本/風險：{(r.get('tech_score', float('nan')) or float('nan')):.1f}/{(r.get('chip_score', float('nan')) or float('nan')):.1f}/{(r.get('fund_score', float('nan')) or float('nan')):.1f}/{(r.get('risk_score', float('nan')) or float('nan')):.1f}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}
