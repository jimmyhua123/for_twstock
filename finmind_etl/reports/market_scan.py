from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..scoring_core import build_scores

KEEP = ["stock_id","stock_name","industry","score_total","tech_score","chip_score","fund_score","risk_score"]

def run_market_scan(features_df: pd.DataFrame, config: dict, universe: str | None, out_dir: str) -> dict:
    outdir = Path(out_dir); outdir.mkdir(parents=True, exist_ok=True)
    feats = features_df.copy()
    uni = universe or config.get("universe", "market_neutralized_by_industry")
    scored = build_scores(feats, config, universe=uni)
    # 輸出
    csv_path = outdir / "market_scan_scores.csv"
    cols = [c for c in KEEP if c in scored.columns]
    scored.loc[:, cols].sort_values("score_total", ascending=False).to_csv(csv_path, index=False, encoding="utf-8")
    # 簡易 MD
    md_path = outdir / "market_scan_report.md"
    top = scored.loc[:, cols].sort_values("score_total", ascending=False).head(50)
    lines = ["# Market Scan (Top 50)\n"]
    for _, r in top.iterrows():
        lines.append(f"- {r.get('stock_id','?')} {r.get('stock_name','?')} [{r.get('industry','?')}] → {int(round(r.get('score_total', float('nan')) or 0,0))}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}
