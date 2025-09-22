# filter_watchlist.py
# 依條件過濾 scores_watchlist.csv（可合併 features_snapshot.csv 取得日期後去重）
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def log(m): print(m, flush=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[錯誤] 找不到檔案：{path}")
    # dtype & 編碼相容 UTF-8-SIG
    return pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser(description="Filter scores_watchlist by conditions")
    ap.add_argument("--scores", default="finmind_scores/scores_watchlist.csv", help="scores_watchlist.csv 路徑")
    ap.add_argument("--snapshot", default="finmind_scores/features_snapshot.csv", help="features_snapshot.csv 路徑（用來帶入 date 與去重）")
    ap.add_argument("--asof", type=str, default=None, help="只保留此日期（YYYY-MM-DD）；未提供則取 snapshot 內最新日期")
    ap.add_argument("--min-total", type=float, default=70.0, help="score_total 下限（預設 70）")
    ap.add_argument("--min-tech", type=float, default=60.0, help="score_tech 下限（預設 60）")
    ap.add_argument("--min-chip", type=float, default=60.0, help="score_chip 下限（預設 60）")
    ap.add_argument("--min-risk", type=float, default=40.0, help="score_risk 下限（預設 40）")
    ap.add_argument("--excess-positive", action="store_true", help="要求 excess_ret_20d > 0")
    ap.add_argument("--top", type=int, default=0, help="只輸出前 N 名（0 表示不限制）")
    ap.add_argument("--out", default="finmind_scores/watchlist_filtered.csv", help="輸出檔路徑（UTF-8-SIG）")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    snap_path   = Path(args.snapshot)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_csv(scores_path)

    # 盡量補上日期：若 watchlist 沒有 date 欄，從 snapshot 併入
    if "date" not in df.columns and snap_path.exists():
        snap = read_csv(snap_path)[["stock_id","date"]].copy()
        # 取各檔最新日期
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce")
        snap = snap.sort_values(["stock_id","date"]).drop_duplicates("stock_id", keep="last")
        df = df.merge(snap, on="stock_id", how="left")

    # 若仍無 date 欄，建一個空的，避免後續報錯
    if "date" not in df.columns:
        df["date"] = pd.NaT

    # 去重（同一 stock_id 多列時，保留最新 date）
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["stock_id","date"]).drop_duplicates("stock_id", keep="last")

    # 評分日期過濾
    if args.asof:
        asof = pd.to_datetime(args.asof)
    else:
        # 未指定則用資料內最大日期
        asof = df["date"].max()
    if pd.notna(asof):
        df = df[df["date"] == asof]

    # 條件過濾（欄位缺失就視為不過濾那一項）
    def has(col): return col in df.columns

    if has("score_total"): df = df[df["score_total"] >= args.min_total]
    if has("score_tech"):  df = df[df["score_tech"]  >= args.min_tech]
    if has("score_chip"):  df = df[df["score_chip"]  >= args.min_chip]
    if has("score_risk"):  df = df[df["score_risk"]  >= args.min_risk]
    if args.excess_positive and has("excess_ret_20d"):
        df = df[df["excess_ret_20d"] > 0]

    # 排序
    sort_cols = [c for c in ["score_total","score_tech","score_chip"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    # 只留 Top N
    if args.top and args.top > 0:
        df = df.head(args.top)

    # 輸出 UTF-8-SIG
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[完成] 已輸出：{out_path}（{len(df)} 筆；asof={asof.date() if pd.notna(asof) else 'N/A'}）")

if __name__ == "__main__":
    main()
