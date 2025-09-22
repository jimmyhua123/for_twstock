# -*- coding: utf-8 -*-
"""
把 FinMind 的 TaiwanStockInfo.json 轉成：
1) finmind_in/universe_all.csv        # stock_id, stock_name, industry, market, is_active
2) finmind_in/stocks_all.csv          # 一欄 stock_id（全市場）
3) finmind_in/stocks_all.txt          # 逗號分隔 stock_id 字串（方便 --stocks 用）
4) finmind_in/batches/batch_XXX.csv   # 每批 N 檔，方便分批抓 RAW
使用方式：
  python tools/make_universe_all.py --input finmind_raw/TaiwanStockInfo.json --out finmind_in --batch-size 200
"""
from __future__ import annotations
import json, argparse, math, re
from pathlib import Path
import pandas as pd

def load_info(path: Path) -> pd.DataFrame:
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(js, dict) and "data" in js:
        data = js["data"]
    elif isinstance(js, list):
        data = js
    else:
        raise SystemExit("無法解析 TaiwanStockInfo.json（頂層非 list，也沒有 data）")
    df = pd.DataFrame(data)
    # 標準欄位對齊
    def pick(*names, default=None):
        for n in names:
            if n in df.columns: return n
        return None
    col_id   = pick("stock_id","code","證券代號")
    col_name = pick("stock_name","name","證券名稱")
    col_ind  = pick("industry_category","industry","產業別")
    col_mkt  = pick("type","market","exchange","上市別")

    df = df.rename(columns={
        col_id: "stock_id",
        col_name: "stock_name",
        col_ind: "industry",
        col_mkt: "market",
    })

    # 僅保留 4 碼數字（傳統台股個股），排除 ETF/ETN/權證/債券等
    df = df[df["stock_id"].astype(str).str.fullmatch(r"\d{4}")]

    # 市場：只留 上市/上櫃（中英文皆可）
    def is_equity_market(x: str) -> bool:
        if not isinstance(x, str): return False
        x = x.upper()
        return any(k in x for k in ["上市","上櫃","TWSE","TSE","TPEx","OTC"])
    if "market" in df.columns:
        df = df[df["market"].apply(is_equity_market)]

    # 狀態欄若存在則排除下市
    for status_col in ["status","listed_status","上巿別","備註"]:
        if status_col in df.columns:
            bad = df[status_col].astype(str).str.contains("下市|終止|停止", regex=True)
            df = df[~bad]

    df = df[["stock_id","stock_name","industry","market"]].drop_duplicates("stock_id")
    df["is_active"] = True
    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TaiwanStockInfo.json 路徑")
    ap.add_argument("--out", default="finmind_in", help="輸出資料夾")
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_info(Path(args.input))

    # 1) universe_all.csv
    uni_csv = out / "universe_all.csv"
    df.to_csv(uni_csv, index=False, encoding="utf-8")
    print(f"[OK] {uni_csv} → {len(df)} 檔")

    # 2) stocks_all.csv / stocks_all.txt
    stocks_csv = out / "stocks_all.csv"
    df[["stock_id"]].to_csv(stocks_csv, index=False, encoding="utf-8")
    stocks_txt = out / "stocks_all.txt"
    stocks_txt.write_text(",".join(df["stock_id"].astype(str).tolist()), encoding="utf-8")
    print(f"[OK] {stocks_csv}, {stocks_txt}")

    # 3) batches
    bdir = out / "batches"; bdir.mkdir(exist_ok=True)
    n = len(df)
    bs = args.batch_size
    for i in range(math.ceil(n/bs)):
        chunk = df.iloc[i*bs:(i+1)*bs][["stock_id"]]
        (bdir / f"batch_{i+1:03d}.csv").write_text(chunk.to_csv(index=False), encoding="utf-8")
    print(f"[OK] 批次輸出：{len(list(bdir.glob('batch_*.csv')))} 個，每批 {bs} 檔（最後一批可能不足）")

if __name__ == "__main__":
    main()
