# -*- coding: utf-8 -*-
"""
抓 FinMind v4 資料，輸出為 clean_standardize 期望的「標準檔名」JSON：
- TaiwanStockInfo.json
- TaiwanStockTradingDate.json
- TaiwanStockPrice.json
- TaiwanStockMonthRevenue.json
- TaiwanStockFinancialStatements.json
- TaiwanStockBalanceSheet.json
- TaiwanStockCashFlowsStatement.json

用法範例（PowerShell）：
$since = (Get-Date).AddDays(-800).ToString('yyyy-MM-dd')
$until = (Get-Date).ToString('yyyy-MM-dd')
python .\fetch_standard_raw.py ^
  --watchlist .\watchlist.csv ^
  --since $since ^
  --until $until ^
  --outdir finmind_raw_std ^
  --sleep-ms 500

Token 來源順序：--token > 環境變數 FINMIND_TOKEN
"""
from __future__ import annotations
import os, sys, time, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0"}

# --------- 參數解析 ---------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser("Fetch FinMind v4 datasets into standard-named JSON files")
    ap.add_argument("--watchlist", required=True, help="CSV 檔，至少含 stock_id 欄（只對需要 data_id 的 dataset 使用）")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD（建議 >= 18~24 個月前，YoY 至少 13 個月）")
    ap.add_argument("--until", required=True, help="YYYY-MM-DD（通常用今天）")
    ap.add_argument("--outdir", default="finmind_raw_std", help="輸出資料夾（根目錄放標準檔名 JSON）")
    ap.add_argument("--sleep-ms", type=int, default=500, help="每次 API 呼叫之間的延遲毫秒（配額保險）")
    ap.add_argument("--token", default=None, help="FinMind token；未提供則讀環境變數 FINMIND_TOKEN")
    # 可選擇要抓哪些 dataset（預設抓最必要的 7 組）
    ap.add_argument("--datasets", nargs="*", default=[
        "TaiwanStockInfo",
        "TaiwanStockTradingDate",
        "TaiwanStockPrice",
        "TaiwanStockMonthRevenue",
        "TaiwanStockFinancialStatements",
        "TaiwanStockBalanceSheet",
        "TaiwanStockCashFlowsStatement",
    ])
    return ap.parse_args()

# --------- API 呼叫 ---------
def finmind_get(dataset: str, data_id: Optional[str], start_date: str, end_date: str, token: str) -> pd.DataFrame:
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
    }
    if data_id:
        params["data_id"] = data_id
    r = requests.get(API, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict) or "data" not in obj:
        return pd.DataFrame()
    df = pd.DataFrame(obj["data"])
    # 統一基本欄位命名（若存在）
    for c in ("date", "stock_id"):
        if c in df.columns:
            # 確保 date 是字串 YYYY-MM-DD；stock_id 是字串
            if c == "date":
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["stock_id"] = df["stock_id"].astype(str)
    return df

# --------- 主流程 ---------
def main():
    args = parse_args()
    token = args.token or os.getenv("FINMIND_TOKEN")
    if not token:
        print("ERROR: 未提供 token，請加 --token 或設定環境變數 FINMIND_TOKEN", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 讀取 watchlist（只用於需要 data_id 的 dataset）
    wl = pd.read_csv(args.watchlist)
    wl_cols = [c.strip().lower() for c in wl.columns]
    if "stock_id" not in wl_cols:
        raise SystemExit("watchlist 缺少 stock_id 欄")
    # 轉成唯一字串列表
    stock_ids: List[str] = (
        wl.loc[:, wl.columns[wl_cols.index("stock_id")]]
        .astype(str).str.strip().dropna().unique().tolist()
    )

    sleep_s = max(0.0, args.sleep_ms / 1000.0)

    # 定義各 dataset 是否需要 data_id（逐檔迭代 vs 一次性）
    needs_id = {
        "TaiwanStockInfo": False,
        "TaiwanStockTradingDate": False,
        "TaiwanStockPrice": True,
        "TaiwanStockMonthRevenue": True,
        "TaiwanStockFinancialStatements": True,
        "TaiwanStockBalanceSheet": True,
        "TaiwanStockCashFlowsStatement": True,
    }

    # 輸出檔名（clean_standardize 會找這些）
    outfile_name = {
        "TaiwanStockInfo": "TaiwanStockInfo.json",
        "TaiwanStockTradingDate": "TaiwanStockTradingDate.json",
        "TaiwanStockPrice": "TaiwanStockPrice.json",
        "TaiwanStockMonthRevenue": "TaiwanStockMonthRevenue.json",
        "TaiwanStockFinancialStatements": "TaiwanStockFinancialStatements.json",
        "TaiwanStockBalanceSheet": "TaiwanStockBalanceSheet.json",
        "TaiwanStockCashFlowsStatement": "TaiwanStockCashFlowsStatement.json",
    }

    # 依序抓取 & 合併寫檔
    for ds in args.datasets:
        if ds not in needs_id:
            print(f"[跳過] 未支援的 dataset：{ds}")
            continue

        print(f"[開始] 抓取 {ds}（{'逐檔' if needs_id[ds] else '一次性'}）")
        parts: List[pd.DataFrame] = []

        if needs_id[ds]:
            for sid in stock_ids:
                try:
                    df = finmind_get(ds, sid, args.since, args.until, token)
                except Exception as e:
                    print(f"[警告] {ds} {sid} 抓取失敗：{e}")
                    df = pd.DataFrame()
                if not df.empty:
                    parts.append(df)
                time.sleep(sleep_s)
        else:
            try:
                df = finmind_get(ds, None, args.since, args.until, token)
            except Exception as e:
                print(f"[警告] {ds} 抓取失敗：{e}")
                df = pd.DataFrame()
            if not df.empty:
                parts.append(df)

        if parts:
            all_df = pd.concat(parts, ignore_index=True)
            # 寫成 JSON 陣列（clean_standardize 會找這個檔名）
            outpath = outdir / outfile_name[ds]
            outpath.write_text(all_df.to_json(orient="records", force_ascii=False), encoding="utf-8")
            print(f"[完成] {ds} → {outpath}（{len(all_df):,} 列）")
        else:
            print(f"[略過] {ds}（本次沒有資料）")

    print("[ALL DONE] 請接著執行 finmind_clean_standardize.py --raw-dir", str(outdir))

if __name__ == "__main__":
    main()
