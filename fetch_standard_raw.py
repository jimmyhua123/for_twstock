# fetch_standard_raw.py
# -*- coding: utf-8 -*-
"""
抓 FinMind v4 各資料集，輸出為 clean_standardize 期望的「標準檔名」JSON（放在 raw 根目錄）：
- TaiwanStockInfo.json
- TaiwanStockTradingDate.json
- TaiwanStockPrice.json
- TaiwanStockTotalReturnIndex.json
- TaiwanStockMonthRevenue.json
- TaiwanStockFinancialStatements.json
- TaiwanStockBalanceSheet.json
- TaiwanStockCashFlowsStatement.json
- TaiwanStockShareholding.json
- TaiwanStockMarginPurchaseShortSale.json
- TaiwanStockBorrowingBalance.json
- TaiwanStockSecuritiesLending.json
- TaiwanStockInstitutionalInvestors.json

用法（PowerShell）：
$since = (Get-Date).AddDays(-800).ToString('yyyy-MM-dd')
$until = (Get-Date).ToString('yyyy-MM-dd')
python .\fetch_standard_raw.py `
  --watchlist .\watchlist.csv `
  --since $since `
  --until $until `
  --outdir finmind_raw_std `
  --sleep-ms 500

Token 來源：--token > 環境變數 FINMIND_TOKEN
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0"}

def parse_args():
    import argparse
    ap = argparse.ArgumentParser("Fetch FinMind v4 datasets into standard-named JSON files (root-level)")
    ap.add_argument("--watchlist", required=True, help="CSV，至少含 stock_id 欄（僅對需要 data_id 的 dataset 使用）")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD")
    ap.add_argument("--until", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="finmind_raw_std", help="輸出目錄（標準檔名 JSON 直接放根目錄）")
    ap.add_argument("--sleep-ms", type=int, default=500, help="每次 API 呼叫之間延遲（毫秒）")
    ap.add_argument("--token", default=None, help="FinMind token，未提供則讀 FINMIND_TOKEN")
    # 預設抓「細篩必備 + 籌碼/指數」的 12 組
    ap.add_argument("--datasets", nargs="*", default=[
        "TaiwanStockInfo",
        "TaiwanStockTradingDate",
        "TaiwanStockPrice",
        "TaiwanStockTotalReturnIndex",
        "TaiwanStockMonthRevenue",
        "TaiwanStockFinancialStatements",
        "TaiwanStockBalanceSheet",
        "TaiwanStockCashFlowsStatement",
        "TaiwanStockShareholding",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanStockBorrowingBalance",
        "TaiwanStockSecuritiesLending",
        "TaiwanStockInstitutionalInvestors",
    ])
    return ap.parse_args()

def finmind_get(dataset: str, data_id: Optional[str], start_date: str, end_date: str, token: str) -> pd.DataFrame:
    params = {"dataset": dataset, "start_date": start_date, "end_date": end_date, "token": token}
    if data_id:
        params["data_id"] = data_id
    r = requests.get(API, params=params, headers=UA, timeout=45)
    r.raise_for_status()
    obj = r.json()
    df = pd.DataFrame(obj.get("data", []))
    if df.empty:
        return df
    # 基本欄位正規化
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df

def main():
    args = parse_args()
    token = args.token or os.getenv("FINMIND_TOKEN")
    if not token:
        print("ERROR: 未提供 token，請加 --token 或設 FINMIND_TOKEN", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    wl = pd.read_csv(args.watchlist)
    # 容錯取得 stock_id 欄
    cols_lower = [c.strip().lower() for c in wl.columns]
    if "stock_id" not in cols_lower:
        raise SystemExit("watchlist 缺少 stock_id 欄")
    stock_ids: List[str] = (
        wl.iloc[:, cols_lower.index("stock_id")].astype(str).str.strip().dropna().unique().tolist()
    )

    # 哪些 dataset 需要 data_id
    needs_id: Dict[str, bool] = {
        "TaiwanStockInfo": False,
        "TaiwanStockTradingDate": False,
        "TaiwanStockPrice": True,
        "TaiwanStockTotalReturnIndex": False,
        "TaiwanStockMonthRevenue": True,
        "TaiwanStockFinancialStatements": True,
        "TaiwanStockBalanceSheet": True,
        "TaiwanStockCashFlowsStatement": True,
        "TaiwanStockShareholding": True,
        "TaiwanStockMarginPurchaseShortSale": True,
        "TaiwanStockBorrowingBalance": True,
        "TaiwanStockSecuritiesLending": True,
        "TaiwanStockInstitutionalInvestors": True,
    }

    # 輸出檔名（clean_standardize 會找這些）
    outfile_name: Dict[str, str] = {
        ds: f"{ds}.json" for ds in needs_id.keys()
    }

    sleep_s = max(0.0, args.sleep_ms / 1000.0)

    for ds in args.datasets:
        if ds not in needs_id:
            print(f"[跳過] 未支援的 dataset：{ds}")
            continue

        print(f"[開始] {ds}（{'逐檔' if needs_id[ds] else '一次性'}） {args.since} → {args.until}")
        parts: List[pd.DataFrame] = []

        if needs_id[ds]:
            for sid in stock_ids:
                try:
                    df = finmind_get(ds, sid, args.since, args.until, token)
                    if not df.empty:
                        parts.append(df)
                except Exception as e:
                    print(f"[警告] {ds} {sid} 抓取失敗：{e}")
                time.sleep(sleep_s)
        else:
            try:
                df = finmind_get(ds, None, args.since, args.until, token)
                if not df.empty:
                    parts.append(df)
            except Exception as e:
                print(f"[警告] {ds} 抓取失敗：{e}")

        if parts:
            all_df = pd.concat(parts, ignore_index=True)
            # 去重保險
            subset = ["date", "stock_id"] if "stock_id" in all_df.columns else ["date"]
            all_df = all_df.drop_duplicates(subset=[c for c in subset if c in all_df.columns])
            # 輸出為 JSON 陣列
            outpath = outdir / outfile_name[ds]
            outpath.write_text(all_df.to_json(orient="records", force_ascii=False), encoding="utf-8")
            rows = len(all_df)
            print(f"[完成] {ds} → {outpath}（{rows:,} 列）")
        else:
            print(f"[略過] {ds}（本次沒有資料）")

    print("[ALL DONE] 請接著執行：python finmind_clean_standardize.py --raw-dir", str(outdir))

if __name__ == "__main__":
    main()
