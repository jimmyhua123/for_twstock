# fetch_standard_raw.py
# -*- coding: utf-8 -*-
"""
抓 FinMind v4 各資料集，輸出為 clean_standardize 期望的「根目錄標準檔名」JSON。
支援：配額觸頂 (429/訊息) 或自設 --max-calls 時，寫入 _fetch_state.json 可續跑；
逐檔資料先存 _parts/<dataset>/<stock_id>.csv，結束後整併成 <DatasetName>.json。
"""
from __future__ import annotations
import os, sys, time, json, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0"}

# ---------------------- 參數 ----------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser("Fetch FinMind datasets into standard-named JSON (root-level) with resume")
    ap.add_argument("--watchlist", required=True, help="CSV，至少含 stock_id 欄（僅對需要 data_id 的 dataset 使用）")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD（建議 >= 18~24 個月）")
    ap.add_argument("--until", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="finmind_raw_std", help="輸出資料夾（根目錄放標準檔名 JSON）")
    ap.add_argument("--sleep-ms", type=int, default=500, help="每次 API 呼叫間隔（毫秒）")
    ap.add_argument("--token", default=None, help="FinMind token，未提供則讀 FINMIND_TOKEN")
    ap.add_argument("--max-calls", type=int, default=0, help="單次執行最多 API 呼叫數（0=不限制）")
    ap.add_argument("--no-resume", action="store_true", help="忽略既有狀態檔，從頭開始")
    # ✅ 預設清單（依你提供）：技術、籌碼、基本面 + 市場指數與 PER
    ap.add_argument("--datasets", nargs="*", default=[
        # 技術面
        "TaiwanStockPrice",
        "TaiwanStockPER",
        "TaiwanStockDayTrading",
        # 籌碼面（個股）
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockShareholding",
        "TaiwanStockSecuritiesLending",
        "TaiwanStockMarginShortSaleSuspension",
        "TaiwanDailyShortSaleBalances",
        # 基本面（個股）
        "TaiwanStockFinancialStatements",
        "TaiwanStockBalanceSheet",
        "TaiwanStockCashFlowsStatement",
        "TaiwanStockDividend",
        "TaiwanStockDividendResult",
        "TaiwanStockMonthRevenue",
        # 其他（維度/市場）
        "TaiwanStockInfo",
        "TaiwanStockTradingDate",
        "TaiwanStockTotalReturnIndex",
    ])
    return ap.parse_args()

# ---------------------- 工具 ----------------------
def wl_ids(path: str) -> List[str]:
    wl = pd.read_csv(path)
    cols_lower = [c.strip().lower() for c in wl.columns]
    if "stock_id" not in cols_lower:
        raise SystemExit("watchlist 缺少 stock_id 欄")
    return (wl.iloc[:, cols_lower.index("stock_id")]
              .astype(str).str.strip().dropna().unique().tolist())

def finmind_get(dataset: str, data_id: Optional[str], start_date: str, end_date: str, token: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """回傳 (df, err_reason). err_reason: 'rate_limit' | 'http_error' | None"""
    params = {"dataset": dataset, "start_date": start_date, "end_date": end_date, "token": token}
    if data_id:
        params["data_id"] = data_id
    try:
        r = requests.get(API, params=params, headers=UA, timeout=45)
        if r.status_code == 429:
            return pd.DataFrame(), "rate_limit"
        r.raise_for_status()
        obj = r.json()
    except requests.exceptions.HTTPError:
        return pd.DataFrame(), "http_error"
    except Exception:
        return pd.DataFrame(), "http_error"

    msg = str((obj or {}).get("msg", "")).lower()
    if any(k in msg for k in ["too many", "rate limit", "exceed", "limit"]):
        return pd.DataFrame(), "rate_limit"

    df = pd.DataFrame((obj or {}).get("data", []))
    if df.empty:
        return df, None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df, None

def hash_watchlist(ids: List[str]) -> str:
    return hashlib.md5((",".join(sorted(ids))).encode("utf-8")).hexdigest()

# ---------------------- 狀態 ----------------------
def load_state(p: Path) -> dict:
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}
def save_state(p: Path, payload: dict) -> None:
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------- 主流程 ----------------------
def main():
    args = parse_args()
    token = args.token or os.getenv("FINMIND_TOKEN")
    if not token:
        print("ERROR: 未提供 token，請加 --token 或設 FINMIND_TOKEN", file=sys.stderr); sys.exit(2)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "_fetch_state.json"

    ids = wl_ids(args.watchlist)
    wl_hash = hash_watchlist(ids)

    # 哪些 dataset 需要 data_id（逐檔呼叫）
    needs_id: Dict[str, bool] = {
        # 技術面
        "TaiwanStockPrice": True,
        "TaiwanStockPER": True,             # 原生就讀 raw/TaiwanStockPER.json（scoring 會用）
        "TaiwanStockDayTrading": True,
        # 籌碼面
        "TaiwanStockMarginPurchaseShortSale": True,
        "TaiwanStockInstitutionalInvestorsBuySell": True,
        "TaiwanStockShareholding": True,
        "TaiwanStockSecuritiesLending": True,
        "TaiwanStockMarginShortSaleSuspension": True,
        "TaiwanDailyShortSaleBalances": True,   # ← 取代錯誤的 BorrowingBalance
        # 基本面
        "TaiwanStockFinancialStatements": True,
        "TaiwanStockBalanceSheet": True,
        "TaiwanStockCashFlowsStatement": True,
        "TaiwanStockDividend": True,
        "TaiwanStockDividendResult": True,
        "TaiwanStockMonthRevenue": True,
        # 其他（一次性）
        "TaiwanStockInfo": False,
        "TaiwanStockTradingDate": False,
        "TaiwanStockTotalReturnIndex": False,
    }

    # 根目錄輸出檔名（清理腳本會找這些）
    outfile_name: Dict[str, str] = {ds: f"{ds}.json" for ds in needs_id.keys()}

    # 載入/初始化狀態
    st = {} if args.no_resume else load_state(state_path)
    cfg_key = {"since": args.since, "until": args.until, "datasets": args.datasets, "watchlist_md5": wl_hash}
    if not st or st.get("config") != cfg_key:
        st = {"config": cfg_key, "progress": {"dataset_idx": 0, "id_idx": 0},
              "used_calls": 0, "stopped": False, "stopped_reason": None, "last_msg": None}
    used_calls = int(st.get("used_calls", 0))
    ds_start_idx = int(st["progress"].get("dataset_idx", 0))
    id_start_idx = int(st["progress"].get("id_idx", 0))
    sleep_s = max(0.0, args.sleep_ms / 1000.0)

    for ds_i in range(ds_start_idx, len(args.datasets)):
        ds = args.datasets[ds_i]
        if ds not in needs_id:
            print(f"[跳過] 未支援的 dataset：{ds}")
            continue

        print(f"[開始] {ds}（{'逐檔' if needs_id[ds] else '一次性'}） {args.since} → {args.until}  [from {id_start_idx if needs_id[ds] else '-'}]")
        parts_dir = outdir / "_parts" / ds
        parts_dir.mkdir(parents=True, exist_ok=True)

        # 不需 data_id → 一次性
        if not needs_id[ds]:
            outpath = outdir / outfile_name[ds]
            if outpath.exists():
                print(f"[略過] {ds} 標準檔已存在：{outpath.name}")
            else:
                df, err = finmind_get(ds, None, args.since, args.until, token)
                if err == "rate_limit":
                    st.update({"progress": {"dataset_idx": ds_i, "id_idx": 0}, "used_calls": used_calls,
                               "stopped": True, "stopped_reason": "rate_limit"})
                    save_state(state_path, st); print("[停止] 觸發 API 上限。已保存進度。"); return
                used_calls += 1 if err is None else 0
                if args.max_calls and used_calls >= args.max_calls:
                    st.update({"progress": {"dataset_idx": ds_i, "id_idx": 0}, "used_calls": used_calls,
                               "stopped": True, "stopped_reason": "max_calls"})
                    save_state(state_path, st); print("[停止] 已達 --max-calls。已保存進度。"); return
                if not df.empty:
                    outpath.write_text(df.to_json(orient="records", force_ascii=False), encoding="utf-8")
                    print(f"[完成] {ds} → {outpath}（{len(df):,} 列）")
                else:
                    print(f"[略過] {ds}（本次沒有資料）")
            id_start_idx = 0
            st.update({"progress": {"dataset_idx": ds_i + 1, "id_idx": 0}, "used_calls": used_calls})
            save_state(state_path, st)
            continue

        # 需 data_id → 逐檔
        for j in range(id_start_idx, len(ids)):
            sid = ids[j]
            part_file = parts_dir / f"{sid}.csv"
            if part_file.exists() and part_file.stat().st_size > 0:
                st.update({"progress": {"dataset_idx": ds_i, "id_idx": j + 1}, "used_calls": used_calls})
                save_state(state_path, st); continue

            df, err = finmind_get(ds, sid, args.since, args.until, token)
            if err == "rate_limit":
                st.update({"progress": {"dataset_idx": ds_i, "id_idx": j}, "used_calls": used_calls,
                           "stopped": True, "stopped_reason": "rate_limit", "last_msg": f"{ds}/{sid} rate limited"})
                save_state(state_path, st); print(f"[停止] 觸發 API 上限於 {ds}/{sid}。已保存進度。"); return
            if err == "http_error":
                print(f"[警告] {ds} {sid} 抓取失敗（HTTP）。略過此檔。")
                st.update({"progress": {"dataset_idx": ds_i, "id_idx": j + 1}, "used_calls": used_calls})
                save_state(state_path, st); time.sleep(sleep_s); continue

            used_calls += 1
            if not df.empty:
                df.to_csv(part_file, index=False, encoding="utf-8")

            st.update({"progress": {"dataset_idx": ds_i, "id_idx": j + 1}, "used_calls": used_calls})
            save_state(state_path, st)

            if args.max_calls and used_calls >= args.max_calls:
                st.update({"stopped": True, "stopped_reason": "max_calls", "last_msg": f"hit max_calls at {ds}/{sid}"})
                save_state(state_path, st); print("[停止] 已達 --max-calls。已保存進度。"); return

            time.sleep(sleep_s)

        # 整併分片 → 根目錄標準檔
        outpath = outdir / outfile_name[ds]
        if not outpath.exists():
            chunks = []
            for f in (outdir / "_parts" / ds).glob("*.csv"):
                try: chunks.append(pd.read_csv(f))
                except Exception: pass
            if chunks:
                all_df = pd.concat(chunks, ignore_index=True)
                subset = ["date", "stock_id"] if "stock_id" in all_df.columns else ["date"]
                all_df = all_df.drop_duplicates(subset=[c for c in subset if c in all_df.columns])
                outpath.write_text(all_df.to_json(orient="records", force_ascii=False), encoding="utf-8")
                print(f"[完成] {ds} → {outpath}（{len(all_df):,} 列，整併 {len(chunks)} 片）")
            else:
                print(f"[略過] {ds}（沒有有效分片可整併）")

        # 兼容：把 BuySell 另存同內容為 TaiwanStockInstitutionalInvestors.json（部分 clean/舊腳本會找這個檔名）
        if ds == "TaiwanStockInstitutionalInvestorsBuySell":
            alias = outdir / "TaiwanStockInstitutionalInvestors.json"
            bs = outdir / "TaiwanStockInstitutionalInvestorsBuySell.json"
            if bs.exists():
                alias.write_text(bs.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[別名] 另存一份 → {alias.name}（與 BuySell 相同內容）")

        id_start_idx = 0
        st.update({"progress": {"dataset_idx": ds_i + 1, "id_idx": 0}, "used_calls": used_calls})
        save_state(state_path, st)

    print(f"[ALL DONE] 本次呼叫數：{used_calls} 次。接著可執行：python finmind_clean_standardize.py --raw-dir {outdir}")

if __name__ == "__main__":
    main()
