# finmind_clean_standardize.py
# ------------------------------------------------------------
# 把 FinMind 抓下來的 raw JSON 標準化為分析友善的 CSV
# 產出內容見下方 EXPORTS 清單
# Python 3.10+，需要：pandas
# ------------------------------------------------------------
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np


EXPORTS_HINT = """
將輸出下列檔案（若對應原始 JSON 缺失/無資料則跳過）：
- dim_security.csv
- dim_calendar.csv
- fact_price_daily.csv
- fact_market_return.csv
- fact_institutional_flow_daily.csv
- fact_margin_short_daily.csv
- fact_short_balances_daily.csv
- fact_securities_lending.csv
- fact_foreign_shareholding_daily.csv
- fact_revenue_monthly.csv
- fact_financials_income_stmt_quarterly.csv
- fact_financials_balance_sheet_quarterly.csv
- fact_financials_cashflows_quarterly.csv
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json_df(path: Path) -> pd.DataFrame:
    """讀取 JSON（支援 {data: [...]} 或直接是 list），無檔或空資料回傳空 DataFrame。"""
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        data = obj.get("data", [])
    elif isinstance(obj, list):
        data = obj
    else:
        data = []
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> bool:
    if df is None or df.empty:
        return False
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[輸出] {path.name:50s} rows={len(df):6d}")
    return True


def latest_per_stock(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    dff = df.sort_values(["stock_id", date_col]).drop_duplicates("stock_id", keep="last")
    return dff


def build_dim_security(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockInfo.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] dim_security（TaiwanStockInfo.json 不存在或無資料）")
        return None
    # 取每檔最新一筆
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
        df = df.sort_values(["stock_id", "date"]).drop_duplicates("stock_id", keep="last")
    cols = [c for c in ["stock_id", "stock_name", "industry_category", "type", "date"] if c in df.columns]
    df = df[cols]
    out = out_dir / "dim_security.csv"
    save_csv(df, out)
    return out


def build_dim_calendar(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockTradingDate.json"
    df = load_json_df(src)
    if df.empty or "date" not in df.columns:
        log("[略過] dim_calendar（TaiwanStockTradingDate.json 缺或無 date 欄）")
        return None
    df["date"] = to_dt(df["date"])
    df = df.drop_duplicates().sort_values("date")
    df["is_trading_day"] = True
    out = out_dir / "dim_calendar.csv"
    save_csv(df, out)
    return out


def build_fact_price(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockPrice.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_price_daily（TaiwanStockPrice.json 無資料）")
        return None
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    df = df.sort_values(["stock_id", "date"])
    if "close" in df.columns:
        df["ret_1d"] = df.groupby("stock_id")["close"].pct_change()
        df["ret_5d"] = df.groupby("stock_id")["close"].pct_change(5)
        df["ret_20d"] = df.groupby("stock_id")["close"].pct_change(20)
    out = out_dir / "fact_price_daily.csv"
    save_csv(df, out)
    return out


def build_fact_market_return(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockTotalReturnIndex.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_market_return（TaiwanStockTotalReturnIndex.json 無資料）")
        return None
    # 預期欄位：stock_id (TAIEX / TPEx), price, date
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    if "price" in df.columns:
        df = df.rename(columns={"price": "mkt_price"})
        df["mkt_ret_1d"] = df.groupby("stock_id")["mkt_price"].pct_change()
        df["mkt_ret_5d"] = df.groupby("stock_id")["mkt_price"].pct_change(5)
        df["mkt_ret_20d"] = df.groupby("stock_id")["mkt_price"].pct_change(20)
    out = out_dir / "fact_market_return.csv"
    save_csv(df, out)
    return out


def build_fact_institutional_flow(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockInstitutionalInvestorsBuySell.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_institutional_flow_daily（檔案缺或無資料）")
        return None
    # 預期欄位：date, stock_id, buy, name, sell
    for col in ["date"]:
        if col in df.columns:
            df[col] = to_dt(df[col])
    if not {"date", "stock_id", "name"}.issubset(df.columns):
        log("[略過] fact_institutional_flow_daily（缺必要欄位 date/stock_id/name）")
        return None
    df["net"] = (df["buy"] if "buy" in df.columns else 0) - (df["sell"] if "sell" in df.columns else 0)
    wide = df.pivot_table(index=["date", "stock_id"], columns="name",
                          values=["buy", "sell", "net"], aggfunc="sum")
    wide.columns = [f"{a}_{b}".replace(" ", "_") for a, b in wide.columns.to_flat_index()]
    wide = wide.reset_index().sort_values(["stock_id", "date"])
    # 滾動 5D/20D 的淨買賣（各主體）
    for col in [c for c in wide.columns if c.startswith("net_")]:
        grp = wide.groupby("stock_id")[col]
        wide[f"{col}_5d"] = grp.transform(lambda s: s.rolling(5, min_periods=1).sum())
        wide[f"{col}_20d"] = grp.transform(lambda s: s.rolling(20, min_periods=1).sum())
    out = out_dir / "fact_institutional_flow_daily.csv"
    save_csv(wide, out)
    return out


def build_fact_margin_short(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockMarginPurchaseShortSale.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_margin_short_daily（檔案缺或無資料）")
        return None
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    # 自動計算 Today/Yesterday 差
    for col in list(df.columns):
        if col.endswith("TodayBalance"):
            ycol = col.replace("Today", "Yesterday")
            if ycol in df.columns:
                df[f"{col}_delta"] = df[col] - df[ycol]
    df = df.sort_values(["stock_id", "date"])
    out = out_dir / "fact_margin_short_daily.csv"
    save_csv(df, out)
    return out


def build_fact_short_balances(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanDailyShortSaleBalances.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_short_balances_daily（檔案缺或無資料）")
        return None
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    df = df.sort_values(["stock_id", "date"])
    out = out_dir / "fact_short_balances_daily.csv"
    save_csv(df, out)
    return out


def build_fact_securities_lending(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockSecuritiesLending.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_securities_lending（檔案缺或無資料）")
        return None
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    df = df.sort_values(["stock_id", "date"])
    out = out_dir / "fact_securities_lending.csv"
    save_csv(df, out)
    return out


def build_fact_foreign_shareholding(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockShareholding.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_foreign_shareholding_daily（檔案缺或無資料）")
        return None
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    # 優先挑包含 Ratio 關鍵字的外資比例欄位
    ratio_cols = [c for c in df.columns if "ShareholdingRatio" in c or "RemainRatio" in c]
    keep = ["date", "stock_id"] + ratio_cols
    keep = [c for c in keep if c in df.columns]
    out_df = df[keep].sort_values(["stock_id", "date"])
    out = out_dir / "fact_foreign_shareholding_daily.csv"
    save_csv(out_df, out)
    return out


def build_fact_revenue_monthly(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / "TaiwanStockMonthRevenue.json"
    df = load_json_df(src)
    if df.empty:
        log("[略過] fact_revenue_monthly（檔案缺或無資料）")
        return None
    # 建立年月份鍵（優先用 revenue_year/revenue_month）
    if "revenue_year" in df.columns and "revenue_month" in df.columns:
        df["ym"] = (
            df["revenue_year"].astype(int).astype(str)
            + "-"
            + df["revenue_month"].astype(int).astype(str).str.zfill(2)
        )
    else:
        df["ym"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    # 排序後計算 MoM / YoY
    df = df.sort_values(["stock_id", "ym"])
    if "revenue" in df.columns:
        df["revenue_mom"] = df.groupby("stock_id")["revenue"].pct_change(1)
        df["revenue_yoy"] = df.groupby("stock_id")["revenue"].pct_change(12)
    out = out_dir / "fact_revenue_monthly.csv"
    save_csv(df, out)
    return out


def pivot_financials(raw_file: str, out_name: str, raw_dir: Path, out_dir: Path) -> Optional[Path]:
    src = raw_dir / raw_file
    df = load_json_df(src)
    if df.empty:
        log(f"[略過] {out_name}（{raw_file} 缺或無資料）")
        return None
    # 預期欄位：date, stock_id, type, value
    if "date" in df.columns:
        df["date"] = to_dt(df["date"])
    if not {"date", "stock_id", "type", "value"}.issubset(df.columns):
        log(f"[略過] {out_name}（缺必要欄位 date/stock_id/type/value）")
        return None
    wide = df.pivot_table(index=["date", "stock_id"], columns="type", values="value", aggfunc="sum")
    wide.columns = [str(c).replace(" ", "_") for c in wide.columns]
    wide = wide.reset_index().sort_values(["stock_id", "date"])
    out = out_dir / out_name
    save_csv(wide, out)
    return out


def build_all(raw_dir: Path, out_dir: Path) -> List[str]:
    ensure_outdir(out_dir)
    built: List[str] = []

    log(EXPORTS_HINT.strip())

    # 維度
    for fn in [
        build_dim_security(raw_dir, out_dir),
        build_dim_calendar(raw_dir, out_dir),
    ]:
        if fn: built.append(fn.name)

    # 日頻
    for fn in [
        build_fact_price(raw_dir, out_dir),
        build_fact_market_return(raw_dir, out_dir),
        build_fact_institutional_flow(raw_dir, out_dir),
        build_fact_margin_short(raw_dir, out_dir),
        build_fact_short_balances(raw_dir, out_dir),
        build_fact_securities_lending(raw_dir, out_dir),
        build_fact_foreign_shareholding(raw_dir, out_dir),
    ]:
        if fn: built.append(fn.name)

    # 月/季
    fn = build_fact_revenue_monthly(raw_dir, out_dir)
    if fn: built.append(fn.name)

    # 季財報三表（寬表）
    for raw_file, out_name in [
        ("TaiwanStockFinancialStatements.json", "fact_financials_income_stmt_quarterly.csv"),
        ("TaiwanStockBalanceSheet.json", "fact_financials_balance_sheet_quarterly.csv"),
        ("TaiwanStockCashFlowsStatement.json", "fact_financials_cashflows_quarterly.csv"),
    ]:
        fn = pivot_financials(raw_file, out_name, raw_dir, out_dir)
        if fn: built.append(fn.name)

    return built


def main():
    ap = argparse.ArgumentParser(description="FinMind raw → 清理/標準化 CSV")
    ap.add_argument("--raw-dir", type=str, default="finmind_raw", help="原始 JSON 目錄（預設：finmind_raw）")
    ap.add_argument("--out-dir", type=str, default="finmind_out", help="輸出 CSV 目錄（預設：finmind_out）")

    args = ap.parse_args()


    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        log(f"[錯誤] 找不到原始資料夾：{raw_dir}")
        raise SystemExit(1)

    built = build_all(raw_dir, out_dir)
    if built:
        log("\n=== 輸出完成 ===")
        for name in built:
            log(f"- {name}")
    else:
        log("[警告] 沒有可輸出的檔案，請確認 raw 目錄內是否含有對應的 JSON。")


if __name__ == "__main__":
    main()
