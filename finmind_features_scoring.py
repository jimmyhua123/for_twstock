#!/usr/bin/env python3
"""Generate fine-profile raw features from FinMind cleaned datasets.

This script reads daily price/volume, institutional flows, margin/short data,
securities borrowing balances, monthly revenue and quarterly financial
statements. It computes the requested technical, chip, fundamental and risk
features and writes both a full daily panel and the latest snapshot for the
specified ``--until`` date.

Outputs:
    finmind_scores/features_snapshot_fine_YYYYMMDD.csv
    finmind_scores/features_daily.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers copied from spec snippets
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    r = series.diff()
    up = r.clip(lower=0).rolling(period).mean()
    dn = (-r.clip(upper=0)).rolling(period).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))


def _drawdown_rolling(high_series: pd.Series, close_series: pd.Series, window: int = 60) -> pd.Series:
    roll_max = high_series.rolling(window).max()
    dd = (close_series / roll_max) - 1.0
    return dd


def _ttm_sum(df: pd.DataFrame, cols: Iterable[str], by: List[str] | None = None) -> pd.DataFrame:
    by = by or ["stock_id"]
    g = df.sort_values(by + ["date"]).groupby(by, group_keys=False)
    out = {}
    for c in cols:
        out[f"{c}_ttm"] = g[c].apply(lambda s: s.rolling(4, min_periods=1).sum())
    return pd.DataFrame(out)


def _safe_row_mean(mat: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(mat), axis=1)
    s = np.nansum(mat, axis=1)
    out = np.full(mat.shape[0], np.nan, float)
    m = cnt > 0
    out[m] = s[m] / cnt[m]
    return out


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df


def _find_dataset(base: Optional[Path], patterns: Iterable[str]) -> Optional[Path]:
    if not base:
        return None
    for pat in patterns:
        candidate = base / pat
        if candidate.exists():
            return candidate
    # fallback: glob search (first match)
    for pat in patterns:
        matches = sorted(base.rglob(pat))
        if matches:
            return matches[0]
    return None


def _load_dataset(clean_dir: Path, raw_dir: Optional[Path], patterns: Iterable[str]) -> pd.DataFrame:
    patterns = list(patterns)
    path = _find_dataset(clean_dir, patterns)
    if not path and raw_dir:
        path = _find_dataset(raw_dir, patterns)
    if not path:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        df = _read_parquet(path)
    else:
        df = _read_csv(path)
    return df


def _ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _prepare_price_panel(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    df = _load_dataset(clean_dir, raw_dir, ["fact_price_daily.parquet", "fact_price_daily.csv"])
    if df.empty:
        return df
    rename_map = {}
    for col in list(df.columns):
        lc = col.lower()
        if lc in {"max", "high"}:
            rename_map[col] = "high"
        elif lc in {"min", "low"}:
            rename_map[col] = "low"
        elif lc in {"trading_volume", "volume"}:
            rename_map[col] = "volume"
        elif lc == "trading_turnover":
            rename_map[col] = "turnover"
        elif lc == "rsi14":
            rename_map[col] = "rsi_14"
    df = df.rename(columns=rename_map)
    needed = {"date", "stock_id", "open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        return pd.DataFrame()
    df = _ensure_datetime(df, ["date"])
    num_cols = ["open", "high", "low", "close", "volume"]
    _to_numeric(df, num_cols)
    df = df.dropna(subset=["stock_id", "date"])
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values(["stock_id", "date"])
    df = df.drop_duplicates(["stock_id", "date"], keep="last")
    return df


def _compute_price_features(price: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return price
    g = price.groupby("stock_id", group_keys=False)
    price["ret_1d"] = g["close"].pct_change(1)
    price["ret_5d"] = g["close"].pct_change(5)
    price["ret_20d"] = g["close"].pct_change(20)
    price["high_20d"] = g["high"].transform(lambda s: s.rolling(20).max())
    price["breakout_20d"] = price["close"] / price["high_20d"] - 1.0
    price["volatility_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20).std())
    price["volume_mean_20d"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    price["volume_ratio_20d"] = price["volume"] / price["volume_mean_20d"].replace(0, np.nan)
    price["rsi_14"] = g["close"].transform(_rsi)
    price["rolling_high_60d"] = g["high"].transform(lambda s: s.rolling(60).max())
    price["drawdown_60d"] = price["close"] / price["rolling_high_60d"] - 1.0
    price = price.drop(columns=["rolling_high_60d"], errors="ignore")
    return price


def _prepare_institutional(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    inst = _load_dataset(clean_dir, raw_dir, ["fact_institutional_flow_daily.parquet", "fact_institutional_flow_daily.csv"])
    if inst.empty:
        return inst
    inst = _ensure_datetime(inst, ["date"])
    inst = inst.dropna(subset=["stock_id", "date"])
    inst["date"] = inst["date"].dt.normalize()
    inst = inst.sort_values(["stock_id", "date"])
    inst = inst.drop_duplicates(["stock_id", "date"], keep="last")
    net_cols: List[str] = []
    for col in inst.columns:
        lc = col.lower()
        if not lc.startswith("net"):
            continue
        if lc.endswith("_5d") or lc.endswith("_20d"):
            continue
        net_cols.append(col)
    if not net_cols and "inst_net" in inst.columns:
        net_cols = ["inst_net"]
    for col in net_cols:
        inst[col] = pd.to_numeric(inst[col], errors="coerce")
    if net_cols:
        inst["inst_net"] = inst[net_cols].sum(axis=1, skipna=True)
    else:
        inst["inst_net"] = np.nan
    return inst[["stock_id", "date", "inst_net"]]


def _prepare_margin(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    margin = _load_dataset(clean_dir, raw_dir, ["fact_margin_short_daily.parquet", "fact_margin_short_daily.csv"])
    if margin.empty:
        return margin
    margin = _ensure_datetime(margin, ["date"])
    margin = margin.dropna(subset=["stock_id", "date"])
    margin["date"] = margin["date"].dt.normalize()
    margin = margin.sort_values(["stock_id", "date"])
    margin = margin.drop_duplicates(["stock_id", "date"], keep="last")
    short_cols = [
        "ShortSaleSell",
        "ShortSaleTodayBalance",
        "SBLShortSalesShortSales",
        "SBLShortSalesShortCovering",
    ]
    short_col = next((c for c in short_cols if c in margin.columns), None)
    if short_col:
        margin[short_col] = pd.to_numeric(margin[short_col], errors="coerce")
        margin = margin.rename(columns={short_col: "short_sell"})
    else:
        margin["short_sell"] = np.nan
    return margin[["stock_id", "date", "short_sell"]]


def _prepare_borrow_balance(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    borrow = _load_dataset(
        clean_dir,
        raw_dir,
        [
            "fact_short_balances_daily.parquet",
            "fact_short_balances_daily.csv",
            "BorrowingBalance.parquet",
            "BorrowingBalance.csv",
        ],
    )
    if borrow.empty:
        return borrow
    borrow = _ensure_datetime(borrow, ["date"])
    borrow = borrow.dropna(subset=["stock_id", "date"])
    borrow["date"] = borrow["date"].dt.normalize()
    borrow = borrow.sort_values(["stock_id", "date"])
    borrow = borrow.drop_duplicates(["stock_id", "date"], keep="last")
    balance_cols = [
        "BorrowingBalance",
        "SBLShortSalesCurrentDayBalance",
        "MarginShortSalesCurrentDayBalance",
    ]
    balance_col = next((c for c in balance_cols if c in borrow.columns), None)
    if balance_col:
        borrow[balance_col] = pd.to_numeric(borrow[balance_col], errors="coerce")
        borrow = borrow.rename(columns={balance_col: "borrow_balance"})
    else:
        borrow["borrow_balance"] = np.nan
    return borrow[["stock_id", "date", "borrow_balance"]]


def _prepare_revenue(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    rev = _load_dataset(clean_dir, raw_dir, ["fact_revenue_monthly.parquet", "fact_revenue_monthly.csv"])
    if rev.empty:
        return rev
    rev = _ensure_datetime(rev, ["date"])
    rev = rev.dropna(subset=["stock_id", "date"])
    rev["date"] = rev["date"].dt.normalize()
    rev = rev.sort_values(["stock_id", "date"])
    rev = rev.drop_duplicates(["stock_id", "date"], keep="last")
    rev["revenue"] = pd.to_numeric(rev.get("revenue"), errors="coerce")
    rev["revenue_yoy"] = rev.groupby("stock_id")["revenue"].transform(lambda s: s.pct_change(12))
    return rev[["stock_id", "date", "revenue_yoy"]]


def _coalesce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        result = result.fillna(vals)
    return result


def _prepare_financials(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    income = _load_dataset(
        clean_dir,
        raw_dir,
        ["fact_financials_income_stmt_quarterly.parquet", "fact_financials_income_stmt_quarterly.csv"],
    )
    balance = _load_dataset(
        clean_dir,
        raw_dir,
        ["fact_financials_balance_sheet_quarterly.parquet", "fact_financials_balance_sheet_quarterly.csv"],
    )
    if income.empty:
        return pd.DataFrame(columns=["stock_id", "date", "gross_margin_ttm", "op_margin_ttm", "roe_ttm"])
    income = _ensure_datetime(income, ["date"])
    income = income.dropna(subset=["stock_id", "date"])
    income["date"] = income["date"].dt.normalize()
    income = income.sort_values(["stock_id", "date"])
    keep_cols = {
        "Revenue": "Revenue",
        "OperatingIncome": "OperatingIncome",
        "GrossProfit": "GrossProfit",
    }
    for src, dst in keep_cols.items():
        if src in income.columns:
            income[dst] = pd.to_numeric(income[src], errors="coerce")
        else:
            income[dst] = np.nan
    income["NetIncome"] = _coalesce_numeric(
        income,
        ["IncomeAfterTax", "IncomeAfterTaxes", "NetIncome", "Income"],
    )
    ttm = _ttm_sum(income, ["Revenue", "GrossProfit", "OperatingIncome", "NetIncome"])
    income = pd.concat([income[["stock_id", "date"]].reset_index(drop=True), ttm.reset_index(drop=True)], axis=1)
    if balance.empty:
        income["equity_avg_ttm"] = np.nan
    else:
        balance = _ensure_datetime(balance, ["date"])
        balance = balance.dropna(subset=["stock_id", "date"])
        balance["date"] = balance["date"].dt.normalize()
        balance = balance.sort_values(["stock_id", "date"])
        equity = _coalesce_numeric(
            balance,
            ["EquityAttributableToOwnersOfParent", "Equity", "TotalEquity"],
        )
        equity = equity.rename("equity")
        balance = pd.concat(
            [balance[["stock_id", "date"]].reset_index(drop=True), equity.reset_index(drop=True)],
            axis=1,
        )
        balance["equity_avg_ttm"] = (
            balance.groupby("stock_id")["equity"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
        income = income.merge(
            balance[["stock_id", "date", "equity_avg_ttm"]],
            on=["stock_id", "date"],
            how="left",
        )
    income["gross_margin_ttm"] = income["GrossProfit_ttm"] / income["Revenue_ttm"]
    income["op_margin_ttm"] = income["OperatingIncome_ttm"] / income["Revenue_ttm"]
    income["roe_ttm"] = income["NetIncome_ttm"] / income["equity_avg_ttm"]
    return income[["stock_id", "date", "gross_margin_ttm", "op_margin_ttm", "roe_ttm"]]


def _merge_latest_by_stock(
    base: pd.DataFrame, right: pd.DataFrame, value_map: Mapping[str, str]
) -> pd.DataFrame:
    if base.empty or right.empty:
        for new_col in value_map.values():
            base[new_col] = base.get(new_col, np.nan)
        return base
    frames = []
    for sid, left_part in base.groupby("stock_id", group_keys=False):
        right_part = right[right["stock_id"] == sid]
        if right_part.empty:
            temp = left_part.copy()
            for new_col in value_map.values():
                temp[new_col] = np.nan
            frames.append(temp)
            continue
        right_part = right_part.sort_values("date").drop_duplicates("date", keep="last")
        merged = pd.merge_asof(
            left_part.sort_values("date"),
            right_part.sort_values("date").drop(columns=["stock_id"], errors="ignore"),
            left_on="date",
            right_on="date",
            direction="backward",
        )
        merged = merged.rename(columns=value_map)
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    return out


def build_fine_features(clean_dir: Path, raw_dir: Optional[Path]) -> pd.DataFrame:
    price = _prepare_price_panel(clean_dir, raw_dir)
    if price.empty:
        raise SystemExit("[錯誤] 找不到價量日表（fact_price_daily），請先完成 clean 步驟。")
    price = _compute_price_features(price)

    inst = _prepare_institutional(clean_dir, raw_dir)
    margin = _prepare_margin(clean_dir, raw_dir)
    borrow = _prepare_borrow_balance(clean_dir, raw_dir)
    revenue = _prepare_revenue(clean_dir, raw_dir)
    financials = _prepare_financials(clean_dir, raw_dir)
    dim = _load_dataset(clean_dir, raw_dir, ["dim_security.parquet", "dim_security.csv"])

    feats = price.copy()
    feats = feats.merge(inst, on=["stock_id", "date"], how="left") if not inst.empty else feats
    feats = feats.merge(margin, on=["stock_id", "date"], how="left") if not margin.empty else feats
    feats = feats.merge(borrow, on=["stock_id", "date"], how="left") if not borrow.empty else feats

    g = feats.groupby("stock_id", group_keys=False)
    feats["volume_sum_5d"] = g["volume"].transform(lambda s: s.rolling(5).sum())
    feats["inst_net_sum_5d"] = g["inst_net"].transform(lambda s: s.rolling(5).sum())
    feats["inst_net_buy_5d_ratio"] = feats["inst_net_sum_5d"] / feats["volume_sum_5d"].replace(0, np.nan)
    feats["inst_positive"] = (feats["inst_net"].fillna(0) > 0).astype(float)
    feats["inst_consistency_20d"] = g["inst_positive"].transform(lambda s: s.rolling(20).mean())

    feats["short_sell_5d_sum"] = g["short_sell"].transform(lambda s: s.rolling(5).sum()) if "short_sell" in feats.columns else np.nan
    feats["margin_short_ratio_5d"] = feats["short_sell_5d_sum"] / feats["volume_sum_5d"].replace(0, np.nan)

    if "borrow_balance" in feats.columns:
        feats["borrow_balance_chg_5d"] = g["borrow_balance"].transform(lambda s: s.pct_change(5))

    feats = _merge_latest_by_stock(feats, revenue, {"revenue_yoy": "revenue_yoy"})
    feats = _merge_latest_by_stock(
        feats,
        financials,
        {
            "gross_margin_ttm": "gross_margin_ttm",
            "op_margin_ttm": "op_margin_ttm",
            "roe_ttm": "roe_ttm",
        },
    )

    if not dim.empty:
        keep_cols = ["stock_id"] + [c for c in ["stock_name", "industry", "industry_category"] if c in dim.columns]
        feats = feats.merge(dim[keep_cols], on="stock_id", how="left")

    feats = feats.sort_values(["stock_id", "date"]).drop_duplicates(["stock_id", "date"], keep="last")
    feats = feats.drop(columns=["inst_positive"], errors="ignore")
    feats = feats.drop(columns=["volume_sum_5d", "inst_net_sum_5d", "short_sell_5d_sum"], errors="ignore")

    if "industry" not in feats.columns and "industry_category" in feats.columns:
        feats["industry"] = feats["industry_category"]

    return feats


def _format_snapshot_filename(as_of: datetime) -> str:
    return f"features_snapshot_fine_{as_of.strftime('%Y%m%d')}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="產生精算 fine profile 原始特徵")
    parser.add_argument("--clean-dir", default="finmind_out", help="清理後輸出目錄 (default: finmind_out)")
    parser.add_argument("--raw-dir", default="finmind_raw", help="原始 FinMind 落地目錄 (default: finmind_raw)")
    parser.add_argument("--out-dir", default="finmind_scores", help="輸出目錄 (default: finmind_scores)")
    parser.add_argument("--until", default=None, help="快照日期 YYYY-MM-DD；預設取資料最新日期")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = build_fine_features(clean_dir, raw_dir)
    if feats.empty:
        raise SystemExit("[錯誤] 無法建立特徵表。")

    feats["date"] = pd.to_datetime(feats["date"], errors="coerce").dt.normalize()
    latest_date = feats["date"].max()
    if pd.isna(latest_date):
        raise SystemExit("[錯誤] 找不到有效日期。")
    if args.until:
        as_of = pd.to_datetime(args.until).normalize()
    else:
        as_of = latest_date

    snapshot = feats[feats["date"] == as_of].copy()
    if snapshot.empty:
        raise SystemExit(f"[錯誤] 找不到 {as_of.date()} 的資料。")
    snapshot = snapshot.drop_duplicates("stock_id", keep="last").sort_values("stock_id")
    if "industry" not in snapshot.columns and "industry_category" in snapshot.columns:
        snapshot["industry"] = snapshot["industry_category"]

    snap_path = out_dir / _format_snapshot_filename(as_of)
    daily_path = out_dir / "features_daily.csv"

    snapshot_to_save = snapshot.copy()
    snapshot_to_save["date"] = snapshot_to_save["date"].dt.strftime("%Y-%m-%d")
    feats_to_save = feats.copy()
    feats_to_save["date"] = feats_to_save["date"].dt.strftime("%Y-%m-%d")

    snapshot_to_save.to_csv(snap_path, index=False, encoding="utf-8")
    feats_to_save.to_csv(daily_path, index=False, encoding="utf-8")

    print(f"[輸出] {snap_path} ({len(snapshot)} rows, {len(snapshot.columns)} cols)")
    print(f"[輸出] {daily_path} ({len(feats)} rows, {len(feats.columns)} cols)")


if __name__ == "__main__":
    main()
