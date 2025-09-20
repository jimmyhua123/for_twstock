"""基本面資料下載與整理。"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .api import APIClient
from .technical import _apply_translation, _ensure_datetime, _ensure_stock_id, _numeric, _to_snake_case

LOGGER = logging.getLogger("finmind_etl.fundamentals")


def _default_start(since: str, months: int = 18) -> str:
    base = pd.to_datetime(since, errors="coerce")
    if pd.isna(base):
        base = pd.Timestamp.today() - pd.DateOffset(months=months)
    else:
        base = base - pd.DateOffset(months=months)
    return base.strftime("%Y-%m-%d")


def fetch_month_revenue(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """抓取月營收資料。"""

    start = _default_start(since, months=13)
    end = end_date or date.today().strftime("%Y-%m-%d")
    translation = client.try_translation("TaiwanStockMonthRevenue")
    frames: List[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.fetch_dataset("TaiwanStockMonthRevenue", stock, start, end)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("月營收抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            continue
        df = _apply_translation(df, translation)
        df = _ensure_datetime(df)
        df = _ensure_stock_id(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "revenue"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    merged = merged.rename(columns={"revenue_last_year": "revenue_prev_year", "revenue_last_month": "revenue_prev_month"})
    _numeric(
        merged,
        [
            "revenue",
            "revenue_prev_year",
            "revenue_prev_month",
            "accumulated_revenue",
            "accumulated_revenue_last_year",
        ],
    )
    if "revenue_month" not in merged.columns and "date" in merged.columns:
        merged["revenue_month"] = merged["date"].dt.month
    if "revenue_year" not in merged.columns and "date" in merged.columns:
        merged["revenue_year"] = merged["date"].dt.year
    merged["revenue_yoy"] = merged.groupby("stock_id")["revenue"].pct_change(periods=12)
    merged["revenue_mom"] = merged.groupby("stock_id")["revenue"].pct_change()
    merged[["revenue_yoy", "revenue_mom"]] = merged[["revenue_yoy", "revenue_mom"]].replace([np.inf, -np.inf], np.nan)
    if "country" not in merged.columns:
        merged["country"] = "TW"
    return merged


def fetch_financial_statements(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """抓取財報資料並提取 EPS。"""

    start = _default_start(since, months=24)
    end = end_date or date.today().strftime("%Y-%m-%d")
    translation = client.try_translation("TaiwanStockFinancialStatements")
    frames: List[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.fetch_dataset("TaiwanStockFinancialStatements", stock, start, end)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("財報抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            continue
        df = _apply_translation(df, translation)
        df.columns = [_to_snake_case(col) for col in df.columns]
        if "type" not in df.columns:
            candidate = next((col for col in ("origin_name", "name", "report_type") if col in df.columns), None)
            if candidate:
                df["type"] = df[candidate]
        if "value" not in df.columns:
            candidate = next((col for col in ("amount", "val", "data_value") if col in df.columns), None)
            if candidate:
                df["value"] = df[candidate]
        if "type" not in df.columns or "value" not in df.columns:
            LOGGER.warning("財報缺少必要欄位：%s", stock)
            continue
        df["type"] = df["type"].astype(str).str.lower()
        df = df[df["type"].str.contains("eps", case=False, na=False)]
        if df.empty:
            continue
        df = _ensure_datetime(df)
        df = _ensure_stock_id(df)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df[["stock_id", "date", "value"]])
    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "eps"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["stock_id", "date"]).reset_index(drop=True)
    combined = combined.rename(columns={"value": "eps"})
    combined["eps_ttm"] = combined.groupby("stock_id")["eps"].transform(lambda s: s.rolling(window=4, min_periods=1).sum())
    return combined


def fetch_fundamental_data(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """取得基本面相關資料集。"""

    end = end_date or date.today().strftime("%Y-%m-%d")
    LOGGER.info("抓取基本面資料：股票數量=%s", len(stocks))
    month_revenue = fetch_month_revenue(stocks, since, client, end)
    financial = fetch_financial_statements(stocks, since, client, end)
    return {
        "TaiwanStockMonthRevenue": month_revenue,
        "TaiwanStockFinancialStatements": financial,
    }


__all__ = ["fetch_fundamental_data", "fetch_month_revenue", "fetch_financial_statements"]
