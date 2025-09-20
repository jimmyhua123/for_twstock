"""Alignment helpers used to build the daily wide table."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .datasets import DatasetResult

LOGGER = logging.getLogger("finmind_etl.align")


def build_trading_base(
    stocks: Sequence[str],
    calendar: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    """Create a stock x date base using the trading calendar."""

    stocks = sorted({s.strip().zfill(4) for s in stocks})
    if not stocks:
        return pd.DataFrame(columns=["stock_id", "date"])

    if calendar is not None and not calendar.empty and "date" in calendar.columns:
        dates = pd.to_datetime(calendar["date"], errors="coerce")
        if "is_trading_day" in calendar.columns:
            mask = calendar["is_trading_day"].astype(int) == 1
            dates = dates[mask]
        dates = dates.dropna().dt.normalize().unique()
    else:
        LOGGER.warning("交易日曆為空，改用股價資料推算日期。")
        dates = pd.to_datetime(price.get("date"), errors="coerce").dropna().dt.normalize().unique()

    if len(dates) == 0:
        return pd.DataFrame(columns=["stock_id", "date"])

    base = pd.MultiIndex.from_product([stocks, sorted(dates)], names=["stock_id", "date"]).to_frame(index=False)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    return base


def merge_stock_level(
    base: pd.DataFrame,
    df: pd.DataFrame,
    columns: Iterable[str],
    forward_fill: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        for column in columns:
            if column not in {"stock_id", "date"}:
                base[column] = np.nan
        return base

    merged = base.merge(df, on=["stock_id", "date"], how="left")
    value_columns = [col for col in columns if col not in {"stock_id", "date"}]
    if forward_fill and value_columns:
        merged[value_columns] = merged.groupby("stock_id")[value_columns].ffill()
    return merged


def merge_market_level(
    base: pd.DataFrame,
    df: pd.DataFrame,
    columns: Iterable[str],
    forward_fill: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        for column in columns:
            if column != "date":
                base[column] = np.nan
        return base

    merged = base.merge(df, on="date", how="left")
    value_columns = [col for col in columns if col != "date"]
    if forward_fill and value_columns:
        merged[value_columns] = merged[value_columns].ffill()
    return merged


def align_dataset(base: pd.DataFrame, result: DatasetResult) -> pd.DataFrame:
    spec = result.spec
    df = result.clean
    if not spec.include_in_wide:
        return base

    columns = list(df.columns)
    if spec.level == "market" or not spec.requires_stock:
        return merge_market_level(base, df, columns, spec.forward_fill)
    return merge_stock_level(base, df, columns, spec.forward_fill)


def append_stock_info(base: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    if info is None or info.empty:
        return base
    keep_columns = [col for col in info.columns if col in {"stock_id", "stock_name", "industry_category"}]
    if not keep_columns:
        return base
    dedup = info[keep_columns].drop_duplicates(subset=["stock_id"])
    return base.merge(dedup, on="stock_id", how="left")


def extract_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = [col for col in ["open", "high", "low", "close", "volume", "turnover", "transactions", "price_change"] if col in df.columns]
    columns = ["stock_id", "date", *price_cols]
    missing = {"stock_id", "date"} - set(df.columns)
    if missing:
        return pd.DataFrame(columns=columns)
    return df[columns].copy()


__all__ = [
    "build_trading_base",
    "merge_stock_level",
    "merge_market_level",
    "align_dataset",
    "append_stock_info",
    "extract_price_columns",
]

