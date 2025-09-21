"""Assemble cleaned datasets into the required wide tables."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .align import align_dataset, append_stock_info, build_trading_base
from .datasets import DatasetResult

LOGGER = logging.getLogger("finmind_etl.merge")


def _collect_price(result_map: Mapping[str, DatasetResult]) -> pd.DataFrame:
    price = result_map.get("TaiwanStockPrice")
    if price is None or price.clean.empty:
        return pd.DataFrame()
    return price.clean


def _collect_calendar(result_map: Mapping[str, DatasetResult]) -> pd.DataFrame:
    calendar = result_map.get("TaiwanStockTradingDate")
    if calendar is None:
        return pd.DataFrame()
    return calendar.clean


def merge_all(
    stocks: Sequence[str],
    technical: Mapping[str, DatasetResult],
    fundamentals: Mapping[str, DatasetResult],
    chip: Mapping[str, DatasetResult],
) -> pd.DataFrame:
    """Build the daily wide DataFrame according to project spec."""

    price = _collect_price(technical)
    calendar = _collect_calendar(technical)
    base = build_trading_base(stocks, calendar, price)
    if base.empty:
        LOGGER.warning("無可用的交易日期，寬表為空。")
        return base

    for dataset, result in technical.items():
        if dataset in {"TaiwanStockPrice", "TaiwanStockTradingDate", "TaiwanStockInfo"}:
            continue
        base = align_dataset(base, result)

    if "TaiwanStockPrice" in technical:
        base = base.merge(technical["TaiwanStockPrice"].clean, on=["stock_id", "date"], how="left", suffixes=("", ""))

    for result in chip.values():
        base = align_dataset(base, result)
    for result in fundamentals.values():
        base = align_dataset(base, result)

    info = technical.get("TaiwanStockInfo")
    if info is not None:
        base = append_stock_info(base, info.clean)

    base = base.sort_values(["date", "stock_id"]).reset_index(drop=True)

    if "close" in base.columns:
        base["return"] = base.groupby("stock_id")["close"].pct_change()

    if "turnover" in base.columns:
        base["turnover_rank_pct"] = base.groupby("date")["turnover"].rank(method="average", pct=True).clip(0, 1)
        turnover_ma20 = base.groupby("stock_id")["turnover"].transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        base["turnover_change_5d"] = base.groupby("stock_id")["turnover"].transform(lambda s: s.pct_change(periods=5))
        base["turnover_change_vs_ma20"] = base["turnover"] / turnover_ma20 - 1
        base["turnover_change_vs_ma20"] = base["turnover_change_vs_ma20"].replace([np.inf, -np.inf], np.nan)

    if "volume" in base.columns:
        base["volume_rank_pct"] = base.groupby("date")["volume"].rank(method="average", pct=True).clip(0, 1)
        base["volume_ma20"] = base.groupby("stock_id")["volume"].transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        base["volume_ratio"] = base["volume"] / base["volume_ma20"]
        base["volume_ratio"] = base["volume_ratio"].replace([np.inf, -np.inf], np.nan)

    if "transactions" in base.columns:
        base["transactions_change_5d"] = base.groupby("stock_id")["transactions"].transform(lambda s: s.pct_change(periods=5))

    return base


def build_minimal_view(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "stock_id",
        "open",
        "high",
        "low",
        "close",
        "MA10Y",
        "volume",
        "turnover",
        "return",
        "foreign_net",
        "invest_trust_net",
        "dealer_net",
        "dealer_self_net",
        "dealer_hedging_net",
        "margin_long",
        "margin_short",
        "short_selling",
        "revenue",
        "revenue_yoy",
        "revenue_mom",
        "eps",
        "eps_ttm",
        "volume_ma20",
        "volume_ratio",
        "turnover_rank_pct",
        "volume_rank_pct",
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[columns].copy()


__all__ = ["merge_all", "build_minimal_view"]

