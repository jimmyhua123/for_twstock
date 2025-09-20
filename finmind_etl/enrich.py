"""資料集整併與寬表產生模組。"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("finmind_etl.enrich")


def _build_trading_index(
    stocks: Sequence[str],
    calendar: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    """依交易日曆建立股票 x 日期的基底。"""

    stocks = sorted({s.zfill(4) for s in stocks})
    if calendar is not None and not calendar.empty:
        calendar = calendar.copy()
        if "is_trading_day" in calendar.columns:
            calendar = calendar[calendar["is_trading_day"].astype(int) == 1]
        if "date" not in calendar.columns:
            raise ValueError("交易日曆缺少 date 欄位")
        dates = pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.normalize()
    else:
        LOGGER.warning("交易日曆為空，改用股價資料中的日期集合。")
        dates = pd.to_datetime(price.get("date"), errors="coerce").dropna().dt.normalize()
    dates = sorted(dates.unique())
    if not dates:
        return pd.DataFrame(columns=["stock_id", "date"])
    index = pd.MultiIndex.from_product([stocks, dates], names=["stock_id", "date"])
    base = index.to_frame(index=False)
    base["date"] = pd.to_datetime(base["date"], utc=False)
    return base


def _merge_with_forward_fill(
    base: pd.DataFrame,
    df: pd.DataFrame,
    columns: Iterable[str],
    forward_fill: bool = True,
) -> pd.DataFrame:
    """將附加資料合併到基底，必要時進行前值延展。"""

    if df is None or df.empty:
        for column in columns:
            if column not in base.columns:
                base[column] = np.nan
        return base
    merged = base.merge(df, on=["stock_id", "date"], how="left")
    value_columns = [col for col in columns if col not in {"stock_id", "date"}]
    if forward_fill and value_columns:
        # 財報與籌碼資料皆可能非日頻，採前值延展以維持每日資料連續性。
        merged[value_columns] = merged.groupby("stock_id")[value_columns].ffill()
    return merged


def _merge_date_level(base: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return base
    value_columns = [col for col in df.columns if col != "date"]
    merged = base.merge(df, on="date", how="left")
    if value_columns:
        merged[value_columns] = merged[value_columns].ffill()
    return merged


def build_daily_wide(
    stocks: Sequence[str],
    technical: Dict[str, pd.DataFrame],
    fundamentals: Dict[str, pd.DataFrame],
    chip: Dict[str, pd.DataFrame],
    derivative: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """整併所有資料集並產生每日寬表。"""

    price = technical.get("TaiwanStockPrice", pd.DataFrame())
    calendar = technical.get("TaiwanStockTradingDate", pd.DataFrame())
    info = technical.get("TaiwanStockInfo", pd.DataFrame())
    base = _build_trading_index(stocks, calendar, price)
    if base.empty:
        return base

    if not price.empty:
        price = price.copy()
        price_cols = [col for col in price.columns if col not in {"date", "stock_id"}]
        base = base.merge(price, on=["stock_id", "date"], how="left")
    else:
        LOGGER.warning("股價資料為空，寬表僅包含其他欄位。")

    price_adj = technical.get("TaiwanStockPriceAdj")
    if price_adj is not None and not price_adj.empty:
        adj = price_adj.copy()
        rename_map = {
            column: f"adj_{column}" if column not in {"date", "stock_id"} else column
            for column in adj.columns
        }
        adj = adj.rename(columns=rename_map)
        adj_columns = [col for col in adj.columns if col not in {"date", "stock_id"}]
        base = base.merge(adj, on=["stock_id", "date"], how="left")
        for column in adj_columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    # 籌碼資料（法人、融資融券）以前值延展補齊
    for dataset, df in chip.items():
        if df is None:
            continue
        columns = [col for col in df.columns if col not in {"stock_id", "date"}]
        base = _merge_with_forward_fill(base, df, ["stock_id", "date", *columns], forward_fill=True)

    # 基本面資料為月/季頻率，使用前值延展至下一次公告日
    for dataset, df in fundamentals.items():
        if df is None:
            continue
        columns = [col for col in df.columns if col not in {"stock_id", "date"}]
        base = _merge_with_forward_fill(base, df, ["stock_id", "date", *columns], forward_fill=True)

    # 衍生性資料僅以日期為索引
    for dataset, df in derivative.items():
        if df is None:
            continue
        base = _merge_date_level(base, df)

    # 追加股票基本資訊
    if info is not None and not info.empty:
        info = info.copy()
        keep_cols = [col for col in info.columns if col in {"stock_id", "industry_category", "stock_name"}]
        if keep_cols:
            base = base.merge(info[keep_cols].drop_duplicates(subset=["stock_id"]), on="stock_id", how="left")

    # 衍生欄位計算
    if "close" in base.columns:
        base = base.sort_values(["stock_id", "date"]).reset_index(drop=True)
        base["return"] = base.groupby("stock_id")["close"].pct_change()

    if "turnover" in base.columns:
        base["turnover_rank_pct"] = (
            base.groupby("date")["turnover"].rank(method="average", pct=True)
        ).clip(0, 1)
        turnover_ma20 = (
            base.groupby("stock_id")["turnover"].transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        )
        base["turnover_change_vs_ma20"] = base["turnover"] / turnover_ma20 - 1
        base["turnover_change_vs_ma20"] = base["turnover_change_vs_ma20"].replace([np.inf, -np.inf], np.nan)

    if "volume" in base.columns:
        base["volume_rank_pct"] = (
            base.groupby("date")["volume"].rank(method="average", pct=True)
        ).clip(0, 1)
        base["volume_ma20"] = (
            base.groupby("stock_id")["volume"].transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        )
        base["volume_ratio"] = base["volume"] / base["volume_ma20"]
        base["volume_ratio"] = base["volume_ratio"].replace([np.inf, -np.inf], np.nan)

    if "transactions" in base.columns:
        transactions_ma20 = (
            base.groupby("stock_id")["transactions"].transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        )
        base["transactions_change_vs_ma20"] = base["transactions"] / transactions_ma20 - 1
        base["transactions_change_vs_ma20"] = base["transactions_change_vs_ma20"].replace([np.inf, -np.inf], np.nan)

    base = base.sort_values(["date", "stock_id"]).reset_index(drop=True)
    return base


def build_minimal_view(df: pd.DataFrame) -> pd.DataFrame:
    """建立精簡版寬表。"""

    min_columns = [
        "date",
        "stock_id",
        "open",
        "high",
        "low",
        "close",
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
    for column in min_columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[min_columns].copy()


__all__ = ["build_daily_wide", "build_minimal_view"]
