"""資料清洗函式集合。"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .datasets import COLUMN_MAP, DATASET_CATALOG, NUMERIC_COLUMN_HINTS


def coerce_numeric_columns(
    df: pd.DataFrame,
    target_columns: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    """將指定欄位轉換為數值型態。"""

    if target_columns is None:
        target_columns = df.columns
    if exclude is None:
        exclude = []
    exclude_set = {"date", "stock_id", *exclude}
    for column in target_columns:
        if column in exclude_set or column not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        df[column] = (
            pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
        )


def standardize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """統一處理 date 與 stock_id 欄位。"""

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    else:
        df["stock_id"] = np.nan

    return df


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """確保指定欄位存在，不存在時補上 NaN。"""

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def normalize_taiwan_stock_price(df: pd.DataFrame) -> pd.DataFrame:
    """清洗台股日價量資料。"""

    if df.empty:
        columns = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockPrice"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["open", "high", "low", "close", "volume", "turnover"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockPrice"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_taiwan_stock_price_adj(df: pd.DataFrame) -> pd.DataFrame:
    """清洗還原權息日價量資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "adj_close",
        ]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockPriceAdj"])
    if "high" not in df.columns and "max" in df.columns:
        df.rename(columns={"max": "high"}, inplace=True)
    if "low" not in df.columns and "min" in df.columns:
        df.rename(columns={"min": "low"}, inplace=True)
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["open", "high", "low", "close", "volume", "turnover", "adj_close"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockPriceAdj"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_institutional_investors(df: pd.DataFrame) -> pd.DataFrame:
    """清洗三大法人買賣超資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "foreign",
            "investment_trust",
            "dealer",
            "dealer_self",
            "dealer_hedging",
        ]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockInstitutionalInvestorsBuySell"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["foreign", "investment_trust", "dealer", "dealer_self", "dealer_hedging"],
    )
    coerce_numeric_columns(
        df,
        target_columns=NUMERIC_COLUMN_HINTS[
            "TaiwanStockInstitutionalInvestorsBuySell"
        ],
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_margin_short(df: pd.DataFrame) -> pd.DataFrame:
    """清洗融資融券資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "margin_balance",
            "short_balance",
            "margin_change",
            "short_change",
        ]
        return pd.DataFrame(columns=columns)

    if "MarginPurchaseChange" not in df.columns and {
        "MarginPurchaseTodayBalance",
        "MarginPurchaseYesterdayBalance",
    }.issubset(df.columns):
        df["MarginPurchaseChange"] = (
            pd.to_numeric(
                df["MarginPurchaseTodayBalance"], errors="coerce"
            )
            - pd.to_numeric(df["MarginPurchaseYesterdayBalance"], errors="coerce")
        )
    if "ShortSaleChange" not in df.columns and {
        "ShortSaleTodayBalance",
        "ShortSaleYesterdayBalance",
    }.issubset(df.columns):
        df["ShortSaleChange"] = (
            pd.to_numeric(df["ShortSaleTodayBalance"], errors="coerce")
            - pd.to_numeric(df["ShortSaleYesterdayBalance"], errors="coerce")
        )
    df = df.rename(columns=COLUMN_MAP["TaiwanStockMarginPurchaseShortSale"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["margin_balance", "short_balance", "margin_change", "short_change"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockMarginPurchaseShortSale"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_month_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """清洗月營收資料，並加入頻率欄位。"""

    if df.empty:
        columns = ["date", "stock_id", "freq", "revenue"]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockMonthRevenue"])
    df = standardize_common_fields(df)
    df["freq"] = "M"
    coerce_numeric_columns(
        df,
        target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockMonthRevenue"],
        exclude=["freq"],
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


NORMALIZERS = {
    name: globals()[config["normalizer"]]
    for name, config in DATASET_CATALOG.items()
}

__all__ = [
    "coerce_numeric_columns",
    "standardize_common_fields",
    "ensure_columns",
    "normalize_taiwan_stock_price",
    "normalize_taiwan_stock_price_adj",
    "normalize_institutional_investors",
    "normalize_margin_short",
    "normalize_month_revenue",
    "NORMALIZERS",
]
