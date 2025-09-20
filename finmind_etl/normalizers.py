"""Legacy normalisation helpers built on top of the dataset utilities."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .datasets import ALL_SPECS, DatasetSpec, clean_dataset, get_spec


Normalizer = Callable[[pd.DataFrame], pd.DataFrame]


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


def _normalize_with_spec(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run ``clean_dataset`` for a specific dataset name."""

    spec = get_spec(dataset_name)
    return clean_dataset(spec, df)


def normalize_taiwan_stock_price(df: pd.DataFrame) -> pd.DataFrame:
    """清洗台股日價量資料。"""

    return _normalize_with_spec("TaiwanStockPrice", df)


def normalize_taiwan_stock_price_adj(df: pd.DataFrame) -> pd.DataFrame:
    """清洗還原權息日價量資料。"""

    return _normalize_with_spec("TaiwanStockPriceAdj", df)


def normalize_institutional_investors(df: pd.DataFrame) -> pd.DataFrame:
    """清洗三大法人買賣超資料。"""

    return _normalize_with_spec("TaiwanStockInstitutionalInvestorsBuySell", df)


def normalize_margin_short(df: pd.DataFrame) -> pd.DataFrame:
    """清洗融資融券資料。"""

    return _normalize_with_spec("TaiwanStockMarginPurchaseShortSale", df)


def normalize_month_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """清洗月營收資料，並加入頻率欄位。"""

    return _normalize_with_spec("TaiwanStockMonthRevenue", df)


def _make_normalizer(spec: DatasetSpec) -> Normalizer:
    def _normalizer(df: pd.DataFrame, *, _spec: DatasetSpec = spec) -> pd.DataFrame:
        return clean_dataset(_spec, df)

    _normalizer.__name__ = f"normalize_{spec.name}"
    return _normalizer


def _build_normalizers() -> Dict[str, Normalizer]:
    catalog: Dict[str, Normalizer] = {
        name: _make_normalizer(spec) for name, spec in ALL_SPECS.items()
    }
    catalog.update(
        {
            "TaiwanStockPrice": normalize_taiwan_stock_price,
            "TaiwanStockPriceAdj": normalize_taiwan_stock_price_adj,
            "TaiwanStockInstitutionalInvestorsBuySell": normalize_institutional_investors,
            "TaiwanStockMarginPurchaseShortSale": normalize_margin_short,
            "TaiwanStockMonthRevenue": normalize_month_revenue,
        }
    )
    return catalog


NORMALIZERS: Dict[str, Normalizer] = _build_normalizers()


__all__ = [
    "Normalizer",
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
