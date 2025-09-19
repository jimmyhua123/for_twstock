"""資料合併與寬表整理。"""

from __future__ import annotations

import re
from functools import reduce
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .config import LOGGER


def merge_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """依據日期與股票代號合併多個資料集。"""

    prepared_frames = []
    for dataset, df in frames.items():
        if df is None or df.empty:
            continue
        renamed = df.copy()
        rename_map = {
            column: f"{dataset}_{column}"
            for column in renamed.columns
            if column not in {"date", "stock_id"}
        }
        renamed = renamed.rename(columns=rename_map)
        prepared_frames.append(renamed)

    if not prepared_frames:
        LOGGER.warning("沒有任何資料可合併。")
        return pd.DataFrame(columns=["date", "stock_id"])

    def outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(left, right, on=["date", "stock_id"], how="outer")

    merged = reduce(outer_merge, prepared_frames)
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return merged


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """調整日期、股票代號與數值欄位的型態。"""

    if df.empty:
        return df.copy()

    normalized = df.copy()

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=False)
        try:
            normalized["date"] = normalized["date"].dt.tz_localize(None)
        except AttributeError:
            pass
    else:
        LOGGER.warning("缺少 date 欄位，後續輸出可能不完整。")

    if "stock_id" in normalized.columns:
        normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    else:
        LOGGER.warning("缺少 stock_id 欄位，後續輸出可能不完整。")

    text_like = {"date", "stock_id"}
    text_like.update({col for col in normalized.columns if "name" in col.lower()})
    text_like.update({col for col in normalized.columns if col.lower() in {"investor", "dataset"}})

    for column in normalized.columns:
        if column in text_like:
            continue
        series = normalized[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue
        cleaned = series.astype(str).str.strip().replace({"": np.nan})
        cleaned = cleaned.str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().any() or cleaned.eq("0").any():
            normalized[column] = numeric
        else:
            normalized[column] = cleaned

    return normalized


def _extract_price_block(df: pd.DataFrame) -> pd.DataFrame:
    """取得價量相關欄位並重新命名。"""

    if df.empty:
        return pd.DataFrame(columns=["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"])

    price_prefix = "taiwanstockprice_"
    price_columns = [
        column
        for column in df.columns
        if column.lower().startswith(price_prefix)
    ]

    if not price_columns:
        LOGGER.warning("在資料中找不到台股價量欄位，僅輸出法人資料。")
        return pd.DataFrame(columns=["date", "stock_id"])

    selected_columns = [col for col in ["date", "stock_id"] if col in df.columns]
    selected_columns.extend(price_columns)
    price_df = df[selected_columns].copy()

    rename_map = {}
    for column in price_columns:
        lower = column.lower()
        if lower == "taiwanstockprice_open":
            rename_map[column] = "open"
        elif lower == "taiwanstockprice_high":
            rename_map[column] = "high"
        elif lower == "taiwanstockprice_low":
            rename_map[column] = "low"
        elif lower == "taiwanstockprice_close":
            rename_map[column] = "close"
        elif lower == "taiwanstockprice_volume":
            rename_map[column] = "volume"
        elif lower == "taiwanstockprice_turnover":
            rename_map[column] = "turnover"

    price_df = price_df.rename(columns=rename_map)

    group_keys = [col for col in ["date", "stock_id"] if col in price_df.columns]
    if not group_keys:
        return price_df

    agg_map = {
        column: "first"
        for column in price_df.columns
        if column not in {"date", "stock_id"}
    }
    if agg_map:
        price_df = price_df.groupby(group_keys, as_index=False).agg(agg_map)
    else:
        price_df = price_df.drop_duplicates(subset=group_keys)

    return price_df


def _find_candidate_column(
    df: pd.DataFrame, keywords: Iterable[str], exclude: Optional[Iterable[str]] = None
) -> Optional[str]:
    """依據關鍵字尋找最適合的欄位名稱。"""

    excluded = {col.lower() for col in (exclude or [])}
    for column in df.columns:
        lower = column.lower()
        if lower in excluded:
            continue
        for keyword in keywords:
            key = keyword.lower()
            if lower == key or lower.endswith(f"_{key}") or key in lower:
                return column
    return None


def _build_institutional_wide(df: pd.DataFrame) -> pd.DataFrame:
    """整理法人買賣資料為寬表。"""

    empty_inst = pd.DataFrame(
        columns=[
            "date",
            "stock_id",
            "inst_foreign",
            "inst_investment_trust",
            "inst_dealer_self",
            "inst_dealer_hedging",
        ]
    )

    if df.empty:
        return empty_inst

    prefix = "taiwanstockinstitutionalinvestorsbuysell_"
    wide_columns = [
        column
        for column in df.columns
        if column.lower().startswith(prefix)
    ]

    has_long_format = _find_candidate_column(df, ["investor"]) is not None

    if wide_columns:
        if has_long_format:
            LOGGER.warning("同時偵測到法人長表與寬表欄位，將優先使用寬表資料。")

        selected = [col for col in ["date", "stock_id"] if col in df.columns]
        selected.extend(wide_columns)
        inst_df = df[selected].copy()

        rename_map = {}
        for column in wide_columns:
            lower = column.lower()
            if lower.endswith("_foreign"):
                rename_map[column] = "inst_foreign"
            elif lower.endswith("_investment_trust"):
                rename_map[column] = "inst_investment_trust"
            elif lower.endswith("_dealer_self"):
                rename_map[column] = "inst_dealer_self"
            elif lower.endswith("_dealer_hedging"):
                rename_map[column] = "inst_dealer_hedging"

        inst_df = inst_df.rename(columns=rename_map)

        group_keys = [col for col in ["date", "stock_id"] if col in inst_df.columns]
        agg_map = {
            column: "sum"
            for column in inst_df.columns
            if column not in {"date", "stock_id"}
        }
        if agg_map:
            inst_df = inst_df.groupby(group_keys, as_index=False).agg(agg_map)
        else:
            inst_df = inst_df.drop_duplicates(subset=group_keys)

        for column in empty_inst.columns:
            if column not in inst_df.columns:
                inst_df[column] = np.nan
        return inst_df[empty_inst.columns]

    if not has_long_format:
        LOGGER.warning("偵測不到法人欄位，僅輸出價量資料。")
        return empty_inst

    investor_col = _find_candidate_column(df, ["investor"])
    net_col = _find_candidate_column(df, ["net_buy_sell", "buy_sell"])
    buy_col = _find_candidate_column(df, ["buy"], exclude=[net_col] if net_col else None)
    sell_col = _find_candidate_column(df, ["sell"], exclude=[net_col] if net_col else None)

    if not investor_col or (not net_col and (not buy_col or not sell_col)):
        LOGGER.warning("法人長表欄位資訊不足，無法計算淨買賣超。")
        return empty_inst

    inst_df = df[[col for col in ["date", "stock_id", investor_col] if col in df.columns]].copy()

    if net_col:
        values = pd.to_numeric(df[net_col], errors="coerce")
    else:
        buy_series = pd.to_numeric(df[buy_col], errors="coerce").fillna(0)
        sell_series = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)
        values = buy_series - sell_series

    inst_df["value"] = values
    inst_df = inst_df.dropna(subset=["value"], how="all")

    if inst_df.empty:
        LOGGER.warning("法人長表計算後沒有有效數值。")
        return empty_inst

    def normalize_investor(name: object) -> str:
        text = str(name).strip().lower()
        text = re.sub(r"[^a-z]+", "_", text)
        return text.strip("_")

    investor_map = {
        "foreign_investor": "inst_foreign",
        "foreign": "inst_foreign",
        "investment_trust": "inst_investment_trust",
        "investmenttrust": "inst_investment_trust",
        "dealer_self": "inst_dealer_self",
        "dealerself": "inst_dealer_self",
        "dealer_hedging": "inst_dealer_hedging",
        "dealerhedging": "inst_dealer_hedging",
    }

    inst_df["inst_column"] = inst_df[investor_col].map(
        lambda value: investor_map.get(normalize_investor(value))
    )
    inst_df = inst_df.dropna(subset=["inst_column"])

    if inst_df.empty:
        LOGGER.warning("法人長表的投資人類別無法對應至標準欄位。")
        return empty_inst

    pivot = inst_df.pivot_table(
        index=[col for col in ["date", "stock_id"] if col in inst_df.columns],
        columns="inst_column",
        values="value",
        aggfunc="sum",
    )
    pivot = pivot.reset_index()

    pivot.columns = [
        column if isinstance(column, str) else column[1]
        for column in pivot.columns
    ]

    for column in empty_inst.columns:
        if column not in pivot.columns:
            pivot[column] = np.nan

    return pivot[empty_inst.columns]


def _merge_daily_wide(price_df: pd.DataFrame, inst_df: pd.DataFrame) -> pd.DataFrame:
    """合併價量與法人資料，確保每天每檔僅一列。"""

    base_columns = ["date", "stock_id"]

    if price_df is None or price_df.empty:
        combined = inst_df.copy()
    elif inst_df is None or inst_df.empty:
        combined = price_df.copy()
    else:
        combined = pd.merge(price_df, inst_df, on=base_columns, how="outer")

    if combined.empty:
        return combined

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=False)
        try:
            combined["date"] = combined["date"].dt.tz_localize(None)
        except AttributeError:
            pass

    agg_map = {}
    for column in combined.columns:
        if column in base_columns:
            continue
        if column.startswith("inst_"):
            agg_map[column] = "sum"
        else:
            agg_map[column] = "first"

    if agg_map:
        combined = combined.groupby(base_columns, as_index=False).agg(agg_map)
    else:
        combined = combined.drop_duplicates(subset=base_columns)

    for column in [
        "inst_foreign",
        "inst_investment_trust",
        "inst_dealer_self",
        "inst_dealer_hedging",
    ]:
        if column not in combined.columns:
            combined[column] = np.nan

    return combined


__all__ = [
    "merge_frames",
    "_normalize_types",
    "_extract_price_block",
    "_find_candidate_column",
    "_build_institutional_wide",
    "_merge_daily_wide",
]
