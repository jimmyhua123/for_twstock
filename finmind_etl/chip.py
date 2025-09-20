"""籌碼面資料抓取模組。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .api import APIClient
from .technical import _apply_translation, _ensure_datetime, _ensure_stock_id, _numeric

LOGGER = logging.getLogger("finmind_etl.chip")


def _rename_with_candidates(df: pd.DataFrame, candidates: Dict[str, str]) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        lower = column.lower()
        for pattern, target in candidates.items():
            if pattern in lower:
                rename_map[column] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def fetch_institutional(
    stocks: Sequence[str],
    client: APIClient,
    start: str,
    end: str,
) -> pd.DataFrame:
    """抓取三大法人買賣超資料。"""

    translation = client.try_translation("TaiwanStockInstitutionalInvestorsBuySell")
    frames: List[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.fetch_dataset(
                "TaiwanStockInstitutionalInvestorsBuySell",
                stock,
                start,
                end,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("法人資料抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            continue
        df = _apply_translation(df, translation)
        df = _ensure_datetime(df)
        df = _ensure_stock_id(df)
        df = _rename_with_candidates(
            df,
            {
                "foreign_investor_net": "foreign_net",
                "foreign_net_buy_sell": "foreign_net",
                "foreign": "foreign_net",
                "investment_trust_net": "invest_trust_net",
                "investment_trust": "invest_trust_net",
                "dealer_self": "dealer_self_net",
                "dealer_hedging": "dealer_hedging_net",
                "dealer_net": "dealer_net",
                "dealer": "dealer_net",
            },
        )
        frames.append(df)
    if not frames:
        columns = [
            "date",
            "stock_id",
            "foreign_net",
            "invest_trust_net",
            "dealer_net",
            "dealer_self_net",
            "dealer_hedging_net",
        ]
        return pd.DataFrame(columns=columns)
    merged = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "foreign_net",
        "invest_trust_net",
        "dealer_net",
        "dealer_self_net",
        "dealer_hedging_net",
    ]
    _numeric(merged, numeric_columns)
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    for column in numeric_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[["date", "stock_id", *numeric_columns]]


def fetch_margin_short(
    stocks: Sequence[str],
    client: APIClient,
    start: str,
    end: str,
) -> pd.DataFrame:
    """抓取融資融券資料。"""

    translation = client.try_translation("TaiwanStockMarginPurchaseShortSale")
    frames: List[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.fetch_dataset(
                "TaiwanStockMarginPurchaseShortSale",
                stock,
                start,
                end,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("融資融券抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            continue
        df = _apply_translation(df, translation)
        df = _ensure_datetime(df)
        df = _ensure_stock_id(df)
        df = _rename_with_candidates(
            df,
            {
                "margin_purchase_today_balance": "margin_long",
                "margin_purchase_change": "margin_long_change",
                "short_sale_today_balance": "margin_short",
                "short_sale_change": "margin_short_change",
                "short_sale_volume": "short_selling",
                "short_sale": "short_selling",
            },
        )
        frames.append(df)
    if not frames:
        columns = [
            "date",
            "stock_id",
            "margin_long",
            "margin_short",
            "margin_long_change",
            "margin_short_change",
            "short_selling",
        ]
        return pd.DataFrame(columns=columns)
    merged = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "margin_long",
        "margin_short",
        "margin_long_change",
        "margin_short_change",
        "short_selling",
    ]
    _numeric(merged, numeric_columns)
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    for column in numeric_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[["date", "stock_id", *numeric_columns]]


def fetch_chip_data(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """取得籌碼面資料。"""

    end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    LOGGER.info("抓取籌碼面資料：股票數量=%s", len(stocks))
    institutional = fetch_institutional(stocks, client, since, end)
    margin = fetch_margin_short(stocks, client, since, end)
    return {
        "TaiwanStockInstitutionalInvestorsBuySell": institutional,
        "TaiwanStockMarginPurchaseShortSale": margin,
    }


__all__ = ["fetch_chip_data", "fetch_institutional", "fetch_margin_short"]
