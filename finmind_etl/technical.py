"""技術面資料抓取模組。"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .api import APIClient

LOGGER = logging.getLogger("finmind_etl.technical")


def _to_snake_case(value: str) -> str:
    import re

    text = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def _apply_translation(df: pd.DataFrame, translation: Dict[str, str]) -> pd.DataFrame:
    if not translation:
        df = df.rename(columns={col: _to_snake_case(col) for col in df.columns})
        return df
    rename_map = {col: translation.get(col, translation.get(col.lower(), _to_snake_case(col))) for col in df.columns}
    return df.rename(columns=rename_map)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def _ensure_stock_id(df: pd.DataFrame) -> pd.DataFrame:
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    return df


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def fetch_stock_info(client: APIClient) -> pd.DataFrame:
    """取得基本股名表。"""

    try:
        df = client.fetch_dataset("TaiwanStockInfo", None, None, None)
    except Exception as exc:  # noqa: BLE001 - 外部 API 可能失敗
        LOGGER.warning("TaiwanStockInfo 抓取失敗：%s", exc)
        return pd.DataFrame(columns=["stock_id", "industry_category", "stock_name"])

    translation = client.try_translation("TaiwanStockInfo")
    df = _apply_translation(df, translation)
    df = _ensure_stock_id(df)
    return df.drop_duplicates(subset=["stock_id"])


def fetch_trading_calendar(client: APIClient, start: str, end: str) -> pd.DataFrame:
    """取得交易日曆。"""

    try:
        calendar = client.fetch_dataset("TaiwanStockTradingDate", None, start, end)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("TaiwanStockTradingDate 抓取失敗：%s", exc)
        return pd.DataFrame(columns=["date", "is_trading_day"])

    translation = client.try_translation("TaiwanStockTradingDate")
    calendar = _apply_translation(calendar, translation)
    calendar = _ensure_datetime(calendar)
    if "is_trading_day" not in calendar.columns and "trading" in calendar.columns:
        calendar = calendar.rename(columns={"trading": "is_trading_day"})
    if "is_trading_day" in calendar.columns:
        calendar["is_trading_day"] = calendar["is_trading_day"].astype(int)
    calendar = calendar.sort_values("date").reset_index(drop=True)
    return calendar


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "max": "high",
        "min": "low",
        "trading_volume": "volume",
        "trading_value": "turnover",
        "trading_money": "turnover",
        "trading_turnover": "transactions",
        "spread": "price_change",
    }
    df = df.rename(columns=rename_map)
    base_columns = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"]
    for column in base_columns:
        if column not in df.columns:
            df[column] = np.nan
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "transactions",
        "price_change",
    ]
    _numeric(df, numeric_columns)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    if "close" in df.columns:
        df["return"] = df.groupby("stock_id")["close"].pct_change()
    return df


def fetch_price_dataset(
    stocks: Sequence[str],
    client: APIClient,
    dataset: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """共用的價格資料抓取函式。"""

    frames: List[pd.DataFrame] = []
    translation = client.try_translation(dataset)
    for stock in stocks:
        try:
            df = client.fetch_dataset(dataset, stock, start, end)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s 抓取失敗：%s %s", dataset, stock, exc)
            continue
        if df.empty:
            continue
        df = _apply_translation(df, translation)
        df = _ensure_datetime(df)
        df = _ensure_stock_id(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "stock_id"])
    combined = pd.concat(frames, ignore_index=True)
    combined = _prepare_price_frame(combined)
    return combined


def fetch_technical_data(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """抓取主要技術面資料集。"""

    end = end_date or date.today().strftime("%Y-%m-%d")
    LOGGER.info("抓取技術面資料：股票數量=%s", len(stocks))

    results: Dict[str, pd.DataFrame] = {}
    results["TaiwanStockInfo"] = fetch_stock_info(client)
    results["TaiwanStockTradingDate"] = fetch_trading_calendar(client, since, end)
    results["TaiwanStockPrice"] = fetch_price_dataset(stocks, client, "TaiwanStockPrice", since, end)
    results["TaiwanStockPriceAdj"] = fetch_price_dataset(stocks, client, "TaiwanStockPriceAdj", since, end)
    return results


__all__ = ["fetch_technical_data"]
