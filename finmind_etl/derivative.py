"""衍生性商品資料抓取模組。"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import pandas as pd

from .api import APIClient
from .technical import _apply_translation, _ensure_datetime, _to_snake_case

LOGGER = logging.getLogger("finmind_etl.derivative")


def _prefix_columns(df: pd.DataFrame, prefix: str, exclude: Sequence[str]) -> pd.DataFrame:
    rename_map = {
        column: f"{prefix}{_to_snake_case(column)}"
        for column in df.columns
        if column not in exclude
    }
    return df.rename(columns=rename_map)


def _ensure_numeric_all(df: pd.DataFrame, exclude: Sequence[str]) -> None:
    for column in df.columns:
        if column in exclude:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")


def fetch_futures_data(
    client: APIClient,
    start: str,
    end: str,
    contract: str = "TX",
) -> pd.DataFrame:
    """抓取期貨三大法人資料。"""

    try:
        df = client.fetch_dataset("TaiwanFuturesInstitutionalInvestors", contract, start, end)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("期貨資料抓取失敗：%s", exc)
        return pd.DataFrame(columns=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date"])
    translation = client.try_translation("TaiwanFuturesInstitutionalInvestors")
    df = _apply_translation(df, translation)
    df = _ensure_datetime(df)
    df = _prefix_columns(df, "fut_", ["date", "stock_id", "contract", "code"])
    _ensure_numeric_all(df, ["date", "contract", "code", "stock_id"])
    df = df.drop(columns=[col for col in df.columns if col.endswith("contract") or col.endswith("code")], errors="ignore")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_option_data(
    client: APIClient,
    start: str,
    end: str,
    contract: str = "TXO",
) -> pd.DataFrame:
    """抓取選擇權三大法人資料。"""

    try:
        df = client.fetch_dataset("TaiwanOptionInstitutionalInvestors", contract, start, end)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("選擇權資料抓取失敗：%s", exc)
        return pd.DataFrame(columns=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date"])
    translation = client.try_translation("TaiwanOptionInstitutionalInvestors")
    df = _apply_translation(df, translation)
    df = _ensure_datetime(df)
    df = _prefix_columns(df, "opt_", ["date", "stock_id", "contract", "code"])
    _ensure_numeric_all(df, ["date", "contract", "code", "stock_id"])
    df = df.drop(columns=[col for col in df.columns if col.endswith("contract") or col.endswith("code")], errors="ignore")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_derivative_data(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """取得衍生性商品資料。"""

    del stocks  # 衍生性資料多為指數層級，暫不區分個股
    end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    LOGGER.info("抓取衍生性資料")
    futures = fetch_futures_data(client, since, end)
    options = fetch_option_data(client, since, end)
    return {
        "TaiwanFuturesInstitutionalInvestors": futures,
        "TaiwanOptionInstitutionalInvestors": options,
    }


__all__ = ["fetch_derivative_data"]
