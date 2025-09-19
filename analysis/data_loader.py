"""資料載入與清理流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import get_logger
from .utils import TAIPEI_TZ, to_taipei_datetime

LOGGER = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "stock_id",
    "volume",
    "turnover",
    "open",
    "high",
    "low",
    "close",
    "TaiwanStockPrice_spread",
    "TaiwanStockPrice_transactions",
    "inst_foreign",
    "inst_investment_trust",
    "inst_dealer_self",
    "inst_dealer_hedging",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "volume",
    "turnover",
    "open",
    "high",
    "low",
    "close",
    "TaiwanStockPrice_spread",
    "TaiwanStockPrice_transactions",
    "inst_foreign",
    "inst_investment_trust",
    "inst_dealer_self",
    "inst_dealer_hedging",
)


class MissingColumnsError(RuntimeError):
    """缺欄位時拋出的例外。"""


def _validate_columns(columns: Iterable[str]) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise MissingColumnsError(f"檔案缺少必要欄位: {missing}")
    return list(columns)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除重複列與明顯異常值。"""

    before = len(df)
    df = df.drop_duplicates(subset=["date", "stock_id"], keep="last")
    if len(df) != before:
        LOGGER.info("移除 %s 筆重複資料", before - len(df))

    # 移除價格或量為負值的資料
    mask_invalid = (
        (df["volume"] < 0)
        | (df["turnover"] < 0)
        | (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df["low"])
    )
    invalid_count = int(mask_invalid.sum())
    if invalid_count > 0:
        LOGGER.warning("偵測到 %s 筆異常資料，已移除", invalid_count)
        df = df.loc[~mask_invalid]
    return df


def load_wide_csv(path: str | Path) -> pd.DataFrame:
    """載入清理後的寬表資料。

    返回的資料依 `stock_id`, `date` 排序，日期欄位為台北時區。
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {path}")

    LOGGER.info("讀取資料檔案: %s", path)
    df = pd.read_csv(path)

    _validate_columns(df.columns)
    df = _coerce_numeric(df)

    df["date"] = to_taipei_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(4)

    df = _clean_rows(df)

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    LOGGER.info("資料列數: %s，股票數量: %s", len(df), df["stock_id"].nunique())
    return df


if __name__ == "__main__":  # pragma: no cover - 簡單自測
    try:
        sample_path = Path("_clean_daily_wide.csv")
        if sample_path.exists():
            sample_df = load_wide_csv(sample_path)
            print(sample_df.head())
        else:
            print("找不到 _clean_daily_wide.csv，僅確認模組可引用")
    except Exception as exc:  # noqa: BLE001 - 自測需攔截所有例外
        print(f"自測失敗: {exc}")
