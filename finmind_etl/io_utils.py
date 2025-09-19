"""輸入輸出相關工具函式。"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .config import LOGGER


def save_frame(
    df: pd.DataFrame,
    path_csv: str,
    path_parquet: Optional[str] = None,
) -> None:
    """儲存 DataFrame 為 CSV（以及 Parquet）。"""

    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    LOGGER.info("輸出路徑 (CSV)：%s", path_csv)

    if df.empty:
        LOGGER.info(
            "資料為空，仍輸出表頭供參考，欄位：%s", list(df.columns)
        )
    else:
        preview = df.head(3).to_string(index=False)
        LOGGER.info("資料筆數：%d，欄位：%s", len(df), list(df.columns))
        LOGGER.info("前 3 筆預覽：\n%s", preview)

    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    if path_parquet:
        LOGGER.info("輸出路徑 (Parquet)：%s", path_parquet)
        os.makedirs(os.path.dirname(path_parquet), exist_ok=True)
        df.to_parquet(path_parquet, index=False)


def _read_raw_merged(path: str) -> pd.DataFrame:
    """讀取外部提供的合併檔案。"""

    if not os.path.exists(path):
        LOGGER.warning("找不到檔案 %s，跳過清理流程。", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("讀取 %s 時發生錯誤: %s", path, exc)
        return pd.DataFrame()

    if df.empty:
        LOGGER.warning("檔案 %s 為空，無資料可清理。", path)
    return df


__all__ = ["save_frame", "_read_raw_merged"]
