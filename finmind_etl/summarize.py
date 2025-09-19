"""輸出摘要資訊。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _print_summary(df: pd.DataFrame) -> None:
    """在終端機輸出摘要資訊。"""

    print("=== 清理後資料摘要 ===")
    print(f"總筆數: {len(df):,}")

    if "stock_id" in df.columns:
        print(f"股票檔數: {df['stock_id'].nunique():,}")
    else:
        print("股票檔數: 無 stock_id 欄位")

    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
        if date_series.notna().any():
            min_date = date_series.min().strftime("%Y-%m-%d")
            max_date = date_series.max().strftime("%Y-%m-%d")
            print(f"日期範圍: {min_date} ~ {max_date}")
        else:
            print("日期範圍: 無有效日期")
    else:
        print("日期範圍: 無 date 欄位")

    print("-- 欄位缺值比例 --")
    nan_ratio = df.isna().mean()
    for column, ratio in nan_ratio.items():
        print(f"{column}: {ratio:.2%}")

    if df.empty:
        print("資料為空，無樣本可顯示。")
        return

    sample_is_head = np.random.rand() < 0.5
    label = "前 3 筆" if sample_is_head else "後 3 筆"
    sample = df.head(3) if sample_is_head else df.tail(3)
    print(f"-- 隨機樣本 ({label}) --")
    print(sample.to_string(index=False))


__all__ = ["_print_summary"]
