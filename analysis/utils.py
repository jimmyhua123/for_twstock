"""提供分析模組共用工具函式。"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TAIPEI_TZ = "Asia/Taipei"


def ensure_directory(path: Path | str) -> Path:
    """確保輸出目錄存在並返回路徑物件。

    參數
    ----
    path:
        目標目錄路徑。

    返回
    ----
    Path
        已建立的目錄路徑。
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_taipei_datetime(series: pd.Series) -> pd.Series:
    """將日期序列轉換為台北時區的 `Timestamp`。

    若原始值不含時區，會以 `Asia/Taipei` 進行本地化。
    """

    dt_series = pd.to_datetime(series, errors="coerce")
    if getattr(dt_series.dt, "tz", None) is None:
        return dt_series.dt.tz_localize(TAIPEI_TZ)
    return dt_series.dt.tz_convert(TAIPEI_TZ)


def format_date(date: pd.Timestamp | datetime | str) -> str:
    """將日期物件格式化為 `YYYY-MM-DD` 字串。"""

    if isinstance(date, str):
        return date[:10]
    if isinstance(date, pd.Timestamp):
        if date.tzinfo is not None:
            date = date.tz_convert(TAIPEI_TZ)
        return date.strftime("%Y-%m-%d")
    if isinstance(date, datetime):
        return date.strftime("%Y-%m-%d")
    raise TypeError(f"不支援的日期型別: {type(date)!r}")


def clip_score(value: float, lower: float, upper: float) -> float:
    """將分數限制於指定上下限範圍內。"""

    if math.isnan(value):
        return lower
    return max(lower, min(upper, value))


def safe_divide(numerator: float, denominator: float) -> float:
    """執行安全除法，若分母為 0 則回傳 0。"""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """以最小平方法估計滾動斜率。

    斜率代表單位日的變化量，採用簡化線性回歸。
    """

    if window <= 1:
        raise ValueError("window 必須大於 1")

    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        y = np.array(values, dtype=float)
        y_mean = np.nanmean(y)
        cov = np.nansum((x - x_mean) * (y - y_mean))
        if x_var == 0:
            return 0.0
        return cov / x_var

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def consecutive_days(values: Iterable[float], positive: bool = True) -> int:
    """計算序列尾端連續正值或負值的天數。"""

    count = 0
    for value in reversed(list(values)):
        if positive and value > 0:
            count += 1
        elif (not positive) and value < 0:
            count += 1
        else:
            break
    return count


def last_turning_point(series: pd.Series) -> Optional[pd.Timestamp]:
    """取得最近一次正負號轉換的日期。"""

    signs = np.sign(series.fillna(0).to_numpy())
    if len(signs) < 2:
        return None
    for idx in range(len(signs) - 1, 0, -1):
        if signs[idx] == 0:
            continue
        prev_idx = idx - 1
        while prev_idx >= 0 and signs[prev_idx] == 0:
            prev_idx -= 1
        if prev_idx >= 0 and signs[idx] != signs[prev_idx]:
            return series.index[idx]
    return None


def percentile_rank(values: pd.Series) -> pd.Series:
    """計算 0–1 之間的百分位排名。"""

    return values.rank(pct=True, method="average")


def percent_change(series: pd.Series, periods: int) -> pd.Series:
    """計算指定期數的百分比變化。"""

    return series.pct_change(periods=periods)


if __name__ == "__main__":  # pragma: no cover - 自行檢查
    sample = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5))
    slope = rolling_slope(sample, window=3)
    assert slope.iloc[-1] > 0
    assert consecutive_days([1, 2, -1, 3], positive=True) == 1
    assert consecutive_days([1, 2, 3], positive=True) == 3
    assert consecutive_days([-1, -2, -3], positive=False) == 3
    print("utils 自測完成")
