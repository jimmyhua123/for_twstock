"""計算技術分析指標。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import get_logger
from .utils import clip_score, percent_change, rolling_slope

LOGGER = get_logger(__name__)

MA_WINDOWS = (5, 10, 20, 60, 120)
EMA_SHORT = 12
EMA_LONG = 26
EMA_SIGNAL = 9
RSI_PERIODS = (6, 14)
BOLL_WINDOW = 20
BOLL_STD = 2
SLOPE_WINDOW = 5
VOL_WINDOWS = (5, 20)


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _calc_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _bollinger(close: pd.Series) -> dict[str, pd.Series]:
    ma = _moving_average(close, BOLL_WINDOW)
    std = close.rolling(window=BOLL_WINDOW, min_periods=BOLL_WINDOW).std()
    return {
        "BB_MA20": ma,
        "BB_UP": ma + BOLL_STD * std,
        "BB_DOWN": ma - BOLL_STD * std,
    }


def _volume_ma(volume: pd.Series, window: int) -> pd.Series:
    return volume.rolling(window=window, min_periods=window).mean()


def _score_technical(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(2.5, index=df.index, dtype=float)

    ma20 = df["MA20"]
    ma60 = df["MA60"]
    close = df["close"]

    score += np.where((close > ma20) & (ma20 > ma60), 1.0, 0.0)
    score += np.where((close < ma20) & (ma20 < ma60), -1.0, 0.0)

    ma20_prev = ma20.shift(1)
    ma60_prev = ma60.shift(1)
    golden = (ma20 > ma60) & (ma20_prev <= ma60_prev)
    death = (ma20 < ma60) & (ma20_prev >= ma60_prev)
    score += np.where(golden, 0.5, 0.0)
    score += np.where(death, -0.5, 0.0)

    ma5 = df["MA5"]
    ma5_prev = ma5.shift(1)
    ma20_prev2 = ma20.shift(1)
    golden_short = (ma5 > ma20) & (ma5_prev <= ma20_prev2)
    death_short = (ma5 < ma20) & (ma5_prev >= ma20_prev2)
    score += np.where(golden_short, 0.3, 0.0)
    score += np.where(death_short, -0.3, 0.0)

    rsi14 = df["RSI14"]
    score += np.where(rsi14 > 70, -0.7, 0.0)
    score += np.where(rsi14 < 30, 0.4, 0.0)

    boll_mid = df["BB_MA20"]
    score += np.where((close >= boll_mid) & close.notna(), 0.3, 0.0)
    score += np.where(close > df["BB_UP"] * 1.03, -0.4, 0.0)
    score += np.where(close < df["BB_DOWN"] * 0.97, 0.2, 0.0)

    volume_ratio = df["volume"] / df["VOL_MA20"]
    score += np.where(volume_ratio >= 1.5, 0.5, 0.0)
    score += np.where(volume_ratio <= 0.8, -0.3, 0.0)

    return score.apply(lambda v: clip_score(v, 0.0, 5.0))


def calculate_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """為單一股票計算技術指標並返回新資料表。"""

    df = stock_df.copy()

    for window in MA_WINDOWS:
        df[f"MA{window}"] = _moving_average(df["close"], window)

    for window in RSI_PERIODS:
        df[f"RSI{window}"] = _calc_rsi(df["close"], window)

    for window in VOL_WINDOWS:
        df[f"VOL_MA{window}"] = _volume_ma(df["volume"], window)

    ema_short = _ema(df["close"], EMA_SHORT)
    ema_long = _ema(df["close"], EMA_LONG)
    df["EMA12"] = ema_short
    df["EMA26"] = ema_long
    df["DIF"] = ema_short - ema_long
    df["DEA"] = _ema(df["DIF"], EMA_SIGNAL)
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    boll = _bollinger(df["close"])
    for key, value in boll.items():
        df[key] = value

    df["MA20_slope"] = rolling_slope(df["MA20"], SLOPE_WINDOW)
    df["MA60_slope"] = rolling_slope(df["MA60"], SLOPE_WINDOW)

    df["pct_5d"] = percent_change(df["close"], 5)
    df["pct_10d"] = percent_change(df["close"], 10)
    df["pct_20d"] = percent_change(df["close"], 20)

    df["tech_score"] = _score_technical(df)

    return df


if __name__ == "__main__":  # pragma: no cover - 模組自測
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    close = pd.Series(np.linspace(100, 150, len(dates)))
    volume = pd.Series(np.random.randint(1000, 5000, size=len(dates)))
    sample = pd.DataFrame({
        "date": dates,
        "close": close,
        "volume": volume,
    })
    enriched = calculate_indicators(sample)
    assert "MA20" in enriched.columns
    assert enriched["tech_score"].iloc[-1] >= 0
    print("indicators 自測完成")
