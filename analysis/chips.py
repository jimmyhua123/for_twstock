"""計算三大法人籌碼相關指標。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import get_logger
from .utils import clip_score, last_turning_point

LOGGER = get_logger(__name__)
ROLLING_WINDOWS = (5, 10, 20)


@dataclass
class ChipSummary:
    """封裝籌碼面摘要。"""

    score: float
    consecutive_buy: int
    consecutive_sell: int
    turning_point: str | None
    note: str


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).sum()


def _consecutive_positive(series: pd.Series) -> pd.Series:
    mask = series > 0
    groups = (mask != mask.shift(fill_value=False)).cumsum()
    streak = mask.groupby(groups).cumsum()
    return streak.where(mask, 0)


def _consecutive_negative(series: pd.Series) -> pd.Series:
    mask = series < 0
    groups = (mask != mask.shift(fill_value=False)).cumsum()
    streak = mask.groupby(groups).cumsum()
    return streak.where(mask, 0)


def _score_chip(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(2.5, index=df.index, dtype=float)
    for window in ROLLING_WINDOWS:
        total = df[f"net_all_{window}"]
        score += np.where(total > 0, 0.8, 0.0)
        score += np.where(total < 0, -0.8, 0.0)

    score += np.where(df["net_all_consecutive_buy"] >= 3, 0.8, 0.0)
    score += np.where(df["net_all_consecutive_sell"] >= 3, -0.8, 0.0)

    score += np.where(
        (df["foreign_consecutive_buy"] >= 3) | (df["it_consecutive_buy"] >= 3),
        0.6,
        0.0,
    )
    score += np.where(
        (df["foreign_consecutive_sell"] >= 3) | (df["it_consecutive_sell"] >= 3),
        -0.6,
        0.0,
    )
    return score.apply(lambda v: clip_score(v, 0.0, 5.0))


def calculate_chip_metrics(stock_df: pd.DataFrame) -> pd.DataFrame:
    """為單一股票新增籌碼相關欄位。"""

    df = stock_df.copy()

    foreign = df["inst_foreign"].fillna(0)
    it = df["inst_investment_trust"].fillna(0)
    dealer_self = df["inst_dealer_self"].fillna(0)
    dealer_hedge = df["inst_dealer_hedging"].fillna(0)

    df["net_dealer_total"] = dealer_self + dealer_hedge
    df["net_all"] = foreign + it + df["net_dealer_total"]

    for window in ROLLING_WINDOWS:
        df[f"net_foreign_{window}"] = _rolling_sum(foreign, window)
        df[f"net_it_{window}"] = _rolling_sum(it, window)
        df[f"net_dealer_total_{window}"] = _rolling_sum(df["net_dealer_total"], window)
        df[f"net_all_{window}"] = _rolling_sum(df["net_all"], window)

    df["foreign_consecutive_buy"] = _consecutive_positive(foreign)
    df["it_consecutive_buy"] = _consecutive_positive(it)
    df["dealer_consecutive_buy"] = _consecutive_positive(df["net_dealer_total"])
    df["net_all_consecutive_buy"] = _consecutive_positive(df["net_all"])

    df["foreign_consecutive_sell"] = _consecutive_negative(foreign)
    df["it_consecutive_sell"] = _consecutive_negative(it)
    df["dealer_consecutive_sell"] = _consecutive_negative(df["net_dealer_total"])
    df["net_all_consecutive_sell"] = _consecutive_negative(df["net_all"])

    df["chip_score"] = _score_chip(df)

    turn_point = last_turning_point(df.set_index("date")["net_all"])
    if turn_point is not None:
        LOGGER.debug("股票 %s 最近轉折日: %s", df["stock_id"].iloc[0], turn_point)

    return df


def summarize_chip(df: pd.DataFrame) -> ChipSummary:
    """根據最新資料形成籌碼摘要。"""

    latest = df.iloc[-1]
    turning_point = last_turning_point(df.set_index("date")["net_all"])
    turning_str = turning_point.strftime("%Y-%m-%d") if turning_point else None

    consecutive_buy = int(latest["net_all_consecutive_buy"])
    consecutive_sell = int(latest["net_all_consecutive_sell"])

    notes: list[str] = []
    if latest["foreign_consecutive_buy"] >= 3:
        notes.append(f"外資連買 {int(latest['foreign_consecutive_buy'])} 日")
    if latest["it_consecutive_buy"] >= 3:
        notes.append(f"投信連買 {int(latest['it_consecutive_buy'])} 日")
    if latest["foreign_consecutive_sell"] >= 3:
        notes.append(f"外資連賣 {int(latest['foreign_consecutive_sell'])} 日")
    if latest["it_consecutive_sell"] >= 3:
        notes.append(f"投信連賣 {int(latest['it_consecutive_sell'])} 日")

    note = "、".join(notes) if notes else "法人動向無明顯連續趨勢"

    return ChipSummary(
        score=float(latest["chip_score"]),
        consecutive_buy=consecutive_buy,
        consecutive_sell=consecutive_sell,
        turning_point=turning_str,
        note=note,
    )


if __name__ == "__main__":  # pragma: no cover - 自測
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    sample = pd.DataFrame(
        {
            "date": dates,
            "stock_id": "2330",
            "inst_foreign": np.random.randint(-1000, 1000, size=len(dates)),
            "inst_investment_trust": np.random.randint(-500, 500, size=len(dates)),
            "inst_dealer_self": np.random.randint(-200, 200, size=len(dates)),
            "inst_dealer_hedging": np.random.randint(-200, 200, size=len(dates)),
        }
    )
    enriched = calculate_chip_metrics(sample)
    summary = summarize_chip(enriched)
    assert summary.score >= 0
    print("chips 自測完成", summary)
