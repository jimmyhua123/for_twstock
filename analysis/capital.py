"""計算資金與盤勢熱度指標。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import get_logger
from .utils import clip_score, percentile_rank

LOGGER = get_logger(__name__)
ROLLING_WINDOW = 5


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "turnover",
        "volume",
        "TaiwanStockPrice_transactions",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def calculate_capital_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """計算資金熱度相關欄位。"""

    df = _prepare(df)

    df["turnover_rank_pct"] = df.groupby("date")[["turnover"]].transform(percentile_rank)

    df["turnover_ma5"] = df.groupby("stock_id")[["turnover"]].transform(
        lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    df["volume_ma5"] = df.groupby("stock_id")[["volume"]].transform(
        lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    df["transactions_ma5"] = df.groupby("stock_id")[["TaiwanStockPrice_transactions"]].transform(
        lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )

    df["turnover_change_5d"] = (df["turnover"] - df["turnover_ma5"]) / df["turnover_ma5"].replace(0, np.nan)
    df["volume_change_5d"] = (df["volume"] - df["volume_ma5"]) / df["volume_ma5"].replace(0, np.nan)
    df["transactions_change_5d"] = (
        (df["TaiwanStockPrice_transactions"] - df["transactions_ma5"]) / df["transactions_ma5"].replace(0, np.nan)
    )

    df[["turnover_change_5d", "volume_change_5d", "transactions_change_5d"]] = df[
        ["turnover_change_5d", "volume_change_5d", "transactions_change_5d"]
    ].fillna(0)

    df["capital_score"] = _score_capital(df)
    return df


def _score_capital(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(1.5, index=df.index, dtype=float)

    score += np.where(df["turnover_rank_pct"] >= 0.7, 0.8, 0.0)
    score += np.where(df["turnover_rank_pct"] <= 0.3, -0.8, 0.0)

    score += np.where(df["turnover_change_5d"] >= 0.2, 0.6, 0.0)
    score += np.where(df["turnover_change_5d"] <= -0.2, -0.6, 0.0)

    score += np.where(df["volume_change_5d"] >= 0.2, 0.5, 0.0)
    score += np.where(df["volume_change_5d"] <= -0.2, -0.5, 0.0)

    score += np.where(df["transactions_change_5d"] >= 0.15, 0.3, 0.0)
    score += np.where(df["transactions_change_5d"] <= -0.15, -0.3, 0.0)

    score += np.where(df["turnover_rank_pct"] >= 0.9, 0.2, 0.0)
    score += np.where(df["turnover_rank_pct"] <= 0.1, -0.2, 0.0)

    return score.apply(lambda v: clip_score(v, 0.0, 3.0))


if __name__ == "__main__":  # pragma: no cover - 自測
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "stock_id": ["2330"] * 10 + ["2317"] * 10,
            "turnover": np.random.randint(1_000_000, 5_000_000, size=20),
            "volume": np.random.randint(1_000, 5_000, size=20),
            "TaiwanStockPrice_transactions": np.random.randint(100, 500, size=20),
        }
    )
    enriched = calculate_capital_metrics(df)
    assert "capital_score" in enriched.columns
    print("capital 自測完成")
