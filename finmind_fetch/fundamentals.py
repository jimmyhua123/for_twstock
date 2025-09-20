"""FinMind 基本面資料取得與日頻對齊。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import get_logger
from .api import FinMindClient

LOGGER = get_logger(__name__)
TARGET_FIN_TYPES = {"eps", "revenue", "grossprofit", "operatingincome", "netincome"}


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def fetch_month_revenue(
    stocks: Sequence[str],
    start: str,
    end: str,
    client: FinMindClient,
) -> pd.DataFrame:
    """抓取月營收並計算年增率、月增率。

    參數
    ----
    stocks:
        股票代號清單。
    start, end:
        抓取區間，日期格式 ``YYYY-MM-DD``。
    client:
        事先建好的 :class:`FinMindClient` 實例。
    """

    frames: list[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.get_dataset(
                "TaiwanStockMonthRevenue",
                {"data_id": stock, "start_date": start, "end_date": end},
            )
        except Exception as exc:  # noqa: BLE001 - 外部 API 可能失敗
            LOGGER.warning("月營收抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            LOGGER.warning("月營收無資料：%s", stock)
            continue
        df = _ensure_datetime(df)
        df = df.sort_values("date")
        df["revenue"] = pd.to_numeric(df.get("revenue"), errors="coerce")
        df["revenue_yoy"] = df.groupby("stock_id")["revenue"].pct_change(periods=12)
        df["revenue_mom"] = df.groupby("stock_id")["revenue"].pct_change()
        df[["revenue_yoy", "revenue_mom"]] = df[["revenue_yoy", "revenue_mom"]].replace([np.inf, -np.inf], np.nan)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "revenue", "revenue_yoy", "revenue_mom"])
    return pd.concat(frames, ignore_index=True)


def fetch_eps_and_income(
    stocks: Sequence[str],
    start: str,
    end: str,
    client: FinMindClient,
) -> pd.DataFrame:
    """抓取財報資料並轉為寬表，至少包含 EPS。

    會過濾 ``type`` 屬於 EPS、Revenue、GrossProfit、OperatingIncome、
    NetIncome 等常用欄位，最後輸出 ``stock_id``、``date`` 與指標欄位
    的寬表，並額外計算 ``eps_ttm``（最近四季 EPS 累計）。
    """

    frames: list[pd.DataFrame] = []
    for stock in stocks:
        try:
            df = client.get_dataset(
                "TaiwanStockFinancialStatements",
                {"data_id": stock, "start_date": start, "end_date": end},
            )
        except Exception as exc:  # noqa: BLE001 - 外部 API 可能失敗
            LOGGER.warning("財報抓取失敗：%s %s", stock, exc)
            continue
        if df.empty:
            LOGGER.warning("財報無資料：%s", stock)
            continue
        df = _ensure_datetime(df)
        df["type"] = df["type"].astype(str).str.lower()
        df = df[df["type"].isin(TARGET_FIN_TYPES)]
        if df.empty:
            continue
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "eps"])
    merged = pd.concat(frames, ignore_index=True)
    pivot = (
        merged.pivot_table(
            index=["stock_id", "date"],
            columns="type",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns = [col if isinstance(col, str) else "_".join(col).strip() for col in pivot.columns]
    rename_map = {col: col.lower() for col in pivot.columns if isinstance(col, str)}
    pivot = pivot.rename(columns=rename_map)
    for column in pivot.columns:
        if column not in {"stock_id", "date"}:
            pivot[column] = pd.to_numeric(pivot[column], errors="coerce")
    if "eps" in pivot.columns:
        pivot["eps_ttm"] = pivot.groupby("stock_id")["eps"].transform(
            lambda s: s.rolling(window=4, min_periods=1).sum()
        )
    return pivot


def align_monthly_to_daily(
    month_df: pd.DataFrame,
    trading_daily: pd.DataFrame,
    strategy: str = "forward_fill",
) -> pd.DataFrame:
    """將月/季資料對齊至每日交易日。

    預設採 ``forward_fill`` 策略：以資料發布日為基準，向後填補至下一次
    公布日，確保分析期間內每日都有對應的基本面欄位。若設定為
    ``month_end``，則僅保留各月份的最後一個交易日數值。
    """

    if month_df.empty:
        if trading_daily.empty:
            return pd.DataFrame(columns=["stock_id", "date"])
        return trading_daily[["stock_id", "date"]].copy()

    month_df = month_df.copy()
    month_df["date"] = pd.to_datetime(month_df["date"], errors="coerce")
    month_df = month_df.dropna(subset=["date"])
    trading_daily = trading_daily.copy()
    trading_daily["date"] = pd.to_datetime(trading_daily["date"], errors="coerce")
    trading_daily = trading_daily.dropna(subset=["date"])

    month_df = month_df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    trading_daily = trading_daily.sort_values(["stock_id", "date"]).reset_index(drop=True)

    if month_df.empty or trading_daily.empty:
        return pd.DataFrame(columns=list(month_df.columns))

    aligned_frames: list[pd.DataFrame] = []
    for stock_id, daily_group in trading_daily.groupby("stock_id"):
        monthly_group = month_df[month_df["stock_id"] == stock_id]
        if monthly_group.empty:
            empty = daily_group[["stock_id", "date"]].copy()
            for column in month_df.columns:
                if column not in {"stock_id", "date"}:
                    empty[column] = np.nan
            aligned_frames.append(empty)
            continue
        merged = pd.merge_asof(
            daily_group,
            monthly_group.sort_values("date"),
            on="date",
            by="stock_id",
            direction="backward",
        )
        merged.index = daily_group.index
        if strategy == "month_end":
            month_end_dates = daily_group.groupby(daily_group["date"].dt.to_period("M"))["date"].transform("max")
            month_end_mask = month_end_dates == merged["date"]
            value_columns = [col for col in merged.columns if col not in {"stock_id", "date"}]
            merged.loc[~month_end_mask, value_columns] = np.nan
        aligned_frames.append(merged)

    result = pd.concat(aligned_frames, ignore_index=True)
    result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return result


def prepare_fundamental_daily(
    revenue_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    trading_daily: pd.DataFrame,
    strategy: str = "forward_fill",
) -> pd.DataFrame:
    """整合月營收與財報資料並對齊至每日。

    回傳的結果至少包含 ``stock_id``、``date``、``revenue``、
    ``revenue_yoy``、``revenue_mom``、``eps``、``eps_ttm`` 等欄位，
    可直接與每日寬表以左連接合併。
    """

    if revenue_df.empty and financial_df.empty:
        return pd.DataFrame(columns=["stock_id", "date"])

    combined = revenue_df.merge(
        financial_df,
        on=["stock_id", "date"],
        how="outer",
        suffixes=("", "_fs"),
    )
    combined = combined.sort_values(["stock_id", "date"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["stock_id", "date"], keep="last")

    daily = align_monthly_to_daily(combined, trading_daily, strategy=strategy)
    return daily


if __name__ == "__main__":  # pragma: no cover - 模組自測
    client = FinMindClient()
    stocks = ["2330", "2317"]
    revenue = fetch_month_revenue(stocks, "2023-01-01", "2024-12-31", client)
    financial = fetch_eps_and_income(stocks, "2022-01-01", "2024-12-31", client)
    sample_daily = pd.DataFrame(
        {
            "stock_id": ["2330"] * 5,
            "date": pd.date_range("2024-09-01", periods=5, freq="B"),
        }
    )
    aligned = prepare_fundamental_daily(revenue, financial, sample_daily)
    print("revenue rows", revenue.shape, "financial rows", financial.shape)
    print(aligned.head())
