"""將 FinMind 基本面與市場熱度欄位整併至每日寬表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import get_logger
from .api import FinMindClient
from .fundamentals import fetch_eps_and_income, fetch_month_revenue, prepare_fundamental_daily

LOGGER = get_logger(__name__)


def _ensure_dataframe(path: str | Path) -> pd.DataFrame:
    """讀取 CSV 並標準化欄位型別。"""

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(4)
    return df


def add_market_heat(df: pd.DataFrame) -> pd.DataFrame:
    """計算市場相對熱度欄位。

    會針對每日全市場計算成交值、成交量之百分位排名，並比較當日值與
    過去 20 日均值的相對變化，衍生 `turnover_rank_pct`、`volume_rank_pct`、
    `volume_ma20`、`volume_ratio`、`turnover_change_vs_ma20`、
    `transactions_change_vs_ma20` 等欄位。
    """

    df = df.copy()
    for column in ["turnover", "volume", "TaiwanStockPrice_transactions"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "date" not in df.columns:
        raise ValueError("資料缺少 date 欄位，無法計算市場熱度")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["turnover_rank_pct"] = (
        df.groupby("date")["turnover"].rank(method="average", pct=True)
    ).clip(0, 1).round(4)
    df["volume_rank_pct"] = (
        df.groupby("date")["volume"].rank(method="average", pct=True)
    ).clip(0, 1).round(4)

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    volume_ma20 = (
        df.groupby("stock_id")["volume"]
        .transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        .replace(0, np.nan)
    )
    turnover_ma20 = (
        df.groupby("stock_id")["turnover"]
        .transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        .replace(0, np.nan)
    )
    transactions_ma20 = (
        df.groupby("stock_id")["TaiwanStockPrice_transactions"]
        .transform(lambda s: s.rolling(window=20, min_periods=5).mean())
        .replace(0, np.nan)
    )

    df["volume_ma20"] = volume_ma20
    df["volume_ratio"] = df["volume"] / volume_ma20
    df["turnover_change_vs_ma20"] = df["turnover"] / turnover_ma20 - 1
    df["transactions_change_vs_ma20"] = df["TaiwanStockPrice_transactions"] / transactions_ma20 - 1

    df[["volume_ratio", "turnover_change_vs_ma20", "transactions_change_vs_ma20"]] = df[
        ["volume_ratio", "turnover_change_vs_ma20", "transactions_change_vs_ma20"]
    ].replace([np.inf, -np.inf], np.nan)

    return df


@dataclass
class EnrichConfig:
    """封裝寬表擴充過程所需的設定。"""

    input_path: Path
    output_path: Path
    fetch_fundamentals: bool = False
    since: str | None = None
    stocks: Sequence[str] | None = None
    token: str | None = None
    force_refresh: bool = False
    strict: bool = False
    align_strategy: str = "forward_fill"
    cache_dir: Path | None = None
    min_output_path: Path | None = None
    update_min: bool = True


def _determine_min_path(out_path: Path) -> Path:
    """依主檔案路徑推導精簡版寬表檔名。"""

    name = out_path.name
    if name.endswith("_clean_daily_wide.csv"):
        return out_path.with_name(name.replace("_clean_daily_wide.csv", "_clean_daily_wide_min.csv"))
    return out_path.with_name(out_path.stem + "_min.csv")


def enrich_clean_daily(config: EnrichConfig) -> pd.DataFrame:
    """讀取寬表、合併基本面並計算市場熱度後輸出。

    主要步驟如下：

    1. 讀取每日寬表並標準化欄位。
    2. 視需要透過 FinMind API 取得月營收與財報資料，並對齊至每日。
    3. 計算市場熱度（成交值/量分位數與量能比值）。
    4. 寫回主檔案與精簡版檔案。
    """

    df = _ensure_dataframe(config.input_path)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    target_stocks = [s.strip().zfill(4) for s in config.stocks] if config.stocks else sorted(df["stock_id"].unique())
    trading_range = df[["stock_id", "date"]]
    min_date = df["date"].min()
    max_date = df["date"].max()

    warnings: list[str] = []
    added_columns: list[str] = []

    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError("輸入資料缺少有效日期欄位")

    if config.stocks:
        missing = sorted(set(target_stocks) - set(df["stock_id"]))
        if missing:
            LOGGER.warning("指定股票在資料中找不到：%s", missing)

    since_date = pd.to_datetime(config.since, errors="coerce") if config.since else None
    if since_date is not None and pd.isna(since_date):
        LOGGER.warning("--since 參數無法解析：%s", config.since)
        since_date = None
    if since_date is not None and since_date > min_date:
        trading_range = trading_range[trading_range["date"] >= since_date]
    trading_range = trading_range[trading_range["stock_id"].isin(target_stocks)]
    if trading_range.empty:
        warnings.append("指定股票或日期條件後無交易資料，基本面欄位可能為 NaN")

    min_output_path = None
    if config.update_min:
        min_output_path = config.min_output_path or _determine_min_path(config.output_path)

    if config.fetch_fundamentals:
        LOGGER.info("開始抓取基本面資料，股票數量=%s", len(target_stocks))
        client = FinMindClient(
            token=config.token,
            force_refresh=config.force_refresh,
            cache_dir=config.cache_dir,
        )
        start_fetch = (since_date if since_date is not None else min_date) - pd.DateOffset(months=13)
        start_fetch = start_fetch.strftime("%Y-%m-%d")
        end_fetch = max_date.strftime("%Y-%m-%d")

        revenue_df = fetch_month_revenue(target_stocks, start_fetch, end_fetch, client)
        income_df = fetch_eps_and_income(target_stocks, start_fetch, end_fetch, client)
        if revenue_df.empty:
            warnings.append("月營收資料為空，revenue 欄位將為 NaN")
        if income_df.empty:
            warnings.append("財報資料為空，EPS 欄位將為 NaN")
        daily_fund = prepare_fundamental_daily(
            revenue_df,
            income_df,
            trading_range,
            strategy=config.align_strategy,
        )
        fundamental_cols = [col for col in daily_fund.columns if col not in {"stock_id", "date"}]
        if fundamental_cols:
            LOGGER.info("基本面欄位：%s", fundamental_cols)
            df = df.merge(daily_fund, on=["stock_id", "date"], how="left")
            added_columns.extend(fundamental_cols)
            missing_stock_fund = sorted(set(target_stocks) - set(daily_fund["stock_id"].unique()))
            if missing_stock_fund:
                warning = f"部分股票缺少基本面資料：{missing_stock_fund}"
                if config.strict:
                    raise RuntimeError(warning)
                warnings.append(warning)
        else:
            message = "未取得基本面資料"
            if config.strict:
                raise RuntimeError(message)
            warnings.append(message)
    else:
        LOGGER.info("未啟用基本面抓取，僅計算市場熱度欄位")

    df = add_market_heat(df)
    if "country" in df.columns:
        df.drop(columns=["country"], inplace=True)
    heat_columns = [
        "turnover_rank_pct",
        "volume_rank_pct",
        "volume_ma20",
        "volume_ratio",
        "turnover_change_vs_ma20",
        "transactions_change_vs_ma20",
    ]
    added_columns.extend([col for col in heat_columns if col in df.columns])

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df.to_csv(config.output_path, index=False)
    LOGGER.info("已輸出寬表：%s", config.output_path)

    available_heat = [col for col in heat_columns if col in df.columns]
    added_columns_unique = list(dict.fromkeys(added_columns))
    extra_columns = [
        col for col in added_columns_unique if col not in available_heat and col not in {"stock_id", "date"}
    ]
    extra_columns = list(dict.fromkeys(extra_columns))

    if min_output_path and Path(min_output_path).exists():
        LOGGER.info("更新精簡寬表：%s", min_output_path)
        min_df = pd.read_csv(min_output_path)
        min_df["date"] = pd.to_datetime(min_df["date"], errors="coerce")
        min_df["stock_id"] = min_df["stock_id"].astype(str).str.zfill(4)
        merge_columns = ["stock_id", "date"] + available_heat + extra_columns
        merge_columns = list(dict.fromkeys([col for col in merge_columns if col in df.columns]))
        min_df = min_df.merge(df[merge_columns], on=["stock_id", "date"], how="left")
        min_df.to_csv(min_output_path, index=False)
    elif min_output_path:
        LOGGER.info("建立精簡寬表：%s", min_output_path)
        base_columns = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"]
        base_columns = [col for col in base_columns if col in df.columns]
        additional = [col for col in available_heat + extra_columns if col not in base_columns]
        min_df = df[base_columns + additional].copy()
        min_df.to_csv(min_output_path, index=False)

    LOGGER.info("新增欄位：%s", added_columns_unique)
    if warnings:
        for message in warnings:
            LOGGER.warning(message)

    return df


if __name__ == "__main__":  # pragma: no cover - 簡易自測
    demo_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "stock_id": ["2330", "2317", "2330", "2317"],
            "turnover": [1000, 2000, 1500, 1200],
            "volume": [10_000, 5_000, 12_000, 8_000],
            "TaiwanStockPrice_transactions": [100, 80, 120, 90],
        }
    )
    print(add_market_heat(demo_df))
