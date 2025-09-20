"""Dataset metadata and cleaning utilities for FinMind ETL pipeline.

This module centralises all dataset specific knowledge required by the
``finmind_etl`` package.  It exposes:

* :class:`DatasetSpec` which describes how a dataset should be fetched and
  incorporated into the merged daily view.
* :class:`DatasetResult` – a light wrapper holding the raw response and the
  cleaned frame that downstream steps operate on.
* ``iter_category_specs`` which returns the dataset configurations required for
  each category (technical / chip / fundamental).
* ``clean_dataset`` which applies FinMind's translation metadata, converts
  column names to ``snake_case`` and performs dataset specific normalisation.

The implementation intentionally favours readability over cleverness.  The
lists below may look long, yet they encode the project requirements in a single
location which makes it significantly easier to audit the coverage of datasets
and adjust the cleaning logic when FinMind makes changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

Category = str


@dataclass(frozen=True)
class DatasetSpec:
    """Describe how a single FinMind dataset should be handled."""

    name: str
    category: Category
    requires_stock: bool = True
    include_in_wide: bool = True
    forward_fill: bool = False
    level: str = "stock"  # ``stock`` joins on stock_id+date, ``market`` joins on date only
    cleaner: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    numeric_fields: Sequence[str] = field(default_factory=tuple)
    required_fields: Sequence[str] = field(default_factory=tuple)
    description: str = ""


@dataclass
class DatasetResult:
    """Container storing the raw and cleaned DataFrames of a dataset."""

    spec: DatasetSpec
    raw: pd.DataFrame
    clean: pd.DataFrame


def _to_snake_case(value: str) -> str:
    import re

    text = value.strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def _apply_translation(df: pd.DataFrame, translation: Optional[Mapping[str, str]]) -> pd.DataFrame:
    if translation is None:
        translation = {}
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        translated = translation.get(column)
        if translated is None:
            translated = translation.get(column.lower())
        if translated is not None:
            rename_map[column] = _to_snake_case(str(translated))
        else:
            rename_map[column] = _to_snake_case(column)
    return df.rename(columns=rename_map)


def _ensure_datetime(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _ensure_stock_id(df: pd.DataFrame, column: str = "stock_id") -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _rename_by_patterns(df: pd.DataFrame, patterns: Mapping[str, str]) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        lower = column.lower()
        for pattern, target in patterns.items():
            if pattern in lower:
                rename_map[column] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# -- Cleaning functions ------------------------------------------------------


def _clean_default(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _clean_stock_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_stock_id(df)
    keep = [column for column in df.columns if column in {"stock_id", "stock_name", "industry_category"}]
    if not keep:
        keep = [column for column in df.columns if column.startswith("stock")]
    if keep:
        df = df[keep + [c for c in df.columns if c not in keep]]
    return df.drop_duplicates(subset=["stock_id"])


def _clean_trading_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)
    if "is_trading_day" in df.columns:
        df["is_trading_day"] = df["is_trading_day"].astype(int)
    elif "trading" in df.columns:
        df = df.rename(columns={"trading": "is_trading_day"})
        df["is_trading_day"] = df["is_trading_day"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def _clean_stock_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    patterns = {
        "open": "open",
        "max": "high",
        "high": "high",
        "min": "low",
        "low": "low",
        "close": "close",
        "trading_volume": "volume",
        "trade_volume": "volume",
        "volume": "volume",
        "trading_value": "turnover",
        "trading_money": "turnover",
        "trading_turnover": "transactions",
        "transactions": "transactions",
        "spread": "price_change",
    }
    df = _rename_by_patterns(df, patterns)
    df = _ensure_columns(df, ["open", "high", "low", "close", "volume", "turnover"])
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["open", "high", "low", "close", "volume", "turnover", "transactions", "price_change"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df


def _clean_stock_price_adj(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_stock_price(df)
    rename = {col: f"adj_{col}" for col in ["open", "high", "low", "close"] if col in df.columns}
    df = df.rename(columns=rename)
    patterns = {
        "adjclose": "adj_close",
        "adj_open": "adj_open",
        "adj_high": "adj_high",
        "adj_low": "adj_low",
    }
    df = _rename_by_patterns(df, patterns)
    _ensure_numeric(df, ["adj_open", "adj_high", "adj_low", "adj_close"])
    return df


def _clean_stock_per(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"per": "per", "pbr": "pbr", "dividend": "dividend_yield"})
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["per", "pbr", "dividend_yield"])
    return df


def _clean_stock_day_trading(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    patterns = {
        "day_trading_volume": "day_trading_volume",
        "day_trading_value": "day_trading_value",
        "day_trade_value": "day_trading_value",
        "day_trade_buy_value": "day_trading_buy_value",
        "day_trade_sell_value": "day_trading_sell_value",
        "day_trade_turnover": "day_trading_turnover",
    }
    df = _rename_by_patterns(df, patterns)
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(
        df,
        [
            "day_trading_volume",
            "day_trading_value",
            "day_trading_buy_value",
            "day_trading_sell_value",
            "day_trading_turnover",
        ],
    )
    return df


def _clean_stock_10year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    patterns = {
        "avg": "avg_price_10y",
        "value": "avg_price_10y",
        "avg_price": "avg_price_10y",
    }
    df = _rename_by_patterns(df, patterns)
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["avg_price_10y"])
    return df


def _clean_total_return_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"return_index": "total_return_index"})
    df = _ensure_datetime(df)
    _ensure_numeric(df, ["total_return_index"])
    return df


def _clean_institutional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "foreign": "foreign_net",
            "investment_trust": "invest_trust_net",
            "dealer_self": "dealer_self_net",
            "dealer_hedging": "dealer_hedging_net",
            "dealer": "dealer_net",
        },
    )
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    df = _ensure_columns(
        df,
        ["foreign_net", "invest_trust_net", "dealer_net", "dealer_self_net", "dealer_hedging_net"],
    )
    _ensure_numeric(
        df,
        [
            "foreign_net",
            "invest_trust_net",
            "dealer_net",
            "dealer_self_net",
            "dealer_hedging_net",
        ],
    )
    return df


def _clean_margin_short(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "margin_purchase_today_balance": "margin_long",
            "margin_purchase_balance": "margin_long",
            "short_sale_today_balance": "margin_short",
            "short_sale_balance": "margin_short",
            "margin_purchase_change": "margin_long_change",
            "short_sale_change": "margin_short_change",
            "short_sale_volume": "short_selling",
        },
    )
    df = _ensure_columns(
        df,
        ["margin_long", "margin_short", "margin_long_change", "margin_short_change", "short_selling"],
    )
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(
        df,
        ["margin_long", "margin_short", "margin_long_change", "margin_short_change", "short_selling"],
    )
    return df


def _clean_shareholding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "foreign_investor_shares": "foreign_holding_shares",
            "foreign_investor_ratio": "foreign_holding_ratio",
            "ratio": "foreign_holding_ratio",
        },
    )
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["foreign_holding_shares", "foreign_holding_ratio"])
    return df


def _clean_securities_lending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "lending_volume": "securities_lending_volume",
            "lending_balance": "securities_lending_balance",
            "lending_fees": "securities_lending_fee",
        },
    )
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(
        df,
        ["securities_lending_volume", "securities_lending_balance", "securities_lending_fee"],
    )
    return df


def _clean_daily_short_sale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"balance": "short_sale_balance", "limit": "short_sale_limit"})
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["short_sale_balance", "short_sale_limit"])
    return df


def _clean_government_bank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"net_buy_sell": "gov_bank_net", "buy": "gov_bank_buy", "sell": "gov_bank_sell"})
    df = _ensure_datetime(df)
    _ensure_numeric(df, ["gov_bank_net", "gov_bank_buy", "gov_bank_sell"])
    return df


def _clean_total_institutional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "foreign": "market_foreign_net",
            "investment_trust": "market_invest_trust_net",
            "dealer": "market_dealer_net",
            "dealer_self": "market_dealer_self_net",
            "dealer_hedging": "market_dealer_hedging_net",
        },
    )
    df = _ensure_datetime(df)
    _ensure_numeric(
        df,
        [
            "market_foreign_net",
            "market_invest_trust_net",
            "market_dealer_net",
            "market_dealer_self_net",
            "market_dealer_hedging_net",
        ],
    )
    return df


def _clean_total_margin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "marginpurchasebalance": "market_margin_long",
            "shortsalebalance": "market_margin_short",
            "margin_balance": "market_margin_long",
            "short_balance": "market_margin_short",
        },
    )
    df = _ensure_datetime(df)
    _ensure_numeric(df, ["market_margin_long", "market_margin_short"])
    return df


def _clean_exchange_margin_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"maintenance": "market_margin_maintenance"})
    df = _ensure_datetime(df)
    _ensure_numeric(df, ["market_margin_maintenance"])
    return df


def _clean_month_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    df = _rename_by_patterns(
        df,
        {
            "revenue": "revenue",
            "revenue_month": "revenue_month",
            "revenue_year": "revenue_year",
            "revenue_last_month": "revenue_prev_month",
            "revenue_last_year": "revenue_prev_year",
            "accumulated_revenue_last_year": "accumulated_revenue_prev_year",
        },
    )
    if "revenue_month" not in df.columns and "date" in df.columns:
        df["revenue_month"] = df["date"].dt.month
    if "revenue_year" not in df.columns and "date" in df.columns:
        df["revenue_year"] = df["date"].dt.year
    _ensure_numeric(
        df,
        [
            "revenue",
            "revenue_prev_month",
            "revenue_prev_year",
            "accumulated_revenue",
            "accumulated_revenue_prev_year",
        ],
    )
    if "revenue" in df.columns:
        df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
        df["revenue_yoy"] = df.groupby("stock_id")["revenue"].pct_change(periods=12)
        df["revenue_mom"] = df.groupby("stock_id")["revenue"].pct_change()
    if "country" not in df.columns:
        df["country"] = "TW"
    return df


def _clean_financial_statements(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()
    else:
        candidates = [col for col in df.columns if "type" in col or "item" in col]
        if candidates:
            df["type"] = df[candidates[0]].astype(str).str.lower()
    value_col = "value"
    if value_col not in df.columns:
        candidates = [col for col in df.columns if col.endswith("value") or col.endswith("amount")]
        if candidates:
            value_col = candidates[0]
            df["value"] = pd.to_numeric(df[candidates[0]], errors="coerce")
    df = df[df.get("type", "").str.contains("eps", na=False)]
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    if "value" in df.columns:
        df["eps"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        df["eps"] = np.nan
    df = df[[col for col in df.columns if col in {"date", "stock_id", "eps"}] or []]
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    if not df.empty:
        df["eps_ttm"] = df.groupby("stock_id")["eps"].transform(lambda s: s.rolling(window=4, min_periods=1).sum())
    return df


def _clean_cash_flows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    return df


def _clean_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    return df


def _clean_dividend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(
        df,
        {
            "cash_dividend": "cash_dividend",
            "stock_dividend": "stock_dividend",
            "cash": "cash_dividend",
            "stock": "stock_dividend",
        },
    )
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["cash_dividend", "stock_dividend"])
    return df


def _clean_dividend_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"cash": "cash_dividend_actual", "stock": "stock_dividend_actual"})
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["cash_dividend_actual", "stock_dividend_actual"])
    return df


def _clean_market_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"market_value": "market_value", "market_cap": "market_value"})
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["market_value"])
    return df


def _clean_market_value_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _rename_by_patterns(df, {"weight": "market_value_weight"})
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    _ensure_numeric(df, ["market_value_weight"])
    return df


def _clean_capital_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)
    df = _ensure_stock_id(df)
    return df


# -- Dataset catalogue -------------------------------------------------------


TECHNICAL_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        name="TaiwanStockInfo",
        category="technical",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=False,
        level="stock",
        cleaner=_clean_stock_info,
        required_fields=("stock_id", "stock_name", "industry_category"),
        description="Stock master information",
    ),
    DatasetSpec(
        name="TaiwanStockTradingDate",
        category="technical",
        requires_stock=False,
        include_in_wide=False,
        forward_fill=False,
        level="market",
        cleaner=_clean_trading_calendar,
        required_fields=("date", "is_trading_day"),
        description="Trading calendar",
    ),
    DatasetSpec(
        name="TaiwanStockPrice",
        category="technical",
        cleaner=_clean_stock_price,
        numeric_fields=("open", "high", "low", "close", "volume", "turnover"),
        required_fields=("date", "stock_id", "open", "high", "low", "close", "volume", "turnover"),
        description="Daily OHLC price",
    ),
    DatasetSpec(
        name="TaiwanStockPriceAdj",
        category="technical",
        cleaner=_clean_stock_price_adj,
        numeric_fields=(
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
        ),
        required_fields=("date", "stock_id"),
        description="Daily OHLC (adjusted)",
    ),
    DatasetSpec(
        name="TaiwanStockPriceTick",
        category="technical",
        include_in_wide=False,
        forward_fill=False,
        cleaner=_clean_default,
        description="Historical tick data",
    ),
    DatasetSpec(
        name="TaiwanStockPER",
        category="technical",
        cleaner=_clean_stock_per,
        numeric_fields=("per", "pbr", "dividend_yield"),
        required_fields=("date", "stock_id"),
        description="PER/PBR information",
    ),
    DatasetSpec(
        name="TaiwanStockStatisticsOfOrderBookAndTrade",
        category="technical",
        include_in_wide=False,
        cleaner=_clean_default,
        description="5-second order book statistics",
    ),
    DatasetSpec(
        name="TaiwanVariousIndicators5Seconds",
        category="technical",
        requires_stock=False,
        include_in_wide=False,
        level="market",
        cleaner=_clean_default,
        description="Index indicators every 5 seconds",
    ),
    DatasetSpec(
        name="TaiwanStockDayTrading",
        category="technical",
        cleaner=_clean_stock_day_trading,
        numeric_fields=(
            "day_trading_volume",
            "day_trading_value",
            "day_trading_buy_value",
            "day_trading_sell_value",
            "day_trading_turnover",
        ),
        required_fields=("date", "stock_id"),
        description="Day trading statistics",
    ),
    DatasetSpec(
        name="TaiwanStockTotalReturnIndex",
        category="technical",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=True,
        level="market",
        cleaner=_clean_total_return_index,
        numeric_fields=("total_return_index",),
        required_fields=("date", "total_return_index"),
        description="Total return index",
    ),
    DatasetSpec(
        name="TaiwanStock10Year",
        category="technical",
        cleaner=_clean_stock_10year,
        numeric_fields=("avg_price_10y",),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Ten year moving averages",
    ),
    DatasetSpec(
        name="TaiwanStockKBar",
        category="technical",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Intraday K bar data",
    ),
    DatasetSpec(
        name="TaiwanStockWeekPrice",
        category="technical",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Weekly price data",
    ),
    DatasetSpec(
        name="TaiwanStockMonthPrice",
        category="technical",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Monthly price data",
    ),
    DatasetSpec(
        name="TaiwanStockEvery5SecondsIndex",
        category="technical",
        requires_stock=False,
        include_in_wide=False,
        level="market",
        cleaner=_clean_default,
        description="Index every 5 seconds",
    ),
]


CHIP_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        name="TaiwanStockMarginPurchaseShortSale",
        category="chip",
        cleaner=_clean_margin_short,
        numeric_fields=(
            "margin_long",
            "margin_short",
            "margin_long_change",
            "margin_short_change",
            "short_selling",
        ),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Margin purchase / short sale",
    ),
    DatasetSpec(
        name="TaiwanStockTotalMarginPurchaseShortSale",
        category="chip",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=True,
        level="market",
        cleaner=_clean_total_margin,
        numeric_fields=("market_margin_long", "market_margin_short"),
        required_fields=("date",),
        description="Market wide margin / short balances",
    ),
    DatasetSpec(
        name="TaiwanStockInstitutionalInvestorsBuySell",
        category="chip",
        cleaner=_clean_institutional,
        numeric_fields=(
            "foreign_net",
            "invest_trust_net",
            "dealer_net",
            "dealer_self_net",
            "dealer_hedging_net",
        ),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Institutional investors buy/sell",
    ),
    DatasetSpec(
        name="TaiwanStockTotalInstitutionalInvestors",
        category="chip",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=True,
        level="market",
        cleaner=_clean_total_institutional,
        numeric_fields=(
            "market_foreign_net",
            "market_invest_trust_net",
            "market_dealer_net",
            "market_dealer_self_net",
            "market_dealer_hedging_net",
        ),
        required_fields=("date",),
        description="Market wide institutional flow",
    ),
    DatasetSpec(
        name="TaiwanStockShareholding",
        category="chip",
        cleaner=_clean_shareholding,
        numeric_fields=("foreign_holding_shares", "foreign_holding_ratio"),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Foreign shareholding",
    ),
    DatasetSpec(
        name="TaiwanStockHoldingSharesPer",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Shareholding distribution",
    ),
    DatasetSpec(
        name="TaiwanStockSecuritiesLending",
        category="chip",
        cleaner=_clean_securities_lending,
        numeric_fields=(
            "securities_lending_volume",
            "securities_lending_balance",
            "securities_lending_fee",
        ),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Securities lending",
    ),
    DatasetSpec(
        name="TaiwanStockMarginShortSaleSuspension",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Short sale suspension list",
    ),
    DatasetSpec(
        name="TaiwanDailyShortSaleBalances",
        category="chip",
        cleaner=_clean_daily_short_sale,
        numeric_fields=("short_sale_balance", "short_sale_limit"),
        forward_fill=True,
        required_fields=("date", "stock_id"),
        description="Daily short sale balances",
    ),
    DatasetSpec(
        name="TaiwanSecuritiesTraderInfo",
        category="chip",
        requires_stock=False,
        include_in_wide=False,
        cleaner=_clean_default,
        description="Broker information",
    ),
    DatasetSpec(
        name="TaiwanStockTradingDailyReport",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Broker daily report",
    ),
    DatasetSpec(
        name="TaiwanStockWarrantTradingDailyReport",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Warrant broker daily report",
    ),
    DatasetSpec(
        name="TaiwanStockGovernmentBankBuySell",
        category="chip",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=True,
        level="market",
        cleaner=_clean_government_bank,
        numeric_fields=("gov_bank_net", "gov_bank_buy", "gov_bank_sell"),
        required_fields=("date",),
        description="Government bank net buy/sell",
    ),
    DatasetSpec(
        name="TaiwanTotalExchangeMarginMaintenance",
        category="chip",
        requires_stock=False,
        include_in_wide=True,
        forward_fill=True,
        level="market",
        cleaner=_clean_exchange_margin_maintenance,
        numeric_fields=("market_margin_maintenance",),
        required_fields=("date",),
        description="Exchange margin maintenance",
    ),
    DatasetSpec(
        name="TaiwanStockTradingDailyReportSecIdAgg",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Broker aggregated report",
    ),
    DatasetSpec(
        name="TaiwanStockDispositionSecuritiesPeriod",
        category="chip",
        include_in_wide=False,
        cleaner=_clean_default,
        description="Disposition securities period",
    ),
]


FUNDAMENTAL_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        name="TaiwanStockCashFlowsStatement",
        category="fundamental",
        cleaner=_clean_cash_flows,
        include_in_wide=False,
        description="Cash flow statements",
    ),
    DatasetSpec(
        name="TaiwanStockFinancialStatements",
        category="fundamental",
        cleaner=_clean_financial_statements,
        forward_fill=True,
        numeric_fields=("eps", "eps_ttm"),
        required_fields=("date", "stock_id", "eps", "eps_ttm"),
        description="Financial statements (EPS extraction)",
    ),
    DatasetSpec(
        name="TaiwanStockBalanceSheet",
        category="fundamental",
        cleaner=_clean_balance_sheet,
        include_in_wide=False,
        description="Balance sheet",
    ),
    DatasetSpec(
        name="TaiwanStockDividend",
        category="fundamental",
        cleaner=_clean_dividend,
        forward_fill=True,
        numeric_fields=("cash_dividend", "stock_dividend"),
        required_fields=("date", "stock_id"),
        description="Dividend policy",
    ),
    DatasetSpec(
        name="TaiwanStockDividendResult",
        category="fundamental",
        cleaner=_clean_dividend_result,
        forward_fill=True,
        numeric_fields=("cash_dividend_actual", "stock_dividend_actual"),
        required_fields=("date", "stock_id"),
        description="Ex-dividend results",
    ),
    DatasetSpec(
        name="TaiwanStockMonthRevenue",
        category="fundamental",
        cleaner=_clean_month_revenue,
        forward_fill=True,
        numeric_fields=(
            "revenue",
            "revenue_prev_month",
            "revenue_prev_year",
            "accumulated_revenue",
            "accumulated_revenue_prev_year",
            "revenue_yoy",
            "revenue_mom",
        ),
        required_fields=("date", "stock_id", "revenue", "revenue_yoy", "revenue_mom"),
        description="Monthly revenue",
    ),
    DatasetSpec(
        name="TaiwanStockCapitalReductionReferencePrice",
        category="fundamental",
        cleaner=_clean_capital_change,
        forward_fill=False,
        include_in_wide=False,
        description="Capital reduction reference price",
    ),
    DatasetSpec(
        name="TaiwanStockMarketValue",
        category="fundamental",
        cleaner=_clean_market_value,
        forward_fill=True,
        numeric_fields=("market_value",),
        required_fields=("date", "stock_id", "market_value"),
        description="Market value",
    ),
    DatasetSpec(
        name="TaiwanStockDelisting",
        category="fundamental",
        cleaner=_clean_capital_change,
        include_in_wide=False,
        description="Delisting records",
    ),
    DatasetSpec(
        name="TaiwanStockMarketValueWeight",
        category="fundamental",
        cleaner=_clean_market_value_weight,
        forward_fill=True,
        numeric_fields=("market_value_weight",),
        required_fields=("date", "stock_id"),
        description="Market value weight",
    ),
    DatasetSpec(
        name="TaiwanStockSplitPrice",
        category="fundamental",
        cleaner=_clean_capital_change,
        include_in_wide=False,
        description="Split price reference",
    ),
    DatasetSpec(
        name="TaiwanStockParValueChange",
        category="fundamental",
        cleaner=_clean_capital_change,
        include_in_wide=False,
        description="Par value change",
    ),
]


ALL_SPECS: Dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (*TECHNICAL_SPECS, *CHIP_SPECS, *FUNDAMENTAL_SPECS)
}


def iter_category_specs(category: Category) -> Iterator[DatasetSpec]:
    for spec in ALL_SPECS.values():
        if spec.category == category:
            yield spec


def get_spec(name: str) -> DatasetSpec:
    try:
        return ALL_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name}") from exc


def clean_dataset(spec: DatasetSpec, df: pd.DataFrame, translation: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Apply translation and dataset specific cleaning rules."""

    if df is None:
        return pd.DataFrame()

    sanitized = _apply_translation(df, translation)
    cleaner = spec.cleaner or _clean_default
    cleaned = cleaner(sanitized)

    # Finalise common fields
    cleaned = cleaned.copy()
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    if spec.requires_stock or "stock_id" in cleaned.columns:
        cleaned = _ensure_stock_id(cleaned)

    cleaned = _ensure_columns(cleaned, spec.required_fields)
    _ensure_numeric(cleaned, spec.numeric_fields)

    sort_keys: List[str] = []
    if "stock_id" in cleaned.columns:
        sort_keys.append("stock_id")
    if "date" in cleaned.columns:
        sort_keys.append("date")
    if sort_keys:
        cleaned = cleaned.sort_values(sort_keys).reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)
    return cleaned


__all__ = [
    "DatasetSpec",
    "DatasetResult",
    "ALL_SPECS",
    "TECHNICAL_SPECS",
    "CHIP_SPECS",
    "FUNDAMENTAL_SPECS",
    "iter_category_specs",
    "get_spec",
    "clean_dataset",
]

