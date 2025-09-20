from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from finmind_etl.datasets import DatasetResult, clean_dataset, get_spec
from finmind_etl.merge import build_minimal_view, merge_all


def _make_result(name: str, raw: pd.DataFrame) -> DatasetResult:
    spec = get_spec(name)
    cleaned = clean_dataset(spec, raw)
    return DatasetResult(spec=spec, raw=raw, clean=cleaned)


def test_merge_all_combines_sources_and_forward_fills() -> None:
    price_raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "stock_id": ["2330", "2330"],
            "open": [600, 610],
            "max": [605, 615],
            "min": [595, 605],
            "close": [604, 612],
            "trading_volume": [1000, 1200],
            "trading_value": [600_000, 700_000],
        }
    )
    calendar_raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "is_trading_day": [1, 1],
        }
    )
    info_raw = pd.DataFrame({"stock_id": ["2330"], "stock_name": ["TSMC"], "industry_category": ["Semiconductor"]})
    inst_raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "stock_id": ["2330"],
            "Foreign_Investor_Net_Buy_Sell": [100],
        }
    )
    margin_raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "stock_id": ["2330"],
            "MarginPurchaseTodayBalance": [2000],
            "ShortSaleTodayBalance": [500],
        }
    )
    revenue_raw = pd.DataFrame(
        {
            "date": ["2023-12-10", "2024-01-10"],
            "stock_id": ["2330", "2330"],
            "revenue": [100.0, 120.0],
        }
    )

    technical = {
        "TaiwanStockPrice": _make_result("TaiwanStockPrice", price_raw),
        "TaiwanStockTradingDate": _make_result("TaiwanStockTradingDate", calendar_raw),
        "TaiwanStockInfo": _make_result("TaiwanStockInfo", info_raw),
    }
    fundamentals = {"TaiwanStockMonthRevenue": _make_result("TaiwanStockMonthRevenue", revenue_raw)}
    chip = {
        "TaiwanStockInstitutionalInvestorsBuySell": _make_result(
            "TaiwanStockInstitutionalInvestorsBuySell", inst_raw
        ),
        "TaiwanStockMarginPurchaseShortSale": _make_result(
            "TaiwanStockMarginPurchaseShortSale", margin_raw
        ),
    }

    merged = merge_all(["2330"], technical, fundamentals, chip)

    assert not merged.empty
    assert "foreign_net" in merged.columns
    assert merged.loc[merged["date"] == pd.Timestamp("2024-01-02"), "foreign_net"].iloc[0] == 100
    # Forward fill ensures the second day keeps the previous institutional value
    assert merged.loc[merged["date"] == pd.Timestamp("2024-01-03"), "foreign_net"].iloc[0] == 100
    assert merged.loc[0, "stock_name"] == "TSMC"


def test_build_minimal_view_contains_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "stock_id": ["2330"],
            "open": [600],
            "high": [605],
            "low": [595],
            "close": [604],
            "volume": [1000],
            "turnover": [600_000],
        }
    )

    minimal = build_minimal_view(df)
    required = {
        "date",
        "stock_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    }
    assert required.issubset(minimal.columns)
    assert len(minimal) == 1

