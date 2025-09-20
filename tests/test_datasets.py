from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from finmind_etl.datasets import clean_dataset, get_spec


def test_clean_dataset_institutional_renames_columns() -> None:
    spec = get_spec("TaiwanStockInstitutionalInvestorsBuySell")
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "stock_id": ["2330"],
            "Foreign_Investor_Net_Buy_Sell": ["100"],
            "Investment_Trust_Net_Buy_Sell": ["-20"],
            "Dealer_Self_Net_Buy_Sell": ["5"],
            "Dealer_Hedging_Net_Buy_Sell": ["2"],
        }
    )

    cleaned = clean_dataset(spec, raw)

    expected = {
        "foreign_net",
        "invest_trust_net",
        "dealer_net",
        "dealer_self_net",
        "dealer_hedging_net",
    }
    assert expected.issubset(cleaned.columns)
    assert cleaned.loc[0, "foreign_net"] == 100
    assert cleaned.loc[0, "invest_trust_net"] == -20


def test_clean_dataset_month_revenue_derives_growth() -> None:
    spec = get_spec("TaiwanStockMonthRevenue")
    raw = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-02-10", "2024-03-10"],
            "stock_id": ["2330", "2330", "2330"],
            "revenue": [100.0, 120.0, 90.0],
        }
    )

    cleaned = clean_dataset(spec, raw)

    jan = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-10"), "revenue_mom"].iloc[0]
    feb = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-02-10"), "revenue_mom"].iloc[0]

    assert pd.isna(jan)
    assert pytest.approx(feb, rel=1e-6) == 0.2

