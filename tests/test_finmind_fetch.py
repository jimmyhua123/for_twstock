from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from finmind_fetch.api import FinMindAPIError, FinMindClient
from finmind_fetch.fundamentals import align_monthly_to_daily, prepare_fundamental_daily
from finmind_fetch.enrich import add_market_heat


def test_add_market_heat_generates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "stock_id": ["2330", "2317", "2330", "2317"],
            "turnover": [1000, 2000, 1500, 1000],
            "volume": [10, 20, 15, 5],
            "TaiwanStockPrice_transactions": [100, 50, 120, 40],
        }
    )
    enriched = add_market_heat(df)

    for column in [
        "turnover_rank_pct",
        "volume_rank_pct",
        "volume_ma20",
        "volume_ratio",
        "turnover_change_5d",
        "transactions_change_5d",
    ]:
        assert column in enriched.columns

    assert (enriched["turnover_rank_pct"].between(0, 1, inclusive="both")).all()
    assert not enriched["volume_ratio"].isna().all()


def test_align_monthly_to_daily_forward_fill() -> None:
    monthly = pd.DataFrame(
        {
            "stock_id": ["2330", "2330"],
            "date": pd.to_datetime(["2024-01-10", "2024-02-10"]),
            "revenue": [100, 150],
            "revenue_yoy": [0.1, 0.2],
        }
    )
    daily = pd.DataFrame(
        {
            "stock_id": ["2330"] * 30,
            "date": pd.date_range("2024-01-15", periods=30, freq="B"),
        }
    )
    aligned = align_monthly_to_daily(monthly, daily)
    assert len(aligned) == len(daily)
    assert aligned.loc[aligned["date"] == pd.Timestamp("2024-01-15"), "revenue"].iloc[0] == 100
    assert aligned.loc[aligned["date"] == pd.Timestamp("2024-02-21"), "revenue"].iloc[0] == 150


def test_prepare_fundamental_daily_merges_sources() -> None:
    revenue_df = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "date": pd.to_datetime(["2024-01-01"]),
            "revenue": [1000],
            "revenue_yoy": [0.1],
            "revenue_mom": [0.05],
        }
    )
    financial_df = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "date": pd.to_datetime(["2024-01-01"]),
            "eps": [2.5],
            "eps_ttm": [10.0],
        }
    )
    trading = pd.DataFrame(
        {
            "stock_id": ["2330"],
            "date": pd.to_datetime(["2024-01-03"]),
        }
    )
    daily = prepare_fundamental_daily(revenue_df, financial_df, trading)
    assert set(["revenue", "revenue_yoy", "revenue_mom", "eps", "eps_ttm"]).issubset(daily.columns)
    assert daily.loc[0, "eps_ttm"] == 10.0


def test_get_dataset_raises_on_non_dict_payload(tmp_path: Path) -> None:
    client = FinMindClient(cache_dir=tmp_path, force_refresh=True, max_retries=1)

    class DummyResponse:
        status_code = 200
        text = "non-dict"

        @staticmethod
        def json() -> str:
            return "unexpected"

    client._session.get = lambda *args, **kwargs: DummyResponse()  # type: ignore[assignment]

    with pytest.raises(FinMindAPIError):
        client.get_dataset("test_dataset", {"foo": "bar"})
