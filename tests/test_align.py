from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from finmind_etl.align import align_dataset, build_trading_base
from finmind_etl.datasets import DatasetResult, clean_dataset, get_spec


def _make_result(name: str, raw: pd.DataFrame) -> DatasetResult:
    spec = get_spec(name)
    cleaned = clean_dataset(spec, raw)
    return DatasetResult(spec=spec, raw=raw, clean=cleaned)


def test_build_trading_base_uses_calendar() -> None:
    calendar_raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "is_trading_day": [1, 0, 1],
        }
    )
    base = build_trading_base(["2330", "2317"], _make_result("TaiwanStockTradingDate", calendar_raw).clean, pd.DataFrame())
    assert len(base) == 4  # 2 stocks * 2 trading days


def test_align_dataset_market_level_forward_fill() -> None:
    base = pd.DataFrame(
        {
            "stock_id": ["2330", "2330", "2317", "2317"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
        }
    )
    market_raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "market_margin_maintenance": [130.0],
        }
    )
    result = _make_result("TaiwanTotalExchangeMarginMaintenance", market_raw)
    aligned = align_dataset(base, result)
    assert (aligned["market_margin_maintenance"].notna()).all()

