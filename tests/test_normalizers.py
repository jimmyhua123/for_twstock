"""normalizers 模組測試。"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from finmind_etl.normalizers import normalize_taiwan_stock_price


def test_normalize_taiwan_stock_price_columns_and_types() -> None:
    """確認價量清洗後欄位齊全且數值欄位為數值型態。"""

    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "stock_id": ["2330", "2330"],
            "open": ["600", "605"],
            "max": ["610", "615"],
            "min": ["590", "600"],
            "close": ["605", "610"],
            "Trading_Volume": ["1,000", "1,200"],
            "Trading_money": ["600,000", "700,000"],
        }
    )

    cleaned = normalize_taiwan_stock_price(raw)

    expected_columns = {
        "date",
        "stock_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    }
    assert expected_columns.issubset(cleaned.columns)
    assert pd.api.types.is_numeric_dtype(cleaned["volume"])
    assert cleaned["high"].iloc[0] == 610
