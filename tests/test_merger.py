"""merger 模組測試。"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from finmind_etl.merger import _merge_daily_wide, merge_frames


def test_merge_frames_combines_datasets() -> None:
    """確認多資料集合併後保留鍵與欄位。"""

    price = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "stock_id": ["2330", "2330"],
            "close": [600, 610],
        }
    )
    inst = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "stock_id": ["2330"],
            "foreign": [100],
        }
    )
    frames = {
        "TaiwanStockPrice": price,
        "TaiwanStockInstitutionalInvestorsBuySell": inst,
    }

    merged = merge_frames(frames)

    assert {"date", "stock_id"}.issubset(merged.columns)
    assert "TaiwanStockPrice_close" in merged.columns
    assert "TaiwanStockInstitutionalInvestorsBuySell_foreign" in merged.columns
    assert len(merged) == 2


def test_merge_daily_wide_sums_institutional_columns() -> None:
    """確認每日寬表可正確加總法人欄位。"""

    price = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "stock_id": ["2330", "2330"],
            "open": [600, 600],
        }
    )
    inst = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "stock_id": ["2330", "2330"],
            "inst_foreign": [50, 25],
            "inst_investment_trust": [np.nan, 10],
        }
    )

    merged = _merge_daily_wide(price, inst)

    assert merged.loc[0, "inst_foreign"] == 75
    assert merged.loc[0, "inst_investment_trust"] == 10
