from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import pandas as pd

from .base import HttpClient

BASE = "https://www.tpex.org.tw"


def _parse_roc_date(value: str) -> dt.date:
    value = value.replace("/", "-")
    parts = value.split("-")
    if len(parts[0]) == 3:
        year = int(parts[0]) + 1911
    else:
        year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    return dt.date(year, month, day)


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"--", "-"}:
        return None
    return float(value.replace(",", ""))


def _to_int(value: str) -> int | None:
    value = value.strip()
    if not value or value in {"--", "-"}:
        return None
    return int(value.replace(",", ""))


def fetch_stock_day_csv(stock_id: str, roc_year: int, month: int) -> pd.DataFrame:
    """TPEx（上櫃）日成交價量。"""

    client = HttpClient()
    date_str = f"{roc_year:03d}/{month:02d}"
    params = {"l": "zh-tw", "d": date_str, "stkno": stock_id}
    r = client.get(f"{BASE}/web/stock/aftertrading/daily_trading_info/st43_result.php", params=params)
    try:
        payload = r.json()
    except ValueError:
        return pd.DataFrame(columns=["date", "stock_id", "open", "high", "low", "close", "volume"])

    data = payload.get("aaData") or []
    rows: List[Dict[str, Any]] = []
    for row in data:
        try:
            trade_date = _parse_roc_date(str(row[0]))
        except Exception:
            continue
        volume = _to_int(str(row[1]))
        open_price = _to_float(str(row[3]))
        high_price = _to_float(str(row[4]))
        low_price = _to_float(str(row[5]))
        close_price = _to_float(str(row[6]))
        rows.append(
            {
                "date": trade_date,
                "stock_id": str(stock_id),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df
