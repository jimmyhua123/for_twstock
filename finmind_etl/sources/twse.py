from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import pandas as pd

from .base import HttpClient

BASE = "https://www.twse.com.tw"


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


def fetch_stock_day(stock_id: str, date: dt.date) -> pd.DataFrame:
    """TWSE 日成交價量（上市），以月為單位回傳，再自行展平。"""

    client = HttpClient()
    params: Dict[str, Any] = {
        "response": "json",
        "date": date.strftime("%Y%m01"),
        "stockNo": stock_id,
    }
    r = client.get(f"{BASE}/exchangeReport/STOCK_DAY", params=params)
    js = r.json()
    if "data" not in js:
        return pd.DataFrame(columns=["date", "stock_id", "open", "high", "low", "close", "volume"])

    rows: List[Dict[str, Any]] = []
    for row in js.get("data", []):
        try:
            trade_date = _parse_roc_date(str(row[0]))
        except Exception:
            continue
        open_price = _to_float(str(row[3]))
        high_price = _to_float(str(row[4]))
        low_price = _to_float(str(row[5]))
        close_price = _to_float(str(row[6]))
        volume = _to_int(str(row[1]))
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
