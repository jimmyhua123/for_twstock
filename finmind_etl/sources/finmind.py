from __future__ import annotations

import os

import pandas as pd

from .base import HttpClient

BASE = "https://api.finmindtrade.com/api/v4"


def _token() -> str | None:
    return os.environ.get("FINMIND_TOKEN")


def fetch(dataset: str, **params) -> pd.DataFrame:
    client = HttpClient()
    query = {"dataset": dataset, **params}
    token = _token()
    if token:
        query["token"] = token
    r = client.get(BASE, params=query)
    js = r.json()
    if not js.get("data"):
        return pd.DataFrame()
    return pd.DataFrame(js["data"]).reset_index(drop=True)
