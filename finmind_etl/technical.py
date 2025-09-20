"""Download and clean technical datasets from FinMind."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .api import APIClient
from .datasets import DatasetResult, DatasetSpec, clean_dataset, iter_category_specs

LOGGER = logging.getLogger("finmind_etl.technical")


@dataclass
class FetchOptions:
    stocks: Sequence[str]
    since: Optional[str]
    until: Optional[str]


def _fetch_single_dataset(
    client: APIClient,
    spec: DatasetSpec,
    options: FetchOptions,
) -> DatasetResult:
    """Download a dataset (possibly per stock) and return raw + clean frames."""

    frames_raw: List[pd.DataFrame] = []
    frames_clean: List[pd.DataFrame] = []
    translation = client.try_translation(spec.name)

    if spec.requires_stock:
        for stock in options.stocks:
            try:
                raw = client.fetch_dataset(spec.name, stock, options.since, options.until)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("%s 抓取失敗：%s", spec.name, exc)
                continue
            if raw is None or raw.empty:
                continue
            frames_raw.append(raw.assign(stock_id=raw.get("stock_id", stock)))
            clean = clean_dataset(spec, raw, translation)
            if clean.empty:
                continue
            frames_clean.append(clean)
    else:
        try:
            raw = client.fetch_dataset(spec.name, None, options.since, options.until)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s 抓取失敗：%s", spec.name, exc)
            return DatasetResult(spec, pd.DataFrame(), pd.DataFrame())
        frames_raw.append(raw)
        frames_clean.append(clean_dataset(spec, raw, translation))

    if frames_raw:
        raw_df = pd.concat(frames_raw, ignore_index=True)
    else:
        raw_df = pd.DataFrame()

    if frames_clean:
        clean_df = pd.concat(frames_clean, ignore_index=True)
    else:
        clean_df = pd.DataFrame(columns=list(spec.required_fields))

    return DatasetResult(spec=spec, raw=raw_df, clean=clean_df)


def fetch_technical_data(
    stocks: Sequence[str],
    since: str,
    client: APIClient,
    end_date: Optional[str] = None,
) -> Dict[str, DatasetResult]:
    """Fetch all technical datasets defined in :mod:`finmind_etl.datasets`."""

    options = FetchOptions(stocks=list({s.strip().zfill(4) for s in stocks}), since=since, until=end_date)
    results: Dict[str, DatasetResult] = {}

    for spec in iter_category_specs("technical"):
        LOGGER.info("抓取技術面資料：dataset=%s", spec.name)
        result = _fetch_single_dataset(client, spec, options)
        results[spec.name] = result

    return results


__all__ = ["fetch_technical_data"]

