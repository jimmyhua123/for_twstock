from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests


class HttpClient:
    def __init__(self, retry: int = 3, backoff: float = 1.2, timeout: int = 30):
        self.retry = retry
        self.backoff = backoff
        self.timeout = timeout

    def get(
        self,
        url: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> requests.Response:
        for i in range(self.retry):
            r = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (402, 429, 500, 502, 503, 504):
                time.sleep(self.backoff ** (i + 1))
                continue
            r.raise_for_status()
        r.raise_for_status()


class Cache:
    def __init__(self, cache_dir: str):
        self.dir = pathlib.Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, dataset: str, params: Dict[str, Any]) -> pathlib.Path:
        raw = dataset + "|" + json.dumps(params, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return self.dir / f"{dataset}__{h}.parquet"

    def load(self, dataset: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        p = self._key(dataset, params)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                return None
        return None

    def save(self, dataset: str, params: Dict[str, Any], df: pd.DataFrame) -> None:
        p = self._key(dataset, params)
        df.to_parquet(p, index=False)
