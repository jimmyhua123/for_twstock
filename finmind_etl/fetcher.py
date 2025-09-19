"""FinMind API 下載與退避重試邏輯。"""

from __future__ import annotations

import time
from typing import Dict, Optional

import pandas as pd
import requests

from .config import API_URL, DEFAULT_TIMEOUT, LOGGER


class FetchError(Exception):
    """自訂例外，表示資料取得失敗。"""


def build_session() -> requests.Session:
    """建立帶有基本設定的 Session。"""

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def request_with_retries(
    session: requests.Session,
    params: Dict[str, str],
    token: Optional[str],
    rate_limit_sleep: float,
    retries: int,
    use_header: bool,
) -> Optional[Dict[str, object]]:
    """發送請求並處理重試與速率限制。"""

    backoff = rate_limit_sleep
    for attempt in range(1, retries + 1):
        headers = {}
        effective_params = dict(params)
        if token:
            if use_header:
                headers["Authorization"] = f"Bearer {token}"
            else:
                effective_params["token"] = token
        log_params = dict(effective_params)
        if "token" in log_params:
            log_params["token"] = "***MASKED***"
        LOGGER.info("開始請求，第 %d 次，參數：%s", attempt, log_params)
        try:
            response = session.get(
                API_URL,
                params=effective_params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            LOGGER.error("請求發生例外：%s", exc)
            if attempt == retries:
                return None
            time.sleep(backoff)
            backoff *= 1.8
            continue

        if response.status_code in {402, 429}:
            LOGGER.warning("超出使用上限，稍候重試 (HTTP %s)", response.status_code)
            if attempt == retries:
                return None
            time.sleep(backoff)
            backoff *= 1.8
            continue

        if response.status_code != 200:
            LOGGER.error(
                "HTTP 狀態碼 %s，無法取得資料：%s",
                response.status_code,
                response.text,
            )
            if attempt == retries:
                return None
            time.sleep(backoff)
            backoff *= 1.8
            continue

        try:
            payload = response.json()
        except ValueError:
            LOGGER.error("回傳內容非 JSON，內容：%s", response.text[:200])
            return None

        if payload.get("status") != 200:
            LOGGER.error(
                "API 回應錯誤：status=%s, msg=%s",
                payload.get("status"),
                payload.get("msg"),
            )
            if attempt == retries:
                return None
            time.sleep(backoff)
            backoff *= 1.8
            continue

        LOGGER.info("請求成功，筆數：%d", len(payload.get("data", [])))
        return payload

    return None


def fetch_dataset(
    session: requests.Session,
    dataset: str,
    stock_id: Optional[str],
    start: str,
    end: str,
    token: Optional[str],
    rate_limit_sleep: float,
    retries: int,
) -> pd.DataFrame:
    """下載單一 dataset 的資料，並於必要時更換授權方法。"""

    params = {
        "dataset": dataset,
        "start_date": start,
        "end_date": end,
    }
    if stock_id:
        params["data_id"] = stock_id

    payload = request_with_retries(
        session,
        params,
        token,
        rate_limit_sleep,
        retries,
        use_header=True,
    )

    if payload is None and token:
        LOGGER.warning("Header 授權失敗，改以 query 參數附帶 token。")
        payload = request_with_retries(
            session,
            params,
            token,
            rate_limit_sleep,
            retries,
            use_header=False,
        )

    if payload is None:
        raise FetchError(f"dataset={dataset}, stock_id={stock_id} 下載失敗")

    data = payload.get("data", [])
    df = pd.DataFrame(data)
    LOGGER.info("原始資料欄位：%s", list(df.columns))
    return df


__all__ = [
    "FetchError",
    "build_session",
    "request_with_retries",
    "fetch_dataset",
]
