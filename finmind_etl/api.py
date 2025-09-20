"""FinMind API 介面封裝模組。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4"
DATA_ENDPOINT = f"{BASE_URL}/data"
TRANSLATION_ENDPOINT = f"{BASE_URL}/translation"
DATALIST_ENDPOINT = f"{BASE_URL}/datalist"
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
RATE_LIMIT_SLEEP = 0.15

LOGGER = logging.getLogger("finmind_etl.api")


class FinMindAPIError(RuntimeError):
    """表示 FinMind API 回應失敗。"""


@dataclass
class APIClient:
    """簡化的 FinMind API 用戶端。

    Parameters
    ----------
    token:
        FinMind API token。會優先以 Bearer header 附帶，若失敗再回退至
        query string，符合官方建議的授權方式。
    session:
        重複使用的 :class:`requests.Session` 實例，可減少 TCP 開銷。
    retries:
        單一請求的最大重試次數。
    timeout:
        HTTP 請求逾時秒數。
    backoff:
        速率限制時的起始等待秒數，會以 2 的指數倍數增加。
    """

    token: Optional[str] = None
    session: Optional[requests.Session] = None
    retries: int = DEFAULT_RETRIES
    timeout: int = DEFAULT_TIMEOUT
    backoff: float = RATE_LIMIT_SLEEP

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    # -- 核心請求邏輯 -------------------------------------------------

    def request_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """送出 GET 請求並回傳 JSON。

        會自動附帶 Bearer token、實作指數退避與節流，確保在 API 回傳
        ``status != 200`` 或 HTTP 錯誤時能夠重試。僅在成功時回傳字典，否則
        拋出 :class:`FinMindAPIError`。
        """

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        effective_params = {k: v for k, v in params.items() if v not in (None, "")}
        last_error: Optional[str] = None

        # 依指示採用 Bearer header；若三次皆失敗再嘗試 query token。
        use_header_first = bool(self.token)
        attempts: Iterable[bool]
        if use_header_first:
            attempts = (True, False)
        else:
            attempts = (False,)

        for header_first in attempts:
            for attempt in range(1, self.retries + 1):
                headers: Dict[str, str] = {}
                params_to_use = dict(effective_params)
                if self.token:
                    if header_first:
                        headers["Authorization"] = f"Bearer {self.token}"
                    else:
                        params_to_use["token"] = self.token

                try:
                    response = self.session.get(
                        url,
                        params=params_to_use,
                        headers=headers,
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:  # noqa: PERF203 - 需完整捕捉
                    last_error = str(exc)
                    self._sleep(attempt)
                    continue

                if response.status_code in {402, 429}:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    self._sleep(attempt)
                    continue

                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.retries:
                        break
                    self._sleep(attempt)
                    continue

                try:
                    payload = response.json()
                except ValueError as exc:  # noqa: PERF203 - JSON 解析可能失敗
                    last_error = f"JSON decode error: {exc}"
                    self._sleep(attempt)
                    continue

                if not isinstance(payload, dict):
                    last_error = f"Unexpected payload type: {type(payload)}"
                    self._sleep(attempt)
                    continue

                status = payload.get("status")
                if status != 200:
                    message = str(payload.get("msg", "unknown error"))
                    last_error = f"status={status} msg={message}"
                    # FinMind 於額度不足時訊息通常會包含 limit
                    if "limit" not in message.lower():
                        # 對於非額度錯誤不需要重試
                        break
                    self._sleep(attempt)
                    continue

                data = payload.get("data")
                if data is None or not isinstance(data, (list, dict)):
                    raise FinMindAPIError("回傳資料格式異常：缺少 data")

                time.sleep(RATE_LIMIT_SLEEP)
                return payload

            if last_error:
                LOGGER.warning("授權模式 %s 失敗：%s", "header" if header_first else "query", last_error)

        raise FinMindAPIError(last_error or "FinMind API 呼叫失敗")

    # -- 高階封裝 -----------------------------------------------------

    def fetch_dataset(
        self,
        dataset: str,
        data_id: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """下載資料集並轉為 :class:`pandas.DataFrame`。"""

        params: Dict[str, Any] = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        payload = self.request_json("data", params)
        data = payload.get("data")
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:  # 理論上不會到這裡，前面已檢查
            raise FinMindAPIError("無法解析 API 回傳資料格式")

        if not df.empty:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "stock_id" in df.columns:
                df["stock_id"] = df["stock_id"].astype(str)
        return df.sort_values([c for c in ["stock_id", "date"] if c in df.columns]).reset_index(drop=True)

    def try_translation(self, dataset: str) -> Dict[str, str]:
        """嘗試取得欄位翻譯。失敗時回傳空字典。"""

        try:
            payload = self.request_json("translation", {"dataset": dataset})
        except FinMindAPIError as exc:
            LOGGER.info("translation 取得失敗：dataset=%s %s", dataset, exc)
            return {}
        mapping: Dict[str, str] = {}
        for item in payload.get("data", []):
            raw = str(item.get("field") or item.get("origin_field") or "").strip()
            english = str(
                item.get("en")
                or item.get("en_name")
                or item.get("en-us")
                or item.get("english")
                or ""
            ).strip()
            if not raw or not english:
                continue
            mapping[raw] = english.replace("/", "_").replace(" ", "_").replace("-", "_").lower()
        return mapping

    def try_datalist(self, dataset: str) -> Dict[str, Any]:
        """嘗試取得 dataset 說明。失敗時回傳空字典。"""

        try:
            payload = self.request_json("datalist", {"dataset": dataset})
        except FinMindAPIError as exc:
            LOGGER.info("datalist 取得失敗：dataset=%s %s", dataset, exc)
            return {}
        return payload

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff * (2 ** (attempt - 1))
        time.sleep(delay)


def request_json(path: str, params: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    """模組級封裝，方便舊程式呼叫。"""

    client = APIClient(token=token)
    endpoint = path.lstrip("/")
    return client.request_json(endpoint, params)


def fetch_dataset(
    dataset: str,
    data_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    token: Optional[str],
) -> pd.DataFrame:
    """模組級封裝，提供與指示相符的函式介面。"""

    client = APIClient(token=token)
    return client.fetch_dataset(dataset, data_id, start_date, end_date)


def try_translation(dataset: str, token: Optional[str]) -> Dict[str, str]:
    """取得欄位翻譯。"""

    client = APIClient(token=token)
    return client.try_translation(dataset)


def try_datalist(dataset: str, token: Optional[str]) -> Dict[str, Any]:
    """取得 dataset 說明資訊。"""

    client = APIClient(token=token)
    return client.try_datalist(dataset)


__all__ = [
    "APIClient",
    "FinMindAPIError",
    "fetch_dataset",
    "request_json",
    "try_translation",
    "try_datalist",
]
