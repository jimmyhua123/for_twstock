"""FinMind API v4 輔助函式。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from . import CACHE_DIR, get_logger

LOGGER = get_logger(__name__)
BASE_URL = "https://api.finmindtrade.com/api/v4/data"
TRANSLATION_URL = "https://api.finmindtrade.com/api/v4/translation"
DEFAULT_TIMEOUT = 30


class FinMindAPIError(RuntimeError):
    """表示 FinMind API 呼叫失敗。"""


@dataclass
class FinMindClient:
    """FinMind API 包裝器，提供重試、快取與欄位翻譯。"""

    token: str | None = None
    cache_dir: str | Path | None = None
    force_refresh: bool = False
    max_retries: int = 5
    backoff_factor: float = 1.8
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self.token = self.token or os.getenv("FINMIND_TOKEN") or ""
        self.cache_path = Path(self.cache_dir) if self.cache_dir else CACHE_DIR
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._translation_cache: dict[str, dict[str, str]] = {}

    # -- 對外介面 ---------------------------------------------------------

    def get_dataset(self, dataset: str, params: Dict[str, Any]) -> pd.DataFrame:
        """取得指定 dataset 並轉為 DataFrame。

        會自動附加 token、處理 HTTP 重試、解析回傳資料，並以
        ``dataset`` 與查詢參數作為鍵寫入 Parquet 快取，避免重複下載。
        """

        effective_params = {key: value for key, value in params.items() if value not in (None, "")}
        effective_params.setdefault("dataset", dataset)
        if self.token:
            effective_params.setdefault("token", self.token)

        cache_path = self._cache_file_path(dataset, effective_params)
        if not self.force_refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                LOGGER.info("使用快取：dataset=%s rows=%s", dataset, len(cached))
                return cached

        payload = self._request_with_retry(BASE_URL, effective_params)
        data = payload.get("data", []) if isinstance(payload, dict) else []
        df = pd.DataFrame(data)
        if df.empty:
            LOGGER.warning("dataset=%s 於參數 %s 無資料", dataset, self._mask_token(effective_params))
            self._write_cache(cache_path, df)
            return df

        df = self._standardize_dataframe(dataset, df)
        self._write_cache(cache_path, df)
        LOGGER.info(
            "下載 dataset=%s rows=%s range=%s~%s",
            dataset,
            len(df),
            df["date"].min() if "date" in df.columns else "?",
            df["date"].max() if "date" in df.columns else "?",
        )
        return df

    def get_translation(self, dataset: str) -> dict[str, str]:
        """查詢欄位翻譯，回傳 ``{原始欄位: 標準英文欄位}`` 映射。

        FinMind 會依 dataset 提供 ``field`` 與對應英文名稱，本函式會將
        英文名稱轉為 snake_case，以利後續欄位統一處理。
        """

        if dataset in self._translation_cache:
            return self._translation_cache[dataset]

        params = {"dataset": dataset}
        if self.token:
            params["token"] = self.token
        try:
            payload = self._request_with_retry(TRANSLATION_URL, params)
        except FinMindAPIError as exc:
            LOGGER.warning("取得 dataset=%s 欄位翻譯失敗：%s", dataset, exc)
            self._translation_cache[dataset] = {}
            return {}

        translation_map: dict[str, str] = {}
        for item in payload.get("data", []):
            field = item.get("field") or item.get("origin_field") or item.get("column_name")
            if not field:
                continue
            en_value = (
                item.get("en")
                or item.get("en-us")
                or item.get("en_us")
                or item.get("en_name")
                or item.get("english")
            )
            if not en_value:
                continue
            normalized = self._normalize_field_name(str(en_value))
            translation_map[str(field)] = normalized

        if translation_map:
            LOGGER.info("dataset=%s 欄位翻譯：%s", dataset, translation_map)
        self._translation_cache[dataset] = translation_map
        return translation_map

    # -- 內部工具 ---------------------------------------------------------

    def _cache_file_path(self, dataset: str, params: Dict[str, Any]) -> Path:
        filtered = {k: v for k, v in params.items() if k != "token"}
        serialized = json.dumps(filtered, sort_keys=True, default=str)
        digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
        return self.cache_path / f"{dataset}_{digest}.parquet"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - 欲保持流程不中斷
            LOGGER.warning("讀取快取失敗（%s），將重新抓取。", exc)
            return None
        return df

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001 - 快取失敗不影響主流程
            LOGGER.warning("寫入快取失敗：%s", exc)

    def _request_with_retry(self, url: str, params: Dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # noqa: PERF203 - 需攔截所有網路錯誤
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 500 or response.status_code == 429:
                last_error = FinMindAPIError(
                    f"HTTP {response.status_code}: {response.text.strip()}"
                )
                self._sleep_backoff(attempt)
                continue

            try:
                payload = response.json()
            except ValueError as exc:  # noqa: PERF203 - JSON 解析失敗需重試
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if not isinstance(payload, dict):
                last_error = FinMindAPIError(f"非預期回應格式：{payload!r}")
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if payload.get("status") != 200:
                last_error = FinMindAPIError(str(payload.get("msg", "未知錯誤")))
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            return payload

        message = f"FinMind API 呼叫失敗：{last_error}" if last_error else "未知錯誤"
        raise FinMindAPIError(message)

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_factor ** attempt
        LOGGER.debug("第 %s 次嘗試失敗，等待 %.2f 秒後重試", attempt, delay)
        time.sleep(delay)

    def _standardize_dataframe(self, dataset: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.tz_localize(None)
        if "stock_id" in df.columns:
            df["stock_id"] = df["stock_id"].astype(str).str.zfill(4)

        translation = self.get_translation(dataset)
        rename_map = {col: translation[col] for col in df.columns if col in translation}
        if rename_map:
            df = df.rename(columns=rename_map)
        numeric_columns = [col for col in df.columns if col not in {"date", "stock_id", "type"}]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.sort_values([c for c in ["stock_id", "date"] if c in df.columns]).reset_index(drop=True)
        return df

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        name = name.strip().replace("/", "_").replace(" ", "_")
        return name.replace("-", "_").lower()

    @staticmethod
    def _mask_token(params: Dict[str, Any]) -> Dict[str, Any]:
        masked = dict(params)
        if "token" in masked:
            masked["token"] = "***"
        return masked


if __name__ == "__main__":  # pragma: no cover - 方便快速手動測試
    client = FinMindClient()
    try:
        sample = client.get_dataset(
            "TaiwanStockMonthRevenue",
            {"data_id": "2330", "start_date": "2023-01-01", "end_date": "2023-12-31"},
        )
        print("sample shape", sample.shape)
    except FinMindAPIError as exc:
        print("API 測試失敗：", exc)
