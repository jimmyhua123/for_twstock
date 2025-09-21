"""FinMind API 輔助函式：處理資料抓取與十年線計算。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import pandas as pd
import requests

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_TIMEOUT = 20

LOGGER = logging.getLogger("finmind_etl.finmind_api")


def _normalise_symbol(symbol: str | None) -> str:
    text = str(symbol or "").strip()
    if text.isdigit():
        return text.zfill(4)
    return text


def _build_attempts(token: str | None, params: Dict[str, Any]) -> Iterable[tuple[Dict[str, str], Dict[str, Any]]]:
    query = {key: value for key, value in params.items() if value not in (None, "")}
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        query_without_token = dict(query)
        query_without_token.pop("token", None)
        yield headers, query_without_token
        query_with_token = dict(query_without_token)
        query_with_token["token"] = token
        yield {}, query_with_token
    else:
        yield {}, query


def _parse_response(dataset: str, response: requests.Response) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """解析 FinMind 回應並回傳結果或錯誤資訊。"""

    error_message: str | None = None
    level_required: str | None = None

    if response.status_code != 200:
        try:
            payload = response.json()
            error_message = str(payload.get("msg", ""))
        except ValueError:
            error_message = response.text.strip()

        if response.status_code == 400 and error_message and "Your level is register" in error_message:
            level_required = "Sponsor"
            LOGGER.warning(
                "Dataset %s requires higher level (Sponsor). Fallback to price-based MA10Y.",
                dataset,
            )
        else:
            LOGGER.warning(
                "Dataset %s request failed: HTTP %s %s",
                dataset,
                response.status_code,
                error_message or response.text,
            )
        return None, error_message, level_required

    try:
        payload = response.json()
    except ValueError as exc:
        LOGGER.warning("Dataset %s JSON decode error: %s", dataset, exc)
        return None, str(exc), None

    status = payload.get("status")
    if status != 200:
        message = str(payload.get("msg", ""))
        if "Your level is register" in message:
            LOGGER.warning(
                "Dataset %s requires higher level (Sponsor). Fallback to price-based MA10Y.",
                dataset,
            )
            return None, message, "Sponsor"
        LOGGER.warning("Dataset %s request failed: status=%s msg=%s", dataset, status, message)
        return None, message, None

    data = payload.get("data")
    if not data:
        return pd.DataFrame(), None, None

    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df = df.sort_values([col for col in ["stock_id", "date"] if col in df.columns]).reset_index(drop=True)
    return df, None, None


def fetch_dataset(dataset: str, params: dict[str, Any]) -> pd.DataFrame:
    """以通用方式呼叫 FinMind v4 API。"""

    effective = dict(params)
    token = effective.pop("token", None)
    query = {"dataset": dataset, **effective}

    last_error: str | None = None
    level_required: str | None = None

    for headers, query_params in _build_attempts(token, query):
        try:
            response = requests.get(FINMIND_API, params=query_params, headers=headers, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:  # noqa: PERF203 - 需完整捕捉
            last_error = str(exc)
            LOGGER.warning("Dataset %s request failed: %s", dataset, exc)
            continue

        parsed, message, level = _parse_response(dataset, response)
        if parsed is not None:
            return parsed

        last_error = message
        if level is not None:
            level_required = level
            break

    df = pd.DataFrame()
    if last_error:
        df.attrs["error_message"] = last_error
    if level_required:
        df.attrs["level_required"] = level_required
    return df


def fetch_stock_price(
    symbol: str,
    start_date: str,
    end_date: str | None,
    token: str | None,
    use_adj: bool = True,
) -> pd.DataFrame:
    """抓 TaiwanStockPriceAdj（預設）或 TaiwanStockPrice。"""

    stock_id = _normalise_symbol(symbol)
    dataset = "TaiwanStockPriceAdj" if use_adj else "TaiwanStockPrice"
    df = fetch_dataset(
        dataset,
        {
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": token,
        },
    )

    if df.empty:
        return df

    df = df.copy()
    if "stock_id" not in df.columns:
        df["stock_id"] = stock_id
    df["stock_id"] = df["stock_id"].astype(str).str.strip().replace({"": stock_id})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    price_column = "adj_close" if use_adj and "adj_close" in df.columns else "close"
    if price_column not in df.columns and "close" in df.columns:
        price_column = "close"
    df["close"] = pd.to_numeric(df.get(price_column), errors="coerce")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df.attrs["symbol"] = stock_id
    return df


def try_fetch_10y_api(
    symbol: str,
    start_date: str,
    end_date: str | None,
    token: str | None,
) -> pd.DataFrame | None:
    """僅在使用者帶 --use-10y-api 時呼叫 TaiwanStock10Year。"""

    stock_id = _normalise_symbol(symbol)
    df = fetch_dataset(
        "TaiwanStock10Year",
        {
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": token,
        },
    )

    if df.empty:
        if df.attrs.get("level_required") == "Sponsor":
            return None
        error_message = df.attrs.get("error_message")
        if error_message:
            LOGGER.warning("TaiwanStock10Year 抓取失敗：%s", error_message)
        else:
            LOGGER.warning("TaiwanStock10Year 回傳空資料，改用股價推算。")
        return None

    rename_map = {}
    for column in ("MA10Y", "avg_price_10y", "avg", "value", "avg_price"):
        if column in df.columns:
            rename_map[column] = "MA10Y"
            break
    df = df.rename(columns=rename_map)
    if "MA10Y" not in df.columns:
        LOGGER.warning("TaiwanStock10Year 回傳資料缺少十年線欄位，改用股價推算。")
        return None

    df = df.copy()
    df["MA10Y"] = pd.to_numeric(df["MA10Y"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "stock_id" not in df.columns:
        df["stock_id"] = stock_id
    df["stock_id"] = df["stock_id"].astype(str).str.strip().replace({"": stock_id})
    keep = ["stock_id", "date", "MA10Y"]
    df = df[keep]
    df = df.dropna(subset=["date"]).sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df


def compute_ma_from_price(
    df_price: pd.DataFrame,
    window_days: int = 2400,
    price_col: str = "close",
) -> pd.DataFrame:
    """從價格資料計算十年線。"""

    if df_price is None or df_price.empty:
        return pd.DataFrame(columns=["stock_id", "date", "MA10Y"])

    df = df_price.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=["stock_id", "date", "MA10Y"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "date", "MA10Y"])

    if "stock_id" not in df.columns:
        symbol = df_price.attrs.get("symbol")
        df["stock_id"] = _normalise_symbol(symbol or "")

    df["stock_id"] = df["stock_id"].astype(str).str.strip().apply(_normalise_symbol)
    df[price_col] = pd.to_numeric(df.get(price_col), errors="coerce")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    results: list[pd.DataFrame] = []
    notified: set[str] = set()

    for stock_id, group in df.groupby("stock_id", dropna=False):
        if group.empty:
            continue
        sorted_group = group.sort_values("date")
        rolling = sorted_group[price_col].rolling(window=window_days, min_periods=1).mean()
        if len(sorted_group) < window_days:
            since = sorted_group["date"].min()
            symbol = _normalise_symbol(stock_id)
            if symbol not in notified and pd.notna(since):
                LOGGER.info(
                    "Ticker %s has less than 10-year data since %s; MA10Y is an approximation.",
                    symbol,
                    since.strftime("%Y-%m-%d"),
                )
                notified.add(symbol)
        result = pd.DataFrame({"stock_id": stock_id, "date": sorted_group["date"], "MA10Y": rolling})
        results.append(result)

    if not results:
        return pd.DataFrame(columns=["stock_id", "date", "MA10Y"])

    combined = pd.concat(results, ignore_index=True)
    combined["stock_id"] = combined["stock_id"].astype(str).str.strip().apply(_normalise_symbol)
    combined = combined.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return combined


__all__ = [
    "FINMIND_API",
    "fetch_dataset",
    "fetch_stock_price",
    "try_fetch_10y_api",
    "compute_ma_from_price",
]

