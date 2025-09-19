"""\
使用說明:
    python finmind_fetcher.py --stocks 2330,2317 --datasets TaiwanStockPrice,TaiwanStockInstitutionalInvestorsBuySell

參數範例:
    python finmind_fetcher.py \
        --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wOS0xOSAwOTowMjowMSIsInVzZXJfaWQiOiJqaW1teWh1YSIsImlwIjoiMTE4LjE2My44My43MiJ9.kKDLDia6fDUxyEBQDC9Z2GanXsbn0ZmxWk5vr2N-QS8 \
        --stocks 2330,2317,2454 \
        --start 2024-01-01 --end 2024-12-31 \
        --datasets TaiwanStockPrice,TaiwanStockPriceAdj,TaiwanStockInstitutionalInvestorsBuySell \
        --outdir ./finmind_out --parquet --rate-limit-sleep 5 --retries 5

可能的回應代碼與意義:
    200: 成功取得資料。
    400: 參數錯誤或資料不存在，請檢查輸入。
    401/403: 授權失敗，請確認 token 是否有效。
    402: 超出使用上限，請稍候再試。
    429: 觸發速率限制，請減少請求頻率。
    5xx: 伺服器暫時無法服務，建議稍後重試。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
import time
from functools import reduce
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_TIMEOUT = 20

# 集中管理 dataset 與欄位設定，方便未來調整
DATASET_CATALOG: Dict[str, Dict[str, object]] = {
    "TaiwanStockPrice": {
        "description": "台股日價量",
        "normalizer": "normalize_taiwan_stock_price",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockPriceAdj": {
        "description": "台股日價量(還原權息)",
        "normalizer": "normalize_taiwan_stock_price_adj",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "description": "三大法人買賣超",
        "normalizer": "normalize_institutional_investors",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "description": "融資融券餘額",
        "normalizer": "normalize_margin_short",
        "requires_stock": True,
        "frequency": "D",
    },
    "TaiwanStockMonthRevenue": {
        "description": "月營收",
        "normalizer": "normalize_month_revenue",
        "requires_stock": True,
        "frequency": "M",
    },
}

# 針對輸出的檔名提供簡潔的辨識標籤，便於快速分辨資料內容
DATASET_FILENAME_TAG: Dict[str, str] = {
    "TaiwanStockPrice": "stockprice",
    "TaiwanStockPriceAdj": "stockprice_adj",
    "TaiwanStockInstitutionalInvestorsBuySell": "buysell",
    "TaiwanStockMarginPurchaseShortSale": "margin_short",
    "TaiwanStockMonthRevenue": "monthrevenue",
}

# 欄位對映定義區，針對不同 dataset 指定原始欄位與標準欄位的關係
COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "TaiwanStockPrice": {
        "max": "high",
        "min": "low",
        "Trading_Volume": "volume",
        "Trading_money": "turnover",
        "Trading_turnover": "transactions",
    },
    "TaiwanStockPriceAdj": {
        "max": "high",
        "min": "low",
        "Trading_Volume": "volume",
        "Trading_money": "turnover",
        "Trading_turnover": "transactions",
        "Adj_Close": "adj_close",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "Foreign_Investor_Net_Buy_Sell": "foreign",
        "Investment_Trust_Net_Buy_Sell": "investment_trust",
        "Dealer_Net_Buy_Sell": "dealer",
        "Dealer_Self_Net_Buy_Sell": "dealer_self",
        "Dealer_Hedging_Net_Buy_Sell": "dealer_hedging",
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "MarginPurchaseTodayBalance": "margin_balance",
        "ShortSaleTodayBalance": "short_balance",
        "MarginPurchaseChange": "margin_change",
        "ShortSaleChange": "short_change",
    },
    "TaiwanStockMonthRevenue": {
        "revenue": "revenue",
        "revenue_month": "revenue_month",
        "revenue_year": "revenue_year",
        "revenue_last_month": "revenue_last_month",
        "revenue_last_year": "revenue_last_year",
        "revenue_month_growth": "revenue_month_growth",
        "revenue_year_growth": "revenue_year_growth",
        "accumulated_revenue": "accumulated_revenue",
        "accumulated_revenue_last_year": "accumulated_revenue_last_year",
        "accumulated_revenue_growth": "accumulated_revenue_growth",
    },
}

NUMERIC_COLUMN_HINTS: Dict[str, List[str]] = {
    "TaiwanStockPrice": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "spread",
        "transactions",
    ],
    "TaiwanStockPriceAdj": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "adj_close",
        "spread",
        "transactions",
    ],
    "TaiwanStockInstitutionalInvestorsBuySell": [
        "foreign",
        "investment_trust",
        "dealer",
        "dealer_self",
        "dealer_hedging",
        "total",
    ],
    "TaiwanStockMarginPurchaseShortSale": [
        "margin_balance",
        "short_balance",
        "margin_change",
        "short_change",
    ],
    "TaiwanStockMonthRevenue": [
        "revenue",
        "revenue_month",
        "revenue_year",
        "revenue_last_month",
        "revenue_last_year",
        "revenue_month_growth",
        "revenue_year_growth",
        "accumulated_revenue",
        "accumulated_revenue_last_year",
        "accumulated_revenue_growth",
    ],
}

LOGGER = logging.getLogger("finmind")
ERROR_LOGGER = logging.getLogger("finmind.errors")


def configure_logging(outdir: str) -> None:
    """設定 logging，並確保錯誤會寫入 errors.log。"""

    LOGGER.setLevel(logging.INFO)
    ERROR_LOGGER.setLevel(logging.ERROR)
    LOGGER.propagate = False
    ERROR_LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    if not LOGGER.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

    if not any(
        isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout
        for handler in ERROR_LOGGER.handlers
    ):
        error_stream_handler = logging.StreamHandler(sys.stdout)
        error_stream_handler.setLevel(logging.INFO)
        error_stream_handler.setFormatter(formatter)
        ERROR_LOGGER.addHandler(error_stream_handler)

    os.makedirs(outdir, exist_ok=True)
    error_path = os.path.join(outdir, "errors.log")
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == os.path.abspath(error_path)
        for handler in ERROR_LOGGER.handlers
    ):
        file_handler = logging.FileHandler(error_path, encoding="utf-8")
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        ERROR_LOGGER.addHandler(file_handler)


def build_session() -> requests.Session:
    """建立帶有基本設定的 Session。"""

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def coerce_numeric_columns(
    df: pd.DataFrame,
    target_columns: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    """將指定欄位轉換為數值型態。"""

    if target_columns is None:
        target_columns = df.columns
    if exclude is None:
        exclude = []
    exclude_set = {"date", "stock_id", *exclude}
    for column in target_columns:
        if column in exclude_set or column not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        df[column] = (
            pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
        )


def standardize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """統一處理 date 與 stock_id 欄位。"""

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    else:
        df["stock_id"] = np.nan

    return df


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """確保指定欄位存在，不存在時補上 NaN。"""

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def normalize_taiwan_stock_price(df: pd.DataFrame) -> pd.DataFrame:
    """清洗台股日價量資料。"""

    if df.empty:
        columns = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockPrice"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["open", "high", "low", "close", "volume", "turnover"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockPrice"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_taiwan_stock_price_adj(df: pd.DataFrame) -> pd.DataFrame:
    """清洗還原權息日價量資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "adj_close",
        ]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockPriceAdj"])
    if "high" not in df.columns and "max" in df.columns:
        df.rename(columns={"max": "high"}, inplace=True)
    if "low" not in df.columns and "min" in df.columns:
        df.rename(columns={"min": "low"}, inplace=True)
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["open", "high", "low", "close", "volume", "turnover", "adj_close"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockPriceAdj"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_institutional_investors(df: pd.DataFrame) -> pd.DataFrame:
    """清洗三大法人買賣超資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "foreign",
            "investment_trust",
            "dealer",
            "dealer_self",
            "dealer_hedging",
        ]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockInstitutionalInvestorsBuySell"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["foreign", "investment_trust", "dealer", "dealer_self", "dealer_hedging"],
    )
    coerce_numeric_columns(
        df,
        target_columns=NUMERIC_COLUMN_HINTS[
            "TaiwanStockInstitutionalInvestorsBuySell"
        ],
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_margin_short(df: pd.DataFrame) -> pd.DataFrame:
    """清洗融資融券資料。"""

    if df.empty:
        columns = [
            "date",
            "stock_id",
            "margin_balance",
            "short_balance",
            "margin_change",
            "short_change",
        ]
        return pd.DataFrame(columns=columns)

    if "MarginPurchaseChange" not in df.columns and {
        "MarginPurchaseTodayBalance",
        "MarginPurchaseYesterdayBalance",
    }.issubset(df.columns):
        df["MarginPurchaseChange"] = (
            pd.to_numeric(
                df["MarginPurchaseTodayBalance"], errors="coerce"
            )
            - pd.to_numeric(df["MarginPurchaseYesterdayBalance"], errors="coerce")
        )
    if "ShortSaleChange" not in df.columns and {
        "ShortSaleTodayBalance",
        "ShortSaleYesterdayBalance",
    }.issubset(df.columns):
        df["ShortSaleChange"] = (
            pd.to_numeric(df["ShortSaleTodayBalance"], errors="coerce")
            - pd.to_numeric(df["ShortSaleYesterdayBalance"], errors="coerce")
        )
    df = df.rename(columns=COLUMN_MAP["TaiwanStockMarginPurchaseShortSale"])
    df = standardize_common_fields(df)
    df = ensure_columns(
        df,
        ["margin_balance", "short_balance", "margin_change", "short_change"],
    )
    coerce_numeric_columns(
        df, target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockMarginPurchaseShortSale"]
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_month_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """清洗月營收資料，並加入頻率欄位。"""

    if df.empty:
        columns = ["date", "stock_id", "freq", "revenue"]
        return pd.DataFrame(columns=columns)

    df = df.rename(columns=COLUMN_MAP["TaiwanStockMonthRevenue"])
    df = standardize_common_fields(df)
    df["freq"] = "M"
    coerce_numeric_columns(
        df,
        target_columns=NUMERIC_COLUMN_HINTS["TaiwanStockMonthRevenue"],
        exclude=["freq"],
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


NORMALIZERS = {
    name: globals()[config["normalizer"]]
    for name, config in DATASET_CATALOG.items()
}


class FetchError(Exception):
    """自訂例外，表示資料取得失敗。"""


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


def save_frame(
    df: pd.DataFrame,
    path_csv: str,
    path_parquet: Optional[str] = None,
) -> None:
    """儲存 DataFrame 為 CSV（以及 Parquet）。"""

    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    LOGGER.info("輸出路徑 (CSV)：%s", path_csv)

    if df.empty:
        LOGGER.info(
            "資料為空，仍輸出表頭供參考，欄位：%s", list(df.columns)
        )
    else:
        preview = df.head(3).to_string(index=False)
        LOGGER.info("資料筆數：%d，欄位：%s", len(df), list(df.columns))
        LOGGER.info("前 3 筆預覽：\n%s", preview)

    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    if path_parquet:
        LOGGER.info("輸出路徑 (Parquet)：%s", path_parquet)
        os.makedirs(os.path.dirname(path_parquet), exist_ok=True)
        df.to_parquet(path_parquet, index=False)


def merge_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """依據日期與股票代號合併多個資料集。"""

    prepared_frames: List[pd.DataFrame] = []
    for dataset, df in frames.items():
        if df is None or df.empty:
            continue
        renamed = df.copy()
        rename_map = {
            column: f"{dataset}_{column}"
            for column in renamed.columns
            if column not in {"date", "stock_id"}
        }
        renamed = renamed.rename(columns=rename_map)
        prepared_frames.append(renamed)

    if not prepared_frames:
        LOGGER.warning("沒有任何資料可合併。")
        return pd.DataFrame(columns=["date", "stock_id"])

    def outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(left, right, on=["date", "stock_id"], how="outer")

    merged = reduce(outer_merge, prepared_frames)
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return merged


def _read_raw_merged(path: str) -> pd.DataFrame:
    """讀取外部提供的合併檔案。"""

    if not os.path.exists(path):
        LOGGER.warning("找不到檔案 %s，跳過清理流程。", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("讀取 %s 時發生錯誤: %s", path, exc)
        return pd.DataFrame()

    if df.empty:
        LOGGER.warning("檔案 %s 為空，無資料可清理。", path)
    return df


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """調整日期、股票代號與數值欄位的型態。"""

    if df.empty:
        return df.copy()

    normalized = df.copy()

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=False)
        try:
            normalized["date"] = normalized["date"].dt.tz_localize(None)
        except AttributeError:
            pass
    else:
        LOGGER.warning("缺少 date 欄位，後續輸出可能不完整。")

    if "stock_id" in normalized.columns:
        normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    else:
        LOGGER.warning("缺少 stock_id 欄位，後續輸出可能不完整。")

    text_like = {"date", "stock_id"}
    text_like.update({col for col in normalized.columns if "name" in col.lower()})
    text_like.update({col for col in normalized.columns if col.lower() in {"investor", "dataset"}})

    for column in normalized.columns:
        if column in text_like:
            continue
        series = normalized[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue
        cleaned = series.astype(str).str.strip().replace({"": np.nan})
        cleaned = cleaned.str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().any() or cleaned.eq("0").any():
            normalized[column] = numeric
        else:
            normalized[column] = cleaned

    return normalized


def _extract_price_block(df: pd.DataFrame) -> pd.DataFrame:
    """取得價量相關欄位並重新命名。"""

    if df.empty:
        return pd.DataFrame(columns=["date", "stock_id", "open", "high", "low", "close", "volume", "turnover"])

    price_prefix = "taiwanstockprice_"
    price_columns = [
        column
        for column in df.columns
        if column.lower().startswith(price_prefix)
    ]

    if not price_columns:
        LOGGER.warning("在資料中找不到台股價量欄位，僅輸出法人資料。")
        return pd.DataFrame(columns=["date", "stock_id"])

    selected_columns = [col for col in ["date", "stock_id"] if col in df.columns]
    selected_columns.extend(price_columns)
    price_df = df[selected_columns].copy()

    rename_map = {}
    for column in price_columns:
        lower = column.lower()
        if lower == "taiwanstockprice_open":
            rename_map[column] = "open"
        elif lower == "taiwanstockprice_high":
            rename_map[column] = "high"
        elif lower == "taiwanstockprice_low":
            rename_map[column] = "low"
        elif lower == "taiwanstockprice_close":
            rename_map[column] = "close"
        elif lower == "taiwanstockprice_volume":
            rename_map[column] = "volume"
        elif lower == "taiwanstockprice_turnover":
            rename_map[column] = "turnover"

    price_df = price_df.rename(columns=rename_map)

    group_keys = [col for col in ["date", "stock_id"] if col in price_df.columns]
    if not group_keys:
        return price_df

    agg_map = {
        column: "first"
        for column in price_df.columns
        if column not in {"date", "stock_id"}
    }
    if agg_map:
        price_df = price_df.groupby(group_keys, as_index=False).agg(agg_map)
    else:
        price_df = price_df.drop_duplicates(subset=group_keys)

    return price_df


def _find_candidate_column(df: pd.DataFrame, keywords: Iterable[str], exclude: Optional[Iterable[str]] = None) -> Optional[str]:
    """依據關鍵字尋找最適合的欄位名稱。"""

    excluded = {col.lower() for col in (exclude or [])}
    for column in df.columns:
        lower = column.lower()
        if lower in excluded:
            continue
        for keyword in keywords:
            key = keyword.lower()
            if lower == key or lower.endswith(f"_{key}") or key in lower:
                return column
    return None


def _build_institutional_wide(df: pd.DataFrame) -> pd.DataFrame:
    """整理法人買賣資料為寬表。"""

    empty_inst = pd.DataFrame(
        columns=[
            "date",
            "stock_id",
            "inst_foreign",
            "inst_investment_trust",
            "inst_dealer_self",
            "inst_dealer_hedging",
        ]
    )

    if df.empty:
        return empty_inst

    prefix = "taiwanstockinstitutionalinvestorsbuysell_"
    wide_columns = [
        column
        for column in df.columns
        if column.lower().startswith(prefix)
    ]

    has_long_format = _find_candidate_column(df, ["investor"]) is not None

    if wide_columns:
        if has_long_format:
            LOGGER.warning("同時偵測到法人長表與寬表欄位，將優先使用寬表資料。")

        selected = [col for col in ["date", "stock_id"] if col in df.columns]
        selected.extend(wide_columns)
        inst_df = df[selected].copy()

        rename_map = {}
        for column in wide_columns:
            lower = column.lower()
            if lower.endswith("_foreign"):
                rename_map[column] = "inst_foreign"
            elif lower.endswith("_investment_trust"):
                rename_map[column] = "inst_investment_trust"
            elif lower.endswith("_dealer_self"):
                rename_map[column] = "inst_dealer_self"
            elif lower.endswith("_dealer_hedging"):
                rename_map[column] = "inst_dealer_hedging"

        inst_df = inst_df.rename(columns=rename_map)

        group_keys = [col for col in ["date", "stock_id"] if col in inst_df.columns]
        agg_map = {
            column: "sum"
            for column in inst_df.columns
            if column not in {"date", "stock_id"}
        }
        if agg_map:
            inst_df = inst_df.groupby(group_keys, as_index=False).agg(agg_map)
        else:
            inst_df = inst_df.drop_duplicates(subset=group_keys)

        for column in empty_inst.columns:
            if column not in inst_df.columns:
                inst_df[column] = np.nan
        return inst_df[empty_inst.columns]

    if not has_long_format:
        LOGGER.warning("偵測不到法人欄位，僅輸出價量資料。")
        return empty_inst

    investor_col = _find_candidate_column(df, ["investor"])
    net_col = _find_candidate_column(df, ["net_buy_sell", "buy_sell"])
    buy_col = _find_candidate_column(df, ["buy"], exclude=[net_col] if net_col else None)
    sell_col = _find_candidate_column(df, ["sell"], exclude=[net_col] if net_col else None)

    if not investor_col or (not net_col and (not buy_col or not sell_col)):
        LOGGER.warning("法人長表欄位資訊不足，無法計算淨買賣超。")
        return empty_inst

    inst_df = df[[col for col in ["date", "stock_id", investor_col] if col in df.columns]].copy()

    if net_col:
        values = pd.to_numeric(df[net_col], errors="coerce")
    else:
        buy_series = pd.to_numeric(df[buy_col], errors="coerce").fillna(0)
        sell_series = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)
        values = buy_series - sell_series

    inst_df["value"] = values
    inst_df = inst_df.dropna(subset=["value"], how="all")

    if inst_df.empty:
        LOGGER.warning("法人長表計算後沒有有效數值。")
        return empty_inst

    def normalize_investor(name: object) -> str:
        text = str(name).strip().lower()
        text = re.sub(r"[^a-z]+", "_", text)
        return text.strip("_")

    investor_map = {
        "foreign_investor": "inst_foreign",
        "foreign": "inst_foreign",
        "investment_trust": "inst_investment_trust",
        "investmenttrust": "inst_investment_trust",
        "dealer_self": "inst_dealer_self",
        "dealerself": "inst_dealer_self",
        "dealer_hedging": "inst_dealer_hedging",
        "dealerhedging": "inst_dealer_hedging",
    }

    inst_df["inst_column"] = inst_df[investor_col].map(lambda value: investor_map.get(normalize_investor(value)))
    inst_df = inst_df.dropna(subset=["inst_column"])

    if inst_df.empty:
        LOGGER.warning("法人長表的投資人類別無法對應至標準欄位。")
        return empty_inst

    pivot = inst_df.pivot_table(
        index=[col for col in ["date", "stock_id"] if col in inst_df.columns],
        columns="inst_column",
        values="value",
        aggfunc="sum",
    )
    pivot = pivot.reset_index()

    pivot.columns = [
        column if isinstance(column, str) else column[1]
        for column in pivot.columns
    ]

    for column in empty_inst.columns:
        if column not in pivot.columns:
            pivot[column] = np.nan

    return pivot[empty_inst.columns]


def _merge_daily_wide(price_df: pd.DataFrame, inst_df: pd.DataFrame) -> pd.DataFrame:
    """合併價量與法人資料，確保每天每檔僅一列。"""

    base_columns = ["date", "stock_id"]

    if price_df is None or price_df.empty:
        combined = inst_df.copy()
    elif inst_df is None or inst_df.empty:
        combined = price_df.copy()
    else:
        combined = pd.merge(price_df, inst_df, on=base_columns, how="outer")

    if combined.empty:
        return combined

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=False)
        try:
            combined["date"] = combined["date"].dt.tz_localize(None)
        except AttributeError:
            pass

    agg_map = {}
    for column in combined.columns:
        if column in base_columns:
            continue
        if column.startswith("inst_"):
            agg_map[column] = "sum"
        else:
            agg_map[column] = "first"

    if agg_map:
        combined = combined.groupby(base_columns, as_index=False).agg(agg_map)
    else:
        combined = combined.drop_duplicates(subset=base_columns)

    for column in [
        "inst_foreign",
        "inst_investment_trust",
        "inst_dealer_self",
        "inst_dealer_hedging",
    ]:
        if column not in combined.columns:
            combined[column] = np.nan

    return combined


def _print_summary(df: pd.DataFrame) -> None:
    """在終端機輸出摘要資訊。"""

    print("=== 清理後資料摘要 ===")
    print(f"總筆數: {len(df):,}")

    if "stock_id" in df.columns:
        print(f"股票檔數: {df['stock_id'].nunique():,}")
    else:
        print("股票檔數: 無 stock_id 欄位")

    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
        if date_series.notna().any():
            min_date = date_series.min().strftime("%Y-%m-%d")
            max_date = date_series.max().strftime("%Y-%m-%d")
            print(f"日期範圍: {min_date} ~ {max_date}")
        else:
            print("日期範圍: 無有效日期")
    else:
        print("日期範圍: 無 date 欄位")

    print("-- 欄位缺值比例 --")
    nan_ratio = df.isna().mean()
    for column, ratio in nan_ratio.items():
        print(f"{column}: {ratio:.2%}")

    if df.empty:
        print("資料為空，無樣本可顯示。")
        return

    sample_is_head = np.random.rand() < 0.5
    label = "前 3 筆" if sample_is_head else "後 3 筆"
    sample = df.head(3) if sample_is_head else df.tail(3)
    print(f"-- 隨機樣本 ({label}) --")
    print(sample.to_string(index=False))


def parse_arguments() -> argparse.Namespace:
    """解析指令列參數。"""

    today = dt.date.today()
    one_year_ago = today - dt.timedelta(days=365)

    parser = argparse.ArgumentParser(description="FinMind 日資料批次下載工具")
    parser.add_argument("--token", default="", help="FinMind token，可留空")
    parser.add_argument(
        "--stocks",
        default="1519,2379,2383,2454,3035,3293,6231,6643,8358,8932,2344,2308,3535",
        help="股票代號清單，以逗號分隔",
    )
    parser.add_argument(
        "--start",
        default=one_year_ago.isoformat(),
        help="起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=today.isoformat(),
        help="結束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--datasets",
        default="TaiwanStockPrice,TaiwanStockInstitutionalInvestorsBuySell",
        help="欲下載的資料集，以逗號分隔",
    )
    parser.add_argument(
        "--outdir",
        default="./finmind_out",
        help="輸出資料夾",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="是否同時輸出 Parquet",
    )
    parser.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        default=True,
        help="是否輸出合併寬表 (預設開啟)",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="停用合併寬表輸出",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=3.0,
        help="遭遇速率限制時的初始等待秒數",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="同一請求的最大重試次數",
    )
    return parser.parse_args()


def main() -> None:
    """程式進入點。"""

    args = parse_arguments()
    configure_logging(args.outdir)

    token = args.token.strip() or None
    if token:
        LOGGER.info("使用提供的 token 以提升速率限制。")
    else:
        LOGGER.warning("未提供 token，每小時可用額度較低，建議提供 token。")

    stocks = [code.strip() for code in args.stocks.split(",") if code.strip()]
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    session = build_session()

    aggregated_frames: Dict[str, List[pd.DataFrame]] = {name: [] for name in datasets}

    for dataset in datasets:
        if dataset not in DATASET_CATALOG:
            LOGGER.error("資料集 %s 未在 DATASET_CATALOG 中定義，跳過。", dataset)
            continue

        for stock in stocks:
            LOGGER.info(
                "開始下載 dataset=%s, stock_id=%s, 區間=%s~%s",
                dataset,
                stock,
                args.start,
                args.end,
            )
            requires_stock = DATASET_CATALOG[dataset].get("requires_stock", True)
            stock_id = stock if requires_stock else None
            try:
                raw_df = fetch_dataset(
                    session=session,
                    dataset=dataset,
                    stock_id=stock_id,
                    start=args.start,
                    end=args.end,
                    token=token,
                    rate_limit_sleep=args.rate_limit_sleep,
                    retries=args.retries,
                )
            except FetchError as exc:
                ERROR_LOGGER.error(
                    "%s", exc,
                    exc_info=False,
                )
                continue

            normalizer = NORMALIZERS.get(dataset)
            if not normalizer:
                LOGGER.error("資料集 %s 缺少對應的清洗函式。", dataset)
                continue

            cleaned_df = normalizer(raw_df)
            if cleaned_df.empty:
                LOGGER.warning("dataset=%s, stock_id=%s 無資料。", dataset, stock)

            aggregated_frames.setdefault(dataset, []).append(cleaned_df)

            dataset_dir = os.path.join(args.outdir, dataset)
            tag = DATASET_FILENAME_TAG.get(dataset, dataset.lower())
            csv_path = os.path.join(dataset_dir, f"{stock}_{tag}.csv")
            parquet_path = (
                os.path.join(dataset_dir, f"{stock}_{tag}.parquet")
                if args.parquet
                else None
            )
            save_frame(cleaned_df, csv_path, parquet_path)

    dataset_frames: Dict[str, pd.DataFrame] = {}
    for dataset, frames in aggregated_frames.items():
        if not frames:
            dataset_frames[dataset] = pd.DataFrame()
            continue
        dataset_frames[dataset] = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["stock_id", "date"])
            .reset_index(drop=True)
        )

    if args.merge:
        LOGGER.info("開始合併所有資料集。")
        merged_df = merge_frames(dataset_frames)
        merged_csv = os.path.join(args.outdir, "_merged.csv")
        merged_parquet = (
            os.path.join(args.outdir, "_merged.parquet") if args.parquet else None
        )
        save_frame(merged_df, merged_csv, merged_parquet)
    else:
        LOGGER.info("使用者設定不輸出合併寬表。")

    external_merged_path = "/mnt/data/_merged.csv"
    if os.path.exists(external_merged_path):
        LOGGER.info("偵測到 %s，開始整理每日寬表。", external_merged_path)
        raw_external = _read_raw_merged(external_merged_path)
        if raw_external.empty:
            LOGGER.warning("外部合併檔案無資料，跳過每日寬表清理。")
        else:
            normalized = _normalize_types(raw_external)
            price_block = _extract_price_block(normalized)
            inst_block = _build_institutional_wide(normalized)
            merged_daily = _merge_daily_wide(price_block, inst_block)
            if merged_daily.empty:
                LOGGER.warning("合併後資料為空，無法輸出每日寬表。")
            else:
                merged_daily = merged_daily.sort_values(["date", "stock_id"]).reset_index(drop=True)

                display_df = merged_daily.copy()
                if "date" in display_df.columns:
                    date_series = pd.to_datetime(display_df["date"], errors="coerce")
                    display_df["date"] = date_series.dt.strftime("%Y-%m-%d")
                _print_summary(display_df)

                output_path = "/mnt/data/_clean_daily_wide.csv"
                output_min_path = "/mnt/data/_clean_daily_wide_min.csv"

                export_df = merged_daily.copy()
                if "date" in export_df.columns:
                    date_series = pd.to_datetime(export_df["date"], errors="coerce")
                    export_df["date"] = date_series.dt.strftime("%Y-%m-%d")

                required_min_columns = [
                    "date",
                    "stock_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                    "inst_foreign",
                    "inst_investment_trust",
                    "inst_dealer_self",
                    "inst_dealer_hedging",
                ]
                for column in required_min_columns:
                    if column not in export_df.columns:
                        export_df[column] = np.nan

                export_df.to_csv(output_path, index=False, encoding="utf-8")
                export_df[required_min_columns].to_csv(
                    output_min_path, index=False, encoding="utf-8"
                )
                LOGGER.info("每日寬表已輸出至 %s 與 %s。", output_path, output_min_path)
    else:
        LOGGER.info("未偵測到 %s，跳過每日寬表清理。", external_merged_path)


if __name__ == "__main__":
    main()
