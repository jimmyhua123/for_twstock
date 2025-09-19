"""FinMind ETL 組態設定與共用常數。"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys

API_URL = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_TIMEOUT = 20
DEFAULT_STOCKS = "1519,2379,2383,2454,3035,3293,6231,6643,8358,8932,2344,2308,3535"
DEFAULT_DATASETS = "TaiwanStockPrice,TaiwanStockInstitutionalInvestorsBuySell"
DEFAULT_OUTDIR = "./finmind_out"
DEFAULT_RATE_LIMIT_SLEEP = 3.0
DEFAULT_RETRIES = 3
DEFAULT_MERGE = True

LOGGER = logging.getLogger("finmind")
ERROR_LOGGER = logging.getLogger("finmind.errors")


def default_start_date() -> str:
    """取得預設起始日期（今日往前一年）。"""

    today = dt.date.today()
    one_year_ago = today - dt.timedelta(days=365)
    return one_year_ago.isoformat()


def default_end_date() -> str:
    """取得預設結束日期（今日）。"""

    return dt.date.today().isoformat()


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


__all__ = [
    "API_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_STOCKS",
    "DEFAULT_DATASETS",
    "DEFAULT_OUTDIR",
    "DEFAULT_RATE_LIMIT_SLEEP",
    "DEFAULT_RETRIES",
    "DEFAULT_MERGE",
    "LOGGER",
    "ERROR_LOGGER",
    "configure_logging",
    "default_start_date",
    "default_end_date",
]
