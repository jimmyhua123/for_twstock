"""分析模組封裝。"""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = [
    "get_logger",
]


def get_logger(name: str) -> logging.Logger:
    """以統一格式建立日誌記錄器。

    參數
    ----
    name:
        模組名稱。

    返回
    ----
    logging.Logger
        已設定好層級與輸出格式的記錄器。
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


PACKAGE_ROOT = Path(__file__).resolve().parent
