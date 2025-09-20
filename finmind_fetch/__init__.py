"""FinMind 基本面與市場熱度資料擴充模組。"""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = [
    "CACHE_DIR",
    "get_logger",
]

CACHE_DIR = Path("finmind_cache")


def get_logger(name: str) -> logging.Logger:
    """建立模組專用的 logger。"""

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
