"""FinMind 資料下載指令列工具相容層。"""

from __future__ import annotations

from finmind_etl.cli import main, parse_arguments
from finmind_etl.config import *  # noqa: F401,F403
from finmind_etl.datasets import *  # noqa: F401,F403
from finmind_etl.fetcher import *  # noqa: F401,F403
from finmind_etl.io_utils import *  # noqa: F401,F403
from finmind_etl.merger import *  # noqa: F401,F403
from finmind_etl.normalizers import *  # noqa: F401,F403
from finmind_etl.summarize import *  # noqa: F401,F403

__all__ = [
    "main",
    "parse_arguments",
]


if __name__ == "__main__":
    main()
