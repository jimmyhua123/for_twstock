"""允許透過 ``python -m finmind_fetch`` 執行寬表擴充流程。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from . import get_logger
from .enrich import EnrichConfig, enrich_clean_daily

LOGGER = get_logger(__name__)


def _parse_stocks(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """解析 `finmind_fetch` 指令列參數。"""

    parser = argparse.ArgumentParser(description="FinMind 基本面與市場熱度整併工具")
    parser.add_argument("--input", required=True, help="輸入 _clean_daily_wide.csv 路徑")
    parser.add_argument("--out", help="輸出寬表路徑，預設覆蓋原檔")
    parser.add_argument("--min-output", help="精簡寬表路徑，可省略以使用預設命名")
    parser.add_argument("--since", help="僅抓取此日期之後所需的基本面資料 (YYYY-MM-DD)")
    parser.add_argument("--stocks", help="指定股票代號，逗號分隔")
    parser.add_argument("--fetch-fundamentals", action="store_true", help="是否下載基本面資料")
    parser.add_argument("--finmind-token", help="FinMind API token，未提供則讀 FINMIND_TOKEN")
    parser.add_argument("--force-refresh", action="store_true", help="忽略快取並強制重抓")
    parser.add_argument("--strict", action="store_true", help="缺資料時改為拋例外")
    parser.add_argument(
        "--align-strategy",
        choices=["forward_fill", "month_end"],
        default="forward_fill",
        help="月資料對齊策略，預設填滿整段至下一次公布",
    )
    parser.add_argument("--cache-dir", help="快取目錄，預設為 finmind_cache")
    parser.add_argument("--no-min", action="store_true", help="不輸出精簡版寬表")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error("找不到輸入檔案：%s", input_path)
        sys.exit(1)

    output_path = Path(args.out) if args.out else input_path
    min_output_path = None
    update_min = not args.no_min
    if update_min and args.min_output:
        min_output_path = Path(args.min_output)

    stocks = _parse_stocks(args.stocks)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    config = EnrichConfig(
        input_path=input_path,
        output_path=output_path,
        min_output_path=min_output_path,
        fetch_fundamentals=args.fetch_fundamentals,
        since=args.since,
        stocks=stocks,
        token=args.finmind_token or None,
        force_refresh=args.force_refresh,
        strict=args.strict,
        align_strategy=args.align_strategy,
        cache_dir=cache_dir,
        update_min=update_min,
    )

    result = enrich_clean_daily(config)
    LOGGER.info(
        "擴充完成，輸出列數=%s，股票數量=%s",
        len(result),
        result["stock_id"].nunique() if not result.empty else 0,
    )


if __name__ == "__main__":  # pragma: no cover - 直接透過 CLI 執行
    main()
