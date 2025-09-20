"""指令列介面：串接 FinMind API 並輸出寬表。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .api import APIClient
from .chip import fetch_chip_data
from .derivative import fetch_derivative_data
from .enrich import build_daily_wide, build_minimal_view
from .fundamentals import fetch_fundamental_data
from .technical import fetch_technical_data

LOGGER = logging.getLogger("finmind_etl.cli")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """解析 CLI 參數。"""

    parser = argparse.ArgumentParser(description="FinMind v4 台股資料彙整工具")
    parser.add_argument("--tickers", required=True, help="股票代號，逗號分隔")
    parser.add_argument("--since", required=True, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--finmind-token", dest="token", help="FinMind API token")
    parser.add_argument("--outdir", default="./finmind_out", help="輸出資料夾")
    parser.add_argument("--end", help="結束日期，預設為今日")
    return parser.parse_args(argv)


def _parse_tickers(value: str) -> List[str]:
    tickers = [item.strip() for item in value.split(",") if item.strip()]
    return [ticker.zfill(4) for ticker in tickers]


def _snake_case(value: str) -> str:
    import re

    text = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    if "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("輸出 %s 筆資料至 %s", len(export), path)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_arguments(argv)
    configure_logging()

    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("請至少提供一檔股票代號")

    LOGGER.info("目標股票：%s", tickers)
    client = APIClient(token=args.token)

    technical = fetch_technical_data(tickers, args.since, client, end_date=args.end)
    fundamentals = fetch_fundamental_data(tickers, args.since, client, end_date=args.end)
    chip = fetch_chip_data(tickers, args.since, client, end_date=args.end)
    derivative = fetch_derivative_data(tickers, args.since, client, end_date=args.end)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_frames: Dict[str, pd.DataFrame] = {}
    raw_frames.update(technical)
    raw_frames.update(fundamentals)
    raw_frames.update(chip)
    raw_frames.update(derivative)

    for dataset, df in raw_frames.items():
        if df is None:
            continue
        safe_name = _snake_case(dataset)
        path = outdir / f"raw_{safe_name}.csv"
        _write_dataframe(df, path)

    daily_wide = build_daily_wide(tickers, technical, fundamentals, chip, derivative)
    if daily_wide.empty:
        LOGGER.warning("合併後資料為空，請檢查輸入參數或 API 回應。")
        return

    wide_path = outdir / "_clean_daily_wide.csv"
    _write_dataframe(daily_wide, wide_path)

    wide_min = build_minimal_view(daily_wide)
    min_path = outdir / "_clean_daily_wide_min.csv"
    _write_dataframe(wide_min, min_path)

    LOGGER.info(
        "完成：列數=%s 股票數=%s 日期範圍=%s~%s",
        len(daily_wide),
        daily_wide["stock_id"].nunique(),
        daily_wide["date"].min().strftime("%Y-%m-%d"),
        daily_wide["date"].max().strftime("%Y-%m-%d"),
    )


__all__ = ["main", "parse_arguments"]
