"""指令列介面：串接 FinMind API 並輸出寬表。"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .api import APIClient
from .chip import fetch_chip_data
from .fundamentals import fetch_fundamental_data
from .merge import build_minimal_view, merge_all
from .technical import fetch_technical_data
from .finmind_api import compute_ma_from_price, fetch_stock_price, try_fetch_10y_api

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
    parser.add_argument(
        "--use-10y-api",
        action="store_true",
        help="啟用 TaiwanStock10Year API（需 Sponsor 等級，失敗將自動回退）",
    )
    return parser.parse_args(argv)


def _parse_tickers(value: str) -> List[str]:
    tickers = [item.strip() for item in value.split(",") if item.strip()]
    return [ticker.zfill(4) for ticker in tickers]


def _resolve_token(cli_token: str | None) -> str | None:
    candidates = [cli_token, os.getenv("FINMIND_TOKEN")]
    for value in candidates:
        if value is None:
            continue
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    if "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("輸出 %s 筆資料至 %s", len(export), path)


def _select_price_frame(
    ticker: str,
    technical: Dict[str, "DatasetResult"],
    since: str,
    end: str | None,
    token: str | None,
) -> pd.DataFrame:
    from .datasets import DatasetResult  # 延遲匯入避免循環

    def _filter_and_prepare(result: DatasetResult | None, column: str) -> pd.DataFrame:
        if result is None or result.clean.empty:
            return pd.DataFrame()
        df = result.clean.copy()
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
        subset = df[df["stock_id"] == ticker]
        if subset.empty:
            return pd.DataFrame()
        if column in subset.columns:
            subset["close"] = pd.to_numeric(subset[column], errors="coerce")
        elif "close" in subset.columns:
            subset["close"] = pd.to_numeric(subset["close"], errors="coerce")
        else:
            subset["close"] = pd.NA
        return subset

    price_adj = technical.get("TaiwanStockPriceAdj")
    prepared = _filter_and_prepare(price_adj, "adj_close")
    if not prepared.empty:
        return prepared

    price = technical.get("TaiwanStockPrice")
    prepared = _filter_and_prepare(price, "close")
    if not prepared.empty:
        return prepared

    fallback = fetch_stock_price(ticker, since, end, token, use_adj=True)
    if fallback.empty:
        return fallback
    fallback["stock_id"] = fallback.get("stock_id", ticker)
    return fallback


def _build_ma10y_map(
    tickers: List[str],
    technical: Dict[str, "DatasetResult"],
    since: str,
    end: str | None,
    token: str | None,
    use_api: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for ticker in tickers:
        ma_frame: pd.DataFrame | None = None
        if use_api:
            ma_frame = try_fetch_10y_api(ticker, since, end, token)

        if ma_frame is None:
            price_frame = _select_price_frame(ticker, technical, since, end, token)
            if price_frame.empty:
                LOGGER.warning("Ticker %s 無法取得股價資料，MA10Y 欄位將為空。", ticker)
                continue
            ma_frame = compute_ma_from_price(price_frame, price_col="close")

        if ma_frame is None or ma_frame.empty:
            continue

        ma_frame = ma_frame.copy()
        ma_frame["stock_id"] = ma_frame.get("stock_id", ticker)
        ma_frame["stock_id"] = ma_frame["stock_id"].astype(str).str.strip()
        frames.append(ma_frame[["stock_id", "date", "MA10Y"]])

    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "MA10Y"])

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged = merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return merged.drop_duplicates(subset=["stock_id", "date"], keep="last")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_arguments(argv)
    configure_logging()

    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("請至少提供一檔股票代號")

    LOGGER.info("目標股票：%s", tickers)
    token = _resolve_token(args.token)
    if token is None and args.token and not args.token.strip():
        LOGGER.info("CLI 提供的 token 為空字串，將改用環境變數。")
    if token is None and os.getenv("FINMIND_TOKEN"):
        LOGGER.info("使用環境變數 FINMIND_TOKEN 中的 token。")
    client = APIClient(token=token)

    technical = fetch_technical_data(tickers, args.since, client, end_date=args.end)
    fundamentals = fetch_fundamental_data(tickers, args.since, client, end_date=args.end)
    chip = fetch_chip_data(tickers, args.since, client, end_date=args.end)
    ma10y_map = _build_ma10y_map(tickers, technical, args.since, args.end, token, args.use_10y_api)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for category in (technical, fundamentals, chip):
        for dataset, result in category.items():
            raw_path = outdir / f"raw_{dataset}.csv"
            _write_dataframe(result.raw, raw_path)

    daily_wide = merge_all(tickers, technical, fundamentals, chip)
    if daily_wide.empty:
        LOGGER.warning("合併後資料為空，請檢查輸入參數或 API 回應。")
        return

    if ma10y_map.empty:
        daily_wide["MA10Y"] = pd.NA
    else:
        daily_wide = daily_wide.merge(ma10y_map, on=["stock_id", "date"], how="left")
        daily_wide = daily_wide.sort_values(["date", "stock_id"]).reset_index(drop=True)

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
