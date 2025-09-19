"""指令列介面與流程控制。"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DATASETS,
    DEFAULT_MERGE,
    DEFAULT_OUTDIR,
    DEFAULT_RATE_LIMIT_SLEEP,
    DEFAULT_RETRIES,
    DEFAULT_STOCKS,
    ERROR_LOGGER,
    LOGGER,
    configure_logging,
    default_end_date,
    default_start_date,
)
from .datasets import DATASET_CATALOG, DATASET_FILENAME_TAG
from .fetcher import FetchError, build_session, fetch_dataset
from .io_utils import _read_raw_merged, save_frame
from .merger import (
    _build_institutional_wide,
    _extract_price_block,
    _merge_daily_wide,
    _normalize_types,
    merge_frames,
)
from .normalizers import NORMALIZERS
from .summarize import _print_summary


def parse_arguments() -> argparse.Namespace:
    """解析指令列參數。"""

    parser = argparse.ArgumentParser(description="FinMind 日資料批次下載工具")
    parser.add_argument("--token", default="", help="FinMind token，可留空")
    parser.add_argument(
        "--stocks",
        default=DEFAULT_STOCKS,
        help="股票代號清單，以逗號分隔",
    )
    parser.add_argument(
        "--start",
        default=default_start_date(),
        help="起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=default_end_date(),
        help="結束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help="欲下載的資料集，以逗號分隔",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
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
        default=DEFAULT_MERGE,
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
        default=DEFAULT_RATE_LIMIT_SLEEP,
        help="遭遇速率限制時的初始等待秒數",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
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

    external_merged_path = os.path.join(args.outdir, "_merged.csv")
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

                output_path = os.path.join(args.outdir, "_clean_daily_wide.csv")
                output_min_path = os.path.join(args.outdir, "_clean_daily_wide_min.csv")


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


__all__ = ["parse_arguments", "main"]
