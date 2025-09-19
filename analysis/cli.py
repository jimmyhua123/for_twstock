"""指令列介面，串聯整體分析流程。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from . import get_logger
from .capital import calculate_capital_metrics
from .chips import ChipSummary, calculate_chip_metrics, summarize_chip
from .data_loader import load_wide_csv
from .fundamentals import FundamentalAnalyzer, FundamentalSummary
from .indicators import calculate_indicators
from .recommender import RecommendationResult, make_recommendation
from .report import StockReport, generate_reports
from .utils import ensure_directory, format_date

LOGGER = get_logger(__name__)


@dataclass
class StockAnalysis:
    """封裝單一股票分析結果。"""

    stock_id: str
    data: pd.DataFrame
    latest: pd.Series
    chip_summary: ChipSummary
    fund_summary: "FundamentalSummary"
    recommendation: RecommendationResult


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """解析 CLI 參數。"""

    parser = argparse.ArgumentParser(description="台股四大面向分析工具")
    parser.add_argument("--input", required=True, help="_clean_daily_wide.csv 路徑")
    parser.add_argument("--outdir", default="outputs", help="輸出資料夾")
    parser.add_argument("--since", help="僅分析此日期（含）之後資料")
    parser.add_argument("--stocks", help="限定分析之股票代號，逗號分隔")
    parser.add_argument("--with-charts", action="store_true", help="輸出圖表 PNG")
    parser.add_argument("--html-report", action="store_true", help="輸出 HTML 彙總")
    parser.add_argument("--finmind-token", help="FinMind API token")
    parser.add_argument(
        "--fundamentals-dir",
        help="基本面 CSV 所在資料夾，例如 month_revenue.csv",
    )
    return parser.parse_args(argv)


def _filter_data(df: pd.DataFrame, since: str | None, stocks: List[str] | None) -> pd.DataFrame:
    filtered = df
    if since:
        try:
            since_ts = pd.Timestamp(since).tz_localize("Asia/Taipei")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"--since 參數格式錯誤: {since}") from exc
        filtered = filtered[filtered["date"] >= since_ts]
    if stocks:
        stocks = [s.strip().zfill(4) for s in stocks]
        filtered = filtered[filtered["stock_id"].isin(stocks)]
    return filtered


def _prepare_fundamental_analyzer(args: argparse.Namespace) -> FundamentalAnalyzer:
    return FundamentalAnalyzer(
        fundamentals_dir=args.fundamentals_dir,
        finmind_token=args.finmind_token or None,
    )


def run_analysis(args: argparse.Namespace) -> list[StockAnalysis]:
    df = load_wide_csv(args.input)

    stocks_list = args.stocks.split(",") if args.stocks else None
    df = _filter_data(df, args.since, stocks_list)

    if df.empty:
        LOGGER.error("套用條件後無資料，請確認輸入範圍")
        sys.exit(1)

    df = calculate_capital_metrics(df)

    analyzer = _prepare_fundamental_analyzer(args)
    results: list[StockAnalysis] = []

    warnings: list[str] = []

    for stock_id, group in df.groupby("stock_id"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < 20:
            warnings.append(f"股票 {stock_id} 資料不足 20 筆，部分指標為空值")
        group = calculate_indicators(group)
        group = calculate_chip_metrics(group)

        latest = group.iloc[-1]
        chip_summary = summarize_chip(group)
        fund_summary = analyzer.analyze(stock_id)
        if fund_summary.score == 0 and "不足" in fund_summary.note:
            warnings.append(f"股票 {stock_id} 基本面資料不足")
        recommendation = make_recommendation(latest, chip_summary, fund_summary)

        results.append(
            StockAnalysis(
                stock_id=stock_id,
                data=group,
                latest=latest,
                chip_summary=chip_summary,
                fund_summary=fund_summary,
                recommendation=recommendation,
            )
        )

    results.sort(key=lambda item: item.recommendation.total_score, reverse=True)

    outdir = ensure_directory(args.outdir)
    analysis_date = format_date(df["date"].max())

    summary_rows = [_build_summary_row(res) for res in results]
    warnings = list(dict.fromkeys(warnings))
    summary_df = pd.DataFrame(summary_rows)

    generate_reports(
        summary_df=summary_df,
        stock_reports=[
            StockReport(
                stock_id=res.stock_id,
                latest=res.latest,
                chip_summary=res.chip_summary,
                fund_summary=res.fund_summary,
                recommendation=res.recommendation,
                data=res.data,
            )
            for res in results
        ],
        outdir=Path(outdir),
        analysis_date=analysis_date,
        with_charts=args.with_charts,
        html_report=args.html_report,
        warnings=warnings,
    )

    _print_console_summary(args, df, results, warnings)

    return results


def _build_summary_row(result: StockAnalysis) -> dict[str, object]:
    latest = result.latest
    vol_ma20 = latest.get("VOL_MA20")
    vol_ratio = float(latest.get("volume", 0)) / vol_ma20 if vol_ma20 else float("nan")
    return {
        "stock_id": result.stock_id,
        "date_last": format_date(latest.get("date")),
        "close": latest.get("close"),
        "pct_5d": latest.get("pct_5d"),
        "pct_10d": latest.get("pct_10d"),
        "pct_20d": latest.get("pct_20d"),
        "MA20": latest.get("MA20"),
        "MA60": latest.get("MA60"),
        "MA20_slope": latest.get("MA20_slope"),
        "MA60_slope": latest.get("MA60_slope"),
        "VOL_VOL_MA20_ratio": vol_ratio,
        "turnover_rank_pct": latest.get("turnover_rank_pct"),
        "turnover_change_5d": latest.get("turnover_change_5d"),
        "net_foreign_5": latest.get("net_foreign_5"),
        "net_it_5": latest.get("net_it_5"),
        "net_dealer_total_5": latest.get("net_dealer_total_5"),
        "net_all_5": latest.get("net_all_5"),
        "tech_score": latest.get("tech_score"),
        "chip_score": latest.get("chip_score"),
        "capital_score": latest.get("capital_score"),
        "fund_score": result.fund_summary.score,
        "total_score": result.recommendation.total_score,
        "trend_class": result.recommendation.trend_class,
        "chip_trend_note": result.recommendation.chip_trend_note,
        "rationale": result.recommendation.rationale,
        "recommendation": result.recommendation.recommendation,
        "risk_flags": result.recommendation.risk_flags,
    }


def _print_console_summary(
    args: argparse.Namespace,
    df: pd.DataFrame,
    results: list[StockAnalysis],
    warnings: list[str],
) -> None:
    first_date = format_date(df["date"].min())
    last_date = format_date(df["date"].max())
    print(f"資料範圍：{first_date} ~ {last_date}")
    print(f"股票數量：{len(results)}")
    print(f"輸出目錄：{Path(args.outdir).resolve()}")

    print("\n總分前 5 名：")
    for item in results[:5]:
        print(
            f"  {item.stock_id} - {item.recommendation.total_score:.2f} 分 - {item.recommendation.recommendation}"
        )

    print("\n總分後 5 名：")
    for item in results[-5:]:
        print(
            f"  {item.stock_id} - {item.recommendation.total_score:.2f} 分 - {item.recommendation.recommendation}"
        )

    if warnings:
        print("\n警告：")
        for message in warnings:
            print(f"- {message}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":  # pragma: no cover - 自測
    if len(sys.argv) > 1:
        main()
    else:
        print("請使用 --input 指定資料檔案")
