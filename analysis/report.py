"""產生分析報告與圖表。"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import get_logger
from .chips import ChipSummary
from .fundamentals import FundamentalSummary
from .recommender import RecommendationResult
from .utils import ensure_directory, format_date

matplotlib.use("Agg")
LOGGER = get_logger(__name__)


@dataclass
class StockReport:
    """個股報告所需資料。"""

    stock_id: str
    latest: pd.Series
    chip_summary: ChipSummary
    fund_summary: FundamentalSummary
    recommendation: RecommendationResult
    data: pd.DataFrame


# -- 輔助 -------------------------------------------------------------------


def _ascii_histogram(values: Iterable[float], bins: int = 10) -> str:
    values = list(values)
    if not values:
        return "無資料"
    counts, edges = np.histogram(values, bins=bins, range=(0, 18))
    max_count = counts.max() or 1
    lines = []
    for idx, count in enumerate(counts):
        bar = "#" * int(count / max_count * 40)
        lines.append(f"{edges[idx]:4.1f}~{edges[idx+1]:4.1f}: {bar} ({count})")
    return "\n".join(lines)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directory(path.parent)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "（無資料）"
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, divider]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


# -- 圖表 -------------------------------------------------------------------


def _plot_price(df: pd.DataFrame, stock_id: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    price_ax, volume_ax, macd_ax, rsi_ax = axes

    price_ax.plot(df["date"], df["close"], label="收盤價", color="#1f77b4")
    for window in [5, 20, 60]:
        ma_col = f"MA{window}"
        if ma_col in df.columns:
            price_ax.plot(df["date"], df[ma_col], label=f"MA{window}")
    price_ax.set_ylabel("價格")
    price_ax.set_title(f"{stock_id} 價格與均線")
    price_ax.legend(loc="upper left")
    price_ax.grid(True, linestyle="--", alpha=0.3)

    volume_ax.bar(df["date"], df["volume"], color="#ff7f0e", label="成交量")
    if "VOL_MA20" in df.columns:
        volume_ax.plot(df["date"], df["VOL_MA20"], color="#2ca02c", label="20日均量")
    volume_ax.set_ylabel("成交量")
    volume_ax.legend(loc="upper left")

    macd_ax.plot(df["date"], df.get("DIF"), label="DIF")
    macd_ax.plot(df["date"], df.get("DEA"), label="DEA")
    macd_ax.bar(
        df["date"],
        df.get("MACD"),
        label="MACD",
        color=["#d62728" if val >= 0 else "#2ca02c" for val in df.get("MACD", pd.Series(0, index=df.index))],
        alpha=0.6,
    )
    macd_ax.set_ylabel("MACD")
    macd_ax.legend(loc="upper left")

    rsi_ax.plot(df["date"], df.get("RSI14"), label="RSI14", color="#9467bd")
    rsi_ax.axhline(70, color="red", linestyle="--", alpha=0.5)
    rsi_ax.axhline(30, color="green", linestyle="--", alpha=0.5)
    rsi_ax.set_ylabel("RSI")
    rsi_ax.set_xlabel("日期")
    rsi_ax.legend(loc="upper left")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return fig


def _plot_chip(df: pd.DataFrame, stock_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df.get("net_foreign_5"), label="外資 5 日累計")
    ax.plot(df["date"], df.get("net_it_5"), label="投信 5 日累計")
    ax.plot(df["date"], df.get("net_dealer_total_5"), label="自營商 5 日累計")
    ax.plot(df["date"], df.get("net_all_5"), label="三大法人合計 5 日")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{stock_id} 籌碼累計")
    ax.set_ylabel("單位：股數/張數")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


# -- 報告 -------------------------------------------------------------------


def generate_reports(
    summary_df: pd.DataFrame,
    stock_reports: list[StockReport],
    outdir: Path,
    analysis_date: str,
    with_charts: bool,
    html_report: bool,
    warnings: list[str] | None = None,
) -> None:
    """輸出彙總表、Markdown 及個股報告。"""

    ensure_directory(outdir)
    date_tag = analysis_date.replace("-", "")

    csv_path = outdir / f"recent_trend_chip_reco_{date_tag}.csv"
    summary_df.to_csv(csv_path, index=False)
    LOGGER.info("已輸出 CSV：%s", csv_path)

    summary_md_path = outdir / f"summary_{date_tag}.md"
    _write_summary_markdown(summary_md_path, summary_df, stock_reports, warnings)

    if html_report:
        html_path = outdir / f"summary_{date_tag}.html"
        _write_summary_html(html_path, summary_df)

    report_dir = outdir / "reports"
    chart_dir = outdir / "charts"

    for report in stock_reports:
        chart_paths: list[str] = []
        if with_charts:
            price_fig = _plot_price(report.data, report.stock_id)
            price_path = chart_dir / f"{report.stock_id}_{date_tag}_price.png"
            _save_figure(price_fig, price_path)
            chip_fig = _plot_chip(report.data, report.stock_id)
            chip_path = chart_dir / f"{report.stock_id}_{date_tag}_chip.png"
            _save_figure(chip_fig, chip_path)
            chart_paths = [price_path.name, chip_path.name]

        _write_stock_markdown(report_dir, report, chart_paths, analysis_date)


def _write_summary_markdown(
    path: Path,
    summary_df: pd.DataFrame,
    stock_reports: list[StockReport],
    warnings: list[str] | None,
) -> None:
    ensure_directory(path.parent)
    top20 = summary_df.nlargest(20, "total_score")
    bottom20 = summary_df.nsmallest(20, "total_score")

    histogram = _ascii_histogram(summary_df["total_score"], bins=8)

    buffer = io.StringIO()
    max_date = pd.to_datetime(summary_df["date_last"]).max()
    buffer.write(f"# 日度分析總結（{format_date(max_date)}）\n\n")
    buffer.write("## 方法論說明\n")
    buffer.write(
        "- 技術面：觀察均線排列、RSI、布林通道與量價關係。\n"
        "- 籌碼面：重視外資與投信連續性、三大法人合計趨勢。\n"
        "- 資金面：比較成交金額、成交量與成交筆數的升溫程度。\n"
        "- 基本面：以月營收 YoY/MoM 與 EPS 變化評估長期競爭力。\n\n"
    )

    if warnings:
        buffer.write("## 警告與限制\n")
        for warning in warnings:
            buffer.write(f"- {warning}\n")
        buffer.write("\n")

    buffer.write("## 分數分佈概覽\n")
    buffer.write("````text\n")
    buffer.write(histogram + "\n")
    buffer.write("````\n\n")

    buffer.write("## 總分前 20 名\n")
    buffer.write(_df_to_markdown(top20.reset_index(drop=True)) + "\n\n")

    buffer.write("## 總分後 20 名\n")
    buffer.write(_df_to_markdown(bottom20.reset_index(drop=True)) + "\n\n")

    buffer.write("## 指標解讀備忘\n")
    buffer.write(
        "- MACD 正值代表短期動能較強，DIF 向上突破 DEA 為黃金交叉。\n"
        "- RSI 30 以下視為超賣，70 以上可能過熱，需搭配趨勢判斷。\n"
        "- 布林帶上軌突破若無量能支撐，常出現拉回。\n"
        "- 量能放大（>20 日均量 1.5 倍）代表趨勢可信度提升。\n\n"
    )

    with path.open("w", encoding="utf-8") as fh:
        fh.write(buffer.getvalue())
    LOGGER.info("已輸出 Markdown 摘要：%s", path)


def _write_summary_html(path: Path, summary_df: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    table_html = summary_df.to_html(index=False)
    content = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<title>日度分析彙總</title>
<style>
body {{ font-family: "Noto Sans TC", sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: right; }}
th {{ background-color: #f5f5f5; }}
td:first-child, th:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>近期走勢與籌碼建議</h1>
{table_html}
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")
    LOGGER.info("已輸出 HTML 摘要：%s", path)


def _write_stock_markdown(
    report_dir: Path,
    report: StockReport,
    chart_paths: list[str],
    analysis_date: str,
) -> None:
    ensure_directory(report_dir)
    path = report_dir / f"{report.stock_id}_{analysis_date.replace('-', '')}.md"
    latest = report.latest

    buffer = io.StringIO()
    buffer.write(f"# {report.stock_id} 個股分析（{analysis_date}）\n\n")
    buffer.write("## 分數總覽\n")
    buffer.write(
        "| 面向 | 分數 |\n| --- | ---: |\n"
        f"| 技術面 | {latest.get('tech_score', float('nan')):.2f} |\n"
        f"| 籌碼面 | {latest.get('chip_score', float('nan')):.2f} |\n"
        f"| 資金面 | {latest.get('capital_score', float('nan')):.2f} |\n"
        f"| 基本面 | {report.fund_summary.score:.2f} |\n"
        f"| 總分 | {report.recommendation.total_score:.2f} |\n\n"
    )

    buffer.write("## 投資建議摘要\n")
    buffer.write(f"- 建議：{report.recommendation.recommendation}\n")
    buffer.write(f"- 趨勢分類：{report.recommendation.trend_class}\n")
    buffer.write(f"- 籌碼觀察：{report.recommendation.chip_trend_note}\n")
    if report.recommendation.risk_flags:
        buffer.write(f"- 風險提示：{report.recommendation.risk_flags}\n")
    buffer.write(f"- 理由整理：{report.recommendation.rationale}\n\n")

    buffer.write("## 技術面觀察\n")
    buffer.write(
        "- 均線與趨勢斜率：觀察 MA20 與 MA60 的方向，正斜率代表趨勢向上。\n"
        "- MACD：DIF 與 DEA 正交叉代表多方動能，負交叉需留意回檔。\n"
        "- RSI：70 以上過熱、30 以下過冷，搭配布林帶判斷反轉。\n\n"
    )

    buffer.write("## 籌碼面觀察\n")
    buffer.write(
        f"- 三大法人近 5 日合計：{latest.get('net_all_5', float('nan')):.0f}\n"
        f"- 外資近 5 日：{latest.get('net_foreign_5', float('nan')):.0f}\n"
        f"- 投信近 5 日：{latest.get('net_it_5', float('nan')):.0f}\n"
        f"- 自營商近 5 日：{latest.get('net_dealer_total_5', float('nan')):.0f}\n"
        f"- 籌碼摘要：{report.chip_summary.note}\n\n"
    )

    buffer.write("## 資金面觀察\n")
    buffer.write(
        f"- 成交金額分位數：{latest.get('turnover_rank_pct', float('nan')):.2f}\n"
        f"- 成交金額變化（5 日）：{latest.get('turnover_change_5d', float('nan')):.2%}\n"
        f"- 成交量變化（5 日）：{latest.get('volume_change_5d', float('nan')):.2%}\n"
        f"- 成交筆數變化（5 日）：{latest.get('transactions_change_5d', float('nan')):.2%}\n\n"
    )

    buffer.write("## 基本面觀察\n")
    buffer.write(f"- 評語：{report.fund_summary.note}\n")
    if report.fund_summary.revenue_yoy_avg is not None:
        buffer.write(f"- 近 12 個月月營收年增率平均：{report.fund_summary.revenue_yoy_avg:.2%}\n")
    if report.fund_summary.revenue_mom_avg is not None:
        buffer.write(f"- 近 6 個月月營收月增率平均：{report.fund_summary.revenue_mom_avg:.2%}\n")
    if report.fund_summary.eps_recent is not None:
        buffer.write(f"- 最近年度 EPS 平均：{report.fund_summary.eps_recent:.2f}\n")
    buffer.write("\n")

    if chart_paths:
        buffer.write("## 圖表\n")
        for chart in chart_paths:
            buffer.write(f"![圖表](../charts/{chart})\n")
        buffer.write("\n")

    buffer.write("## 指標說明\n")
    buffer.write(
        "- MA：簡單移動平均，反映趨勢方向。\n"
        "- EMA：指數平均，較重視近期價格。\n"
        "- MACD：動能指標，柱狀體由 DIF 與 DEA 差值乘以 2。\n"
        "- RSI：相對強弱指數，常用 14 日衡量超買超賣。\n"
        "- 布林帶：以 20 日均線為中心 ±2 倍標準差。\n"
        "- 量能：以 20 日均量為基礎衡量放量或縮量。\n"
        "- 法人累計：統計外資、投信、自營商近 5/10/20 日動向。\n"
    )

    path.write_text(buffer.getvalue(), encoding="utf-8")
    LOGGER.info("已輸出個股報告：%s", path)
