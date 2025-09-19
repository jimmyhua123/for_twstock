"""根據各面向分數給出投資建議。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import get_logger
from .chips import ChipSummary
from .fundamentals import FundamentalSummary
from .utils import clip_score

LOGGER = get_logger(__name__)

# -- 調整參數區 -------------------------------------------------------------
TREND_VOLUME_THRESHOLD = 1.5
TREND_WEAK_VOLUME_THRESHOLD = 0.8
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 30

RECOMMENDATION_RULES: dict[str, float] = {
    "BULLISH": 13.0,
    "ACCUMULATE": 10.0,
    "NEUTRAL": 7.0,
}


@dataclass
class RecommendationResult:
    """投資建議輸出資料。"""

    total_score: float
    recommendation: str
    rationale: str
    trend_class: str
    chip_trend_note: str
    risk_flags: str


def aggregate_scores(row: pd.Series, fund_score: float) -> float:
    total = float(row.get("tech_score", 0))
    total += float(row.get("chip_score", 0))
    total += float(row.get("capital_score", 0))
    total += float(fund_score)
    return clip_score(total, 0.0, 18.0)


def classify_trend(row: pd.Series) -> str:
    ma20 = row.get("MA20")
    ma60 = row.get("MA60")
    slope20 = row.get("MA20_slope")
    slope60 = row.get("MA60_slope")
    close = row.get("close")

    if pd.notna(close) and pd.notna(ma20) and pd.notna(ma60):
        if close > ma20 > ma60 and (slope20 or 0) > 0 and (slope60 or 0) >= 0:
            return "Up"
        if close < ma20 < ma60 and (slope20 or 0) < 0:
            return "Down"
    return "Sideways"


def build_chip_note(chip_summary: ChipSummary) -> str:
    note_parts = []
    if chip_summary.consecutive_buy:
        note_parts.append(f"近連買 {chip_summary.consecutive_buy} 日")
    if chip_summary.consecutive_sell:
        note_parts.append(f"近連賣 {chip_summary.consecutive_sell} 日")
    if chip_summary.turning_point:
        note_parts.append(f"轉折日 {chip_summary.turning_point}")
    if chip_summary.note:
        note_parts.append(chip_summary.note)
    return "；".join(note_parts)


def detect_risk_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    rsi14 = row.get("RSI14")
    if pd.notna(rsi14) and rsi14 >= RSI_OVERBOUGHT:
        flags.append("RSI 高檔過熱")
    if pd.notna(rsi14) and rsi14 <= RSI_OVERSOLD:
        flags.append("RSI 接近超賣")
    volume_ratio = row.get("VOL_MA20")
    if pd.notna(volume_ratio) and volume_ratio != 0:
        vol_ratio = float(row.get("volume", 0)) / float(volume_ratio)
        if vol_ratio >= TREND_VOLUME_THRESHOLD * 2:
            flags.append("爆量需留意籌碼消化")
        if vol_ratio <= 0.5:
            flags.append("量能急凍")
    if pd.notna(row.get("MACD")) and pd.notna(row.get("DEA")):
        macd = row.get("MACD")
        dea = row.get("DEA")
        if macd < 0 and dea > 0:
            flags.append("MACD 轉弱")
    return flags


def build_rationale(row: pd.Series, chip_summary: ChipSummary, fund_summary: FundamentalSummary) -> str:
    points: list[str] = []
    close = row.get("close")
    ma20 = row.get("MA20")
    ma60 = row.get("MA60")
    if pd.notna(close) and pd.notna(ma20) and pd.notna(ma60):
        if close > ma20 > ma60:
            points.append("均線多頭排列")
        elif close < ma20 < ma60:
            points.append("均線空頭排列")

    if pd.notna(row.get("RSI14")):
        rsi = row.get("RSI14")
        if rsi < 30:
            points.append("RSI 進入超賣區有反彈契機")
        elif rsi > 70:
            points.append("RSI 過熱需謹慎")

    volume_ratio = None
    if pd.notna(row.get("VOL_MA20")) and row.get("VOL_MA20"):
        volume_ratio = float(row.get("volume", 0)) / float(row.get("VOL_MA20"))
        if volume_ratio >= TREND_VOLUME_THRESHOLD:
            points.append("量能放大支撐趨勢")
        elif volume_ratio <= TREND_WEAK_VOLUME_THRESHOLD:
            points.append("量能萎縮趨勢易震盪")

    if row.get("net_all_10", 0) > 0:
        points.append("近 10 日法人合計買超")
    elif row.get("net_all_10", 0) < 0:
        points.append("近 10 日法人合計賣超")

    if fund_summary.score == 0:
        points.append("基本面資料不足，建議另行查證")
    else:
        points.append(f"基本面評分 {fund_summary.score:.1f}，{fund_summary.note}")

    chip_note = build_chip_note(chip_summary)
    if chip_note:
        points.append(chip_note)

    return "；".join(points)


def determine_recommendation(total_score: float) -> str:
    if total_score >= RECOMMENDATION_RULES["BULLISH"]:
        return "偏多/關注買進"
    if total_score >= RECOMMENDATION_RULES["ACCUMULATE"]:
        return "區間偏多/逢回布局"
    if total_score >= RECOMMENDATION_RULES["NEUTRAL"]:
        return "中性觀望/區間操作"
    return "偏空/迴避"


def make_recommendation(
    row: pd.Series,
    chip_summary: ChipSummary,
    fund_summary: FundamentalSummary,
) -> RecommendationResult:
    """整合所有分數產出建議。"""

    total_score = aggregate_scores(row, fund_summary.score)
    trend = classify_trend(row)
    rationale = build_rationale(row, chip_summary, fund_summary)
    risk = detect_risk_flags(row)
    recommendation = determine_recommendation(total_score)

    return RecommendationResult(
        total_score=total_score,
        recommendation=recommendation,
        rationale=rationale,
        trend_class=trend,
        chip_trend_note=build_chip_note(chip_summary),
        risk_flags="、".join(risk) if risk else "",
    )


if __name__ == "__main__":  # pragma: no cover - 自測
    sample = pd.Series(
        {
            "close": 600,
            "MA20": 580,
            "MA60": 550,
            "MA20_slope": 2,
            "MA60_slope": 1,
            "RSI14": 65,
            "VOL_MA20": 1000,
            "volume": 1500,
            "tech_score": 4,
            "chip_score": 3,
            "capital_score": 2,
            "net_all_10": 5000,
        }
    )
    chip_summary = ChipSummary(3, 2, 0, None, "外資小幅買超")
    fund_summary = FundamentalSummary(4, "月營收年增率穩定")
    result = make_recommendation(sample, chip_summary, fund_summary)
    print("recommender 自測完成", result)
