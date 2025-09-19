"""取得並評估基本面資料。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from . import get_logger
from .utils import clip_score

LOGGER = get_logger(__name__)
FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


@dataclass
class FundamentalSummary:
    """封裝基本面分析結果。"""

    score: float
    note: str
    revenue_yoy_avg: float | None = None
    revenue_mom_avg: float | None = None
    eps_recent: float | None = None


class FundamentalAnalyzer:
    """基本面資料彙整器。"""

    def __init__(
        self,
        fundamentals_dir: str | Path | None = None,
        finmind_token: str | None = None,
    ) -> None:
        self.fundamentals_dir = Path(fundamentals_dir) if fundamentals_dir else None
        self.finmind_token = finmind_token or ""
        self._revenue_df: Optional[pd.DataFrame] = None
        self._income_df: Optional[pd.DataFrame] = None
        if self.fundamentals_dir and not self.fundamentals_dir.exists():
            LOGGER.warning("指定的基本面目錄不存在：%s", self.fundamentals_dir)

    def analyze(self, stock_id: str) -> FundamentalSummary:
        """針對單一股票計算基本面評分。"""

        revenue_df = self._get_revenue_data(stock_id)
        income_df = self._get_income_data(stock_id)

        if revenue_df is None and income_df is None:
            return FundamentalSummary(score=0.0, note="基本面資料不足")

        yoy_avg, mom_avg = self._evaluate_revenue(revenue_df) if revenue_df is not None else (None, None)
        eps_recent = self._evaluate_income(income_df) if income_df is not None else None

        score = 1.5
        note_parts: list[str] = []

        if yoy_avg is not None:
            if yoy_avg >= 0.2:
                score += 1.5
                note_parts.append("月營收年增率長期維持兩成以上")
            elif yoy_avg >= 0.05:
                score += 1.0
                note_parts.append("月營收年增率保持成長")
            elif yoy_avg < 0:
                score -= 1.0
                note_parts.append("月營收年增率走弱")
        else:
            note_parts.append("缺少足夠的年增率資料")

        if mom_avg is not None:
            if mom_avg >= 0.05:
                score += 0.5
                note_parts.append("近期月營收月增率轉強")
            elif mom_avg <= -0.05:
                score -= 0.5
                note_parts.append("近期月營收月增率轉弱")
        else:
            note_parts.append("月增率資料不足")

        if eps_recent is not None:
            if eps_recent >= 2:
                score += 1.0
                note_parts.append("最近年度 EPS 顯著且穩健")
            elif eps_recent > 0:
                score += 0.5
                note_parts.append("最近年度維持正獲利")
            else:
                score -= 1.0
                note_parts.append("近期出現虧損")
        else:
            note_parts.append("未取得 EPS 或淨利資料")

        score = clip_score(score, 0.0, 5.0)
        note = "；".join(note_parts)

        return FundamentalSummary(
            score=score,
            note=note,
            revenue_yoy_avg=yoy_avg,
            revenue_mom_avg=mom_avg,
            eps_recent=eps_recent,
        )

    # -- 資料載入 ------------------------------------------------------------

    def _get_revenue_data(self, stock_id: str) -> Optional[pd.DataFrame]:
        if self._revenue_df is None:
            self._revenue_df = self._load_local_csv("month_revenue")
        df = self._revenue_df
        if df is not None and not df.empty:
            subset = df[df["stock_id"].astype(str) == stock_id].copy()
            if subset.empty:
                return None
            subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
            subset = subset.dropna(subset=["date"]).sort_values("date")
            return subset

        if not self.finmind_token:
            return None

        try:
            df = self._fetch_finmind("TaiwanStockMonthRevenue", stock_id)
        except URLError as exc:  # pragma: no cover - 網路存取非測試重點
            LOGGER.error("取得 FinMind 月營收資料失敗：%s", exc)
            return None
        if df is None or df.empty:
            return None
        df.rename(columns={"revenue": "value"}, inplace=True)
        df = df.rename(columns={"value": "revenue"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df

    def _get_income_data(self, stock_id: str) -> Optional[pd.DataFrame]:
        if self._income_df is None:
            self._income_df = self._load_local_csv("income_statement")
        df = self._income_df
        if df is not None and not df.empty:
            subset = df[df["stock_id"].astype(str) == stock_id].copy()
            if subset.empty:
                return None
            subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
            subset = subset.dropna(subset=["date"]).sort_values("date")
            return subset

        if not self.finmind_token:
            return None
        try:
            df = self._fetch_finmind("TaiwanStockFinancialStatements", stock_id)
        except URLError as exc:  # pragma: no cover - 網路存取非測試重點
            LOGGER.error("取得 FinMind 財報資料失敗：%s", exc)
            return None
        if df is None or df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df

    def _load_local_csv(self, keyword: str) -> Optional[pd.DataFrame]:
        if not self.fundamentals_dir:
            return None
        candidates = sorted(self.fundamentals_dir.glob(f"{keyword}*.csv"))
        if not candidates:
            return None
        path = candidates[0]
        LOGGER.info("載入基本面檔案：%s", path)
        df = pd.read_csv(path)
        return df

    def _fetch_finmind(self, dataset: str, stock_id: str) -> Optional[pd.DataFrame]:
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "token": self.finmind_token,
        }
        query = urlencode(params)
        request = Request(f"{FINMIND_API}?{query}")
        with urlopen(request, timeout=30) as response:  # noqa: S310 - 外部 API 存取
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != 200:
            LOGGER.error("FinMind API 回應錯誤：%s", payload.get("msg"))
            return None
        data = payload.get("data", [])
        return pd.DataFrame(data)

    # -- 評估邏輯 ------------------------------------------------------------

    def _evaluate_revenue(self, df: pd.DataFrame) -> tuple[float | None, float | None]:
        if df is None or df.empty:
            return None, None
        if "revenue" not in df.columns:
            LOGGER.warning("月營收檔缺少 revenue 欄位，無法計算年增率")
            return None, None
        df = df.sort_values("date")
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        df = df.dropna(subset=["revenue"])
        if len(df) < 12:
            LOGGER.warning("月營收資料不足 12 個月")
            return None, None

        if "yoy" in df.columns:
            df["yoy"] = pd.to_numeric(df["yoy"], errors="coerce") / 100
        else:
            df["yoy"] = df["revenue"].pct_change(periods=12)

        if "mom" in df.columns:
            df["mom"] = pd.to_numeric(df["mom"], errors="coerce") / 100
        else:
            df["mom"] = df["revenue"].pct_change(periods=1)

        recent = df.iloc[-12:]
        yoy_avg = float(recent["yoy"].mean()) if not recent["yoy"].isna().all() else None
        mom_recent = df.iloc[-6:]
        mom_avg = float(mom_recent["mom"].mean()) if not mom_recent["mom"].isna().all() else None
        return yoy_avg, mom_avg

    def _evaluate_income(self, df: pd.DataFrame) -> float | None:
        if df is None or df.empty:
            return None
        df = df.sort_values("date")
        eps_columns = [col for col in df.columns if col.lower() in {"eps", "eps_after_tax"}]
        if eps_columns:
            df[eps_columns[0]] = pd.to_numeric(df[eps_columns[0]], errors="coerce")
            return float(df[eps_columns[0]].tail(4).mean())

        net_income_columns = [col for col in df.columns if "net" in col.lower() and "income" in col.lower()]
        if net_income_columns:
            df[net_income_columns[0]] = pd.to_numeric(df[net_income_columns[0]], errors="coerce")
            return float(df[net_income_columns[0]].tail(4).mean())
        LOGGER.warning("財報檔案缺少 EPS 或淨利欄位")
        return None


if __name__ == "__main__":  # pragma: no cover - 自測
    analyzer = FundamentalAnalyzer()
    summary = analyzer.analyze("2330")
    print("fundamentals 自測完成", summary)
