# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json, random
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import requests

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}

# ---------------- Quota & resume state ----------------
@dataclass
class QuotaConfig:
    limit_per_hour: int = 600
    state_file: str = "finmind_raw/_quota/finmind_quota.json"

class QuotaState:
    def __init__(self, cfg: QuotaConfig):
        self.cfg = cfg
        self.path = Path(cfg.state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"per_hour": {}, "last": None}

    def _hour_key(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%dT%H:00")

    def used_this_hour(self, now: datetime) -> int:
        key = self._hour_key(now)
        return int(self.state.get("per_hour", {}).get(key, 0))

    def can_request(self, n: int, now: datetime) -> bool:
        return self.used_this_hour(now) + n <= self.cfg.limit_per_hour

    def record(self, n: int, now: datetime):
        key = self._hour_key(now)
        per = self.state.setdefault("per_hour", {})
        per[key] = int(per.get(key, 0)) + n
        self.state["last"] = datetime.now().isoformat(timespec="seconds")
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def next_resume_at(self, now: datetime) -> datetime:
        # 整點視窗：下個整點開始
        base = now.replace(minute=0, second=0, microsecond=0)
        return base + timedelta(hours=1)

# ---------------- FinMind client ----------------
class FinMind:
    def __init__(self, token: str, sleep_ms: int = 800):
        self.token = token
        self.sleep_ms = sleep_ms
        self.s = requests.Session()
        self.s.headers.update(UA)

    def _sleep(self):
        jitter = random.randint(0, self.sleep_ms // 2)
        time.sleep((self.sleep_ms + jitter) / 1000.0)

    def get(self, dataset: str, data_id: str, start: str, end: str, extra: Dict = None) -> pd.DataFrame:
        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start,
            "end_date": end,
            "token": self.token,
        }
        if extra:
            params.update(extra)
        for _ in range(5):
            try:
                r = self.s.get(FINMIND_API, params=params, timeout=30)
                if r.status_code == 200:
                    js = r.json()
                    # FinMind v4: {"data": [...], "msg": "success"}
                    data = js.get("data", [])
                    if not data:
                        return pd.DataFrame()
                    return pd.DataFrame(data)
            except requests.exceptions.RequestException:
                pass
            self._sleep()
        return pd.DataFrame()

# ---------------- Fetch plan (fine profile) ----------------
@dataclass
class FetchPlan:
    # dataset, window_days, enabled
    name: str
    window_days: int
    enabled: bool = True

# 預設抓取計畫（只抓四大面向精算需要）
DEFAULT_PLAN: List[FetchPlan] = [
    FetchPlan("TaiwanStockMonthRevenue", 400),      # 13個月≈400天
    FetchPlan("TaiwanStockBalanceSheet", 800),      # 8季≈2年
    FetchPlan("TaiwanStockFinancialStatements", 800),
    FetchPlan("TaiwanStockCashFlowsStatement", 800),
    FetchPlan("TaiwanStockShareholding", 100),      # 外資持股 ~90–100天
    FetchPlan("TaiwanStockMarginPurchaseShortSale", 80),
    # 借券明細（如需）
    # FetchPlan("TaiwanStockBorrowingBalance", 80, enabled=False),
    # FetchPlan("TaiwanStockSecuritiesLending", 80, enabled=False),
]

# ---------------- Runner ----------------

def _mk_window(until: str, days: int) -> Tuple[str, str]:
    e = datetime.strptime(until, "%Y-%m-%d")
    s = e - timedelta(days=days)
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


def run_fetch_fine(
    watchlist_csv: str,
    since: str,
    until: str,
    outdir: str,
    sleep_ms: int = 800,
    limit_per_hour: int = 600,
    max_requests: int = 500,
    state_file: str = "finmind_raw/_quota/finmind_quota.json",
    datasets: List[str] | None = None,
) -> Dict:
    token = os.getenv("FINMIND_TOKEN")
    if not token:
        raise SystemExit("環境變數 FINMIND_TOKEN 未設置。請先設定你的 FinMind token。")

    wl = pd.read_csv(watchlist_csv)
    ids = [str(x) for x in wl["stock_id"].astype(str).tolist()]

    # 決定抓取計畫
    plan = [p for p in DEFAULT_PLAN if p.enabled]
    if datasets:
        allow = set(datasets)
        plan = [p for p in plan if p.name in allow]

    client = FinMind(token, sleep_ms=sleep_ms)
    quota = QuotaState(QuotaConfig(limit_per_hour=limit_per_hour, state_file=state_file))

    # outdir/dataset/*.csv
    out_root = Path(outdir); out_root.mkdir(parents=True, exist_ok=True)

    used = 0
    started_at = datetime.now()

    def already_have(dataset: str, sid: str, s: str, e: str) -> bool:
        ddir = out_root / dataset
        ddir.mkdir(exist_ok=True, parents=True)
        fname = ddir / f"{sid}_{s}_to_{e}.csv"
        return fname.exists()

    def save_raw(dataset: str, sid: str, s: str, e: str, df: pd.DataFrame):
        ddir = out_root / dataset
        ddir.mkdir(exist_ok=True, parents=True)
        fname = ddir / f"{sid}_{s}_to_{e}.csv"
        if not df.empty:
            df.to_csv(fname, index=False, encoding="utf-8")

    for p in plan:
        # 針對每個 dataset，決定實際視窗（取更短的那個）
        s_auto, e_auto = _mk_window(until, p.window_days)
        s = max(pd.to_datetime(since), pd.to_datetime(s_auto)).strftime("%Y-%m-%d")
        e = until
        for sid in ids:
            # 若已有相同視窗輸出，跳過（避免重覆請求）
            if already_have(p.name, sid, s, e):
                continue
            now = datetime.now()
            if not quota.can_request(1, now) or used >= max_requests:
                resume_at = quota.next_resume_at(now)
                # 寫入提示檔供外部腳本讀取
                tip = {
                    "hit_limit": True,
                    "resume_at": resume_at.isoformat(timespec="seconds"),
                    "used_this_hour": quota.used_this_hour(now),
                    "limit_per_hour": limit_per_hour,
                    "dataset": p.name,
                    "last_stock": sid,
                }
                Path(state_file).write_text(json.dumps({**quota.state, **tip}, ensure_ascii=False, indent=2), encoding="utf-8")
                print("[HIT LIMIT]", json.dumps(tip, ensure_ascii=False))
                return {"stopped": True, **tip}

            df = client.get(p.name, sid, s, e)
            quota.record(1, now)
            used += 1
            save_raw(p.name, sid, s, e, df)

    return {"stopped": False, "used": used, "started_at": started_at.isoformat(timespec='seconds'), "finished_at": datetime.now().isoformat(timespec='seconds')}
