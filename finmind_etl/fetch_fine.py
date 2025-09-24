# finmind_etl/fetch_fine.py
from __future__ import annotations
import os, time, json, math, pathlib, datetime as dt
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import requests

TZ = dt.timezone(dt.timedelta(hours=8))  # Asia/Taipei
API = "https://api.finmindtrade.com/api/v4/data"

DEFAULT_DATASETS = [
    # 基本面
    "TaiwanStockMonthRevenue",
    "TaiwanStockFinancialStatements",
    "TaiwanStockBalanceSheet",
    "TaiwanStockCashFlowsStatement",
    # 籌碼（精算）
    "TaiwanStockShareholding",
    "TaiwanStockMarginPurchaseShortSale",
    "TaiwanStockBorrowingBalance",
    # 輕籌碼（保險補齊）
    "TaiwanStockInstitutionalInvestorsBuySell",
]

# 各資料集需求的最短視窗（天）
MIN_DAYS = {
    "TaiwanStockMonthRevenue": 400,           # 13 個月 YoY
    "TaiwanStockFinancialStatements": 800,    # 8 季做 TTM
    "TaiwanStockBalanceSheet": 800,
    "TaiwanStockCashFlowsStatement": 800,
    "TaiwanStockShareholding": 180,
    "TaiwanStockMarginPurchaseShortSale": 180,
    "TaiwanStockBorrowingBalance": 180,
    "TaiwanStockBorrowingTransactions": 120,
    "TaiwanStockInstitutionalInvestorsBuySell": 90,
}

def _to_date(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s)[:10])

def _fmt(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def _ensure_outdir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _now() -> dt.datetime:
    return dt.datetime.now(TZ)

class QuotaState:
    def __init__(self, path: str, limit_per_hour: int):
        self.path = pathlib.Path(path)
        self.limit = limit_per_hour
        self.data = {"window_start": None, "used_in_window": 0, "resume_at": None, "limit_per_hour": self.limit}

    def load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset_if_needed(self):
        """若已跨整點，重置 window 與 used 計數"""
        now = _now()
        ws = self.data.get("window_start")
        if ws:
            ws_dt = dt.datetime.fromisoformat(ws)
        else:
            # 對齊至當前小時起點
            ws_dt = now.replace(minute=0, second=0, microsecond=0)
            self.data["window_start"] = ws_dt.isoformat()
            self.data["used_in_window"] = 0
            self.data["resume_at"] = None
            self.save()
            return
        if now >= ws_dt + dt.timedelta(hours=1):
            new_ws = now.replace(minute=0, second=0, microsecond=0)
            self.data["window_start"] = new_ws.isoformat()
            self.data["used_in_window"] = 0
            self.data["resume_at"] = None
            self.save()

    def can_use(self, need:int=1) -> bool:
        self.reset_if_needed()
        used = int(self.data.get("used_in_window", 0))
        return (used + need) <= int(self.data.get("limit_per_hour", self.limit))

    def mark_use(self, cnt:int=1):
        self.data["used_in_window"] = int(self.data.get("used_in_window", 0)) + cnt
        self.save()

    def stop_and_schedule_next_hour(self):
        # 設定 resume_at 在下一個整點
        now = _now()
        next_hour = (now + dt.timedelta(hours=1)).replace(minute=0, second=3, microsecond=0)
        self.data["resume_at"] = next_hour.isoformat()
        self.save()

def _effective_since(since: dt.date, until: dt.date, dataset: str) -> dt.date:
    days = MIN_DAYS.get(dataset, 180)
    cutoff = until - dt.timedelta(days=days)
    return max(since, cutoff)

def _build_params(dataset: str, stock_id: str, start: str, end: str, token: str) -> Dict[str, str]:
    # FinMind v4 大多數 dataset 用 data_id，但也接受 stock_id；兩者都傳最穩妥
    return {
        "dataset": dataset,
        "data_id": stock_id,
        "stock_id": stock_id,
        "start_date": start,
        "end_date": end,
        "token": token,
    }

def _target_path(outdir: str, dataset: str, stock_id: str, start: str, end: str) -> pathlib.Path:
    d = pathlib.Path(outdir) / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stock_id}__{start.replace('-','')}__{end.replace('-','')}.parquet"

def _already_done(p: pathlib.Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def _http_get(url: str, params: Dict[str,str], sleep_ms: int) -> Dict[str, Any]:
    # 保守 UA，輕量重試（讓上層配額負責節流）
    headers = {
        "User-Agent": "Mozilla/5.0 (fetch-fine/1.0)",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if sleep_ms > 0:
        time.sleep(sleep_ms/1000.0)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"status":"error","msg":"invalid json","data":[]}

def _save_df(df: pd.DataFrame, path: pathlib.Path):
    if df is None or df.empty:
        # 也落空檔，避免下次重抓
        path.write_bytes(b"")
        return
    df.to_parquet(path, index=False)

def run_fetch_fine(
    watchlist_csv: str,
    since: str,
    until: str,
    outdir: str,
    sleep_ms: int = 900,
    limit_per_hour: int = 600,
    max_requests: int = 550,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    token = os.environ.get("FINMIND_TOKEN")
    if not token:
        raise SystemExit("環境變數 FINMIND_TOKEN 未設定。請先：$env:FINMIND_TOKEN = '<你的 token>'")

    wl = pd.read_csv(watchlist_csv, dtype={"stock_id":str})
    ids = wl["stock_id"].astype(str).tolist()

    s_date = _to_date(since); u_date = _to_date(until)
    ds_list = datasets or DEFAULT_DATASETS

    # quota 狀態
    quota = QuotaState(path=os.path.join(outdir, "_quota", "finmind_quota.json"), limit_per_hour=limit_per_hour)
    quota.load()
    quota.reset_if_needed()

    # 若有 resume_at 未到，直接告知並結束
    ra = quota.data.get("resume_at")
    if ra:
        ra_dt = dt.datetime.fromisoformat(ra)
        if _now() < ra_dt:
            return {"stopped": True, "reason": f"quota reached; resume at {ra_dt.isoformat()}", "used": int(quota.data.get("used_in_window",0)),
                    "started_at": None, "finished_at": None}

    started = _now().isoformat()
    used_total = 0
    tasks: List[Tuple[str,str,str,str]] = []

    # 產生任務（每 dataset × 每 stock，一個時間窗）
    for dataset in ds_list:
        eff_since = _effective_since(s_date, u_date, dataset)
        start_str, end_str = _fmt(eff_since), _fmt(u_date)
        for sid in ids:
            tasks.append((dataset, sid, start_str, end_str))

    # 主迴圈
    for (dataset, sid, start_str, end_str) in tasks:
        # 避免重複：若同鍵檔已存在則跳過
        tgt = _target_path(outdir, dataset, sid, start_str, end_str)
        if _already_done(tgt):
            continue

        # 配額檢查
        if not quota.can_use(1) or (used_total >= max_requests):
            quota.stop_and_schedule_next_hour()
            return {"stopped": True, "reason": "quota or max_requests reached", "used": used_total,
                    "started_at": started, "finished_at": _now().isoformat()}

        # 執行請求
        try:
            params = _build_params(dataset, sid, start_str, end_str, token)
            js = _http_get(API, params=params, sleep_ms=sleep_ms)
            data = js.get("data", [])
            df = pd.DataFrame(data)
            _save_df(df, tgt)
        except Exception as e:
            # 落一個空檔，避免下一輪重複撞錯；錯誤留檔
            _save_df(pd.DataFrame(), tgt)
            errlog = pathlib.Path(outdir) / "_logs" / "fetch_fine_errors.log"
            errlog.parent.mkdir(parents=True, exist_ok=True)
            with errlog.open("a", encoding="utf-8") as f:
                f.write(f"{_now().isoformat()} dataset={dataset} sid={sid} {start_str}~{end_str} error={repr(e)}\n")

        # 計數與配額
        used_total += 1
        quota.mark_use(1)

    finished = _now().isoformat()
    # 全部跑完，清掉 resume_at
    quota.data["resume_at"] = None
    quota.save()
    return {"stopped": False, "used": used_total, "started_at": started, "finished_at": finished}
