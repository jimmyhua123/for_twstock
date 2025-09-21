# -*- coding: utf-8 -*-
"""
FinMind Free 台股資料批次抓取（RAW JSON, 可選 CSV）
- Token 走 Authorization: Bearer（.env: FINMIND_TOKEN）
- 排除：
  * 5秒資料與新聞：TaiwanVariousIndicators5Seconds / TaiwanStockStatisticsOfOrderBookAndTrade / TaiwanStockNews
  * 有會員等級限制：TaiwanExchangeRate / InterestRate / CrudeOilPrices / GovernmentBondsYield
  * 衍生性金融商品與期/選即時
- 分流：
  1) NO_DATE_NO_ID: 不帶日期、不帶 data_id
  2) DATE_ONLY    : 只帶日期
  3) NEED_ID      : 要 data_id（個股）
  4) TOTAL_RETURN : 要 data_id（"TAIEX","TPEx"）
- 預設抓過去一年，可用 --since / --until 覆寫
- 預設股票清單 = 你的清單；可用 --all-market 先抓全市場（較久）
- 只在成功且「有資料」時才存檔；5xx 具簡單重試
"""
import os
import json
import time
import argparse
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# 可選：CSV 轉檔（需要 pandas）
try:
    import pandas as pd  # noqa: F401
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# -------------------------------
# 環境變數與 API 設定
# -------------------------------
load_dotenv()
API_TOKEN = os.getenv("FINMIND_TOKEN")
if not API_TOKEN:
    raise SystemExit("請在 .env 設定 FINMIND_TOKEN=你的token")

API_URL = "https://api.finmindtrade.com/api/v4/data"
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}

# -------------------------------
# 資料集清單（依你要求移除不抓者）
# -------------------------------

# 1) 完全不吃日期、不需 data_id
NO_DATE_NO_ID = [
    "TaiwanStockInfo",
    "TaiwanStockInfoWithWarrant",
    "TaiwanSecuritiesTraderInfo",
]

# 2) 只吃日期、不需 data_id（整體或全市場彙整）
DATE_ONLY = [
    "TaiwanStockTradingDate",
    "TaiwanStockTotalMarginPurchaseShortSale",
    "TaiwanStockTotalInstitutionalInvestors",
    "GoldPrice",
]

# 3) 需要 data_id（通常是個股 stock_id）
NEED_ID = [
    # 技術面
    "TaiwanStockPrice",
    "TaiwanStockPER",
    "TaiwanStockDayTrading",
    # 籌碼面（個股）
    "TaiwanStockMarginPurchaseShortSale",
    "TaiwanStockInstitutionalInvestorsBuySell",
    "TaiwanStockShareholding",
    "TaiwanStockSecuritiesLending",
    "TaiwanStockMarginShortSaleSuspension",
    "TaiwanDailyShortSaleBalances",
    # 基本面（個股）
    "TaiwanStockFinancialStatements",
    "TaiwanStockBalanceSheet",
    "TaiwanStockCashFlowsStatement",
    "TaiwanStockDividend",
    "TaiwanStockDividendResult",
    "TaiwanStockMonthRevenue",
    "TaiwanStockCapitalReductionReferencePrice",
    "TaiwanStockDelisting",
    "TaiwanStockSplitPrice",
    "TaiwanStockParValueChange",
]

# 4) 報酬指數（需 data_id，但不是個股）：TAIEX / TPEx
TOTAL_RETURN_DATASET = "TaiwanStockTotalReturnIndex"
TOTAL_RETURN_IDS = ["TAIEX", "TPEx"]

# 永遠排除（依你要求）
EXCLUDE_ALWAYS = {
    # 5 秒資料 + 新聞
    "TaiwanVariousIndicators5Seconds",
    "TaiwanStockStatisticsOfOrderBookAndTrade",
    "TaiwanStockNews",
    # 會員等級限制
    "TaiwanExchangeRate",
    "InterestRate",
    "CrudeOilPrices",
    "GovernmentBondsYield",
    # 衍生性商品 & 即時
    "TaiwanFutOptDailyInfo",
    "TaiwanFuturesDaily",
    "TaiwanOptionDaily",
    "TaiwanFuturesInstitutionalInvestors",
    "TaiwanOptionInstitutionalInvestors",
    "TaiwanFuturesDealerTradingVolumeDaily",
    "TaiwanOptionDealerTradingVolumeDaily",
    "TaiwanFutOptTickInfo",
}

# 預設股票（你的清單）
DEFAULT_STOCK_IDS = [
    "1519", 
    # "2379", "2383", "2454", "3035", "3293", "6231", "6643", "8358", "8932",
    # "2344", "2308", "3535",
]

# -------------------------------
# 小工具：輸出與存檔
# -------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"[完成] 已存檔 {path}")

def write_csv_from_payload(json_payload: Dict[str, Any], csv_path: str) -> bool:
    if not HAVE_PANDAS:
        log("[CSV] 未安裝 pandas，略過轉檔（pip install pandas）")
        return False
    data = json_payload.get("data", [])
    if not data:
        log(f"[CSV] {csv_path} 無資料略過")
        return False
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log(f"[CSV] 已輸出 {csv_path}")
    return True

def safe_text(resp: requests.Response) -> str:
    try:
        return resp.text[:400]
    except Exception:
        return ""

# -------------------------------
# HTTP with retry
# -------------------------------
def http_get(params: Dict[str, Any], timeout: int = 60, retries: int = 2, backoff: float = 0.8) -> Optional[Dict[str, Any]]:
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException as e:
            if attempt > retries:
                log(f"[連線錯誤] {params.get('dataset')} -> {e}")
                return None
            time.sleep(backoff * attempt)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except Exception as e:
                log(f"[解析失敗] {params.get('dataset')} -> {e}")
                return None

        # 429/5xx 重試
        if r.status_code in (429, 500, 502, 503, 504) and attempt <= retries:
            time.sleep(backoff * attempt)
            continue

        log(f"[警告] 抓取 {params.get('dataset')} 失敗，HTTP {r.status_code}，訊息：{safe_text(r)}")
        return None

# -------------------------------
# Token 檢查（顯示 level/配額）
# -------------------------------
def check_token() -> Dict[str, Any]:
    url = "https://api.web.finmindtrade.com/v2/user_info"
    out = {"ok": False, "level": None, "limit_hour": None}
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        log(f"[user_info] {r.status_code} {r.text[:200]}")
        if r.status_code == 200:
            data = r.json()
            out["ok"] = True
            out["level"] = data.get("level")
            out["limit_hour"] = data.get("api_request_limit_hour")
    except requests.RequestException as e:
        log(f"[user_info 連線錯誤] {e}")
    return out

# -------------------------------
# 取得全市場 stock_id
# -------------------------------
def fetch_all_stock_ids() -> List[str]:
    params = {"dataset": "TaiwanStockInfo"}
    data = http_get(params)
    ids: List[str] = []
    if data and "data" in data:
        ids = sorted({row.get("stock_id") for row in data["data"] if row.get("stock_id")})
        log(f"[全市場] 取得 {len(ids)} 檔 stock_id")
    else:
        log("[全市場] 取得 stock_id 失敗，改用預設清單")
    return ids

# -------------------------------
# 三類 + 報酬指數 抓取
# -------------------------------
def fetch_no_date_no_id(outdir: str, to_csv: bool, sleep_sec: float = 0.0) -> None:
    for ds in NO_DATE_NO_ID:
        if ds in EXCLUDE_ALWAYS:
            continue
        params = {"dataset": ds}
        data = http_get(params)
        if data and "data" in data and len(data["data"]) > 0:
            json_path = os.path.join(outdir, f"{ds}.json")
            save_json(json_path, data)
            if to_csv:
                write_csv_from_payload(data, os.path.join(outdir, f"{ds}.csv"))
        else:
            log(f"[略過] {ds} 無資料或失敗，不存檔")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

def fetch_date_only(start_date: str, end_date: str, outdir: str, to_csv: bool, sleep_sec: float = 0.0) -> None:
    for ds in DATE_ONLY:
        if ds in EXCLUDE_ALWAYS:
            continue
        params = {"dataset": ds, "start_date": start_date, "end_date": end_date}
        data = http_get(params)
        if data and "data" in data and len(data["data"]) > 0:
            json_path = os.path.join(outdir, f"{ds}.json")
            save_json(json_path, data)
            if to_csv:
                write_csv_from_payload(data, os.path.join(outdir, f"{ds}.csv"))
        else:
            log(f"[略過] {ds} 無資料或失敗，不存檔")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

def fetch_need_id(stock_ids: List[str], start_date: str, end_date: str, outdir: str, to_csv: bool, sleep_sec: float = 0.0) -> None:
    for ds in NEED_ID:
        if ds in EXCLUDE_ALWAYS:
            continue
        all_rows: List[Dict[str, Any]] = []
        ok_count = 0
        for sid in stock_ids:
            params = {"dataset": ds, "data_id": sid, "start_date": start_date, "end_date": end_date}
            data = http_get(params)
            if data and "data" in data and len(data["data"]) > 0:
                all_rows.extend(data["data"])
                ok_count += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        if len(all_rows) > 0:
            payload = {"dataset": ds, "start_date": start_date, "end_date": end_date, "count": len(all_rows), "data": all_rows}
            json_path = os.path.join(outdir, f"{ds}.json")
            save_json(json_path, payload)
            if to_csv:
                write_csv_from_payload(payload, os.path.join(outdir, f"{ds}.csv"))
            log(f"  ↳ {ds}: 有資料 {ok_count}/{len(stock_ids)} 檔（彙整筆數 {len(all_rows)}）")
        else:
            log(f"[略過] {ds}: 指定 stock_id 皆無資料或失敗，不存檔")

def fetch_total_return(start_date: str, end_date: str, outdir: str, to_csv: bool, sleep_sec: float = 0.0) -> None:
    """TaiwanStockTotalReturnIndex 需 data_id：TAIEX / TPEx"""
    ds = TOTAL_RETURN_DATASET
    all_rows: List[Dict[str, Any]] = []
    ok_count = 0
    for rid in TOTAL_RETURN_IDS:
        params = {"dataset": ds, "data_id": rid, "start_date": start_date, "end_date": end_date}
        data = http_get(params)
        if data and "data" in data and len(data["data"]) > 0:
            all_rows.extend(data["data"])
            ok_count += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if len(all_rows) > 0:
        payload = {"dataset": ds, "start_date": start_date, "end_date": end_date, "count": len(all_rows), "data": all_rows}
        json_path = os.path.join(outdir, f"{ds}.json")
        save_json(json_path, payload)
        if to_csv:
            write_csv_from_payload(payload, os.path.join(outdir, f"{ds}.csv"))
        log(f"  ↳ {ds}: 有資料 {ok_count}/{len(TOTAL_RETURN_IDS)} 指數（彙整筆數 {len(all_rows)}）")
    else:
        log(f"[略過] {ds}: 無資料或失敗，不存檔")

# -------------------------------
# 參數解析
# -------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FinMind Free 台股資料批次抓取（RAW JSON, 可選 CSV）")
    p.add_argument("--since", type=str, default=None, help="起始日 YYYY-MM-DD（預設=今天往前一年）")
    p.add_argument("--until", type=str, default=None, help="結束日 YYYY-MM-DD（預設=今天）")
    p.add_argument("--outdir", type=str, default="finmind_raw", help="輸出資料夾（預設 finmind_raw）")
    p.add_argument("--sleep", type=float, default=0.0, help="每次請求之間的延遲秒數（預設 0）")
    p.add_argument("--all-market", action="store_true", help="改抓全市場所有 stock_id（會較久）")
    p.add_argument("--ids", type=str, default=None, help="自訂股票清單，逗號分隔，例如 2330,2317,2303")
    p.add_argument("--ids-file", type=str, default=None, help="自訂股票清單檔案，每行一個代碼")
    p.add_argument("--to-csv", action="store_true", help="同時輸出 CSV（需要 pandas）")
    p.add_argument("--no-check", action="store_true", help="啟動時不要打 user_info 檢查 token")
    return p.parse_args()

def parse_date(s: Optional[str], default: date) -> date:
    if not s:
        return default
    return datetime.strptime(s, "%Y-%m-%d").date()

def load_ids_from_file(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                ids.append(t)
    return ids

# -------------------------------
# main
# -------------------------------
def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    # 日期區間：預設過去一年
    today = date.today()
    start_default = today - timedelta(days=365)
    start_d = parse_date(args.since, start_default)
    end_d = parse_date(args.until, today)

    start_str = start_d.isoformat()
    end_str = end_d.isoformat()

    # 檢查 Token（可關閉）
    if not args.no_check:
        info = check_token()
        if not info.get("ok"):
            log("Token 檢查未通過，請確認 .env 的 FINMIND_TOKEN 是否正確（無引號、無多餘空白）。")
            return

    # 準備 stock_id 清單
    if args.all_market:
        stock_ids = fetch_all_stock_ids()
        if not stock_ids:
            stock_ids = DEFAULT_STOCK_IDS[:]
    elif args.ids_file:
        stock_ids = load_ids_from_file(args.ids_file)
    elif args.ids:
        stock_ids = [t.strip() for t in args.ids.split(",") if t.strip()]
    else:
        stock_ids = DEFAULT_STOCK_IDS[:]

    log(f"開始抓取 FinMind（{start_str} ~ {end_str}），輸出：{args.outdir}")
    log(f"個股清單共 {len(stock_ids)} 檔（預覽前 10 檔）：{stock_ids[:10]}")

    # 1) 不吃日期、不需 data_id
    fetch_no_date_no_id(outdir=args.outdir, to_csv=args.to_csv, sleep_sec=args.sleep)

    # 2) 只吃日期、不需 data_id
    fetch_date_only(start_date=start_str, end_date=end_str, outdir=args.outdir, to_csv=args.to_csv, sleep_sec=args.sleep)

    # 3) 報酬指數（需要 data_id：TAIEX / TPEx）
    fetch_total_return(start_date=start_str, end_date=end_str, outdir=args.outdir, to_csv=args.to_csv, sleep_sec=args.sleep)

    # 4) 需要 data_id（個股）
    fetch_need_id(stock_ids=stock_ids, start_date=start_str, end_date=end_str, outdir=args.outdir, to_csv=args.to_csv, sleep_sec=args.sleep)

    log("全部完成。")

if __name__ == "__main__":
    main()
