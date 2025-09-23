

1) 先看 scores_watchlist.csv：排序 + 分級

打開後先照 score_total 由高到低排序。用這張表就能做日常決策面板。

推薦分級規則（先用，後面再微調）

加碼/關注（候選）
score_total ≥ 70 且 score_tech ≥ 60 且 score_chip ≥ 60 並且 excess_ret_20d > 0
（技術走強 + 籌碼進場 + 相對大盤勝出）

觀望/回檔等買點
score_fund ≥ 70 但 score_tech 40–60（基本面佳、技術面尚在轉強）

減碼/風險
score_risk ≤ 40 或 score_total ≤ 40（波動升溫、總評偏弱）

小提醒：這些門檻先拿「你這 13 檔」做相對排序即可；若改抓全市場，再用全體分位數調門檻。

快速判讀欄位（從左到右看）

ret_5d / ret_20d、excess_ret_20d：短中期動能與相對表現

chip_net5_sum / chip_net20_sum：三大法人短/中期淨流入（>0 佳，越大越好）

foreign_ratio_5d_chg：外資持股比變化（>0 代表外資布局）

revenue_yoy（有就看）：年增轉正或加速加分

pe/pb（有就看）：估值低於同業＝更有吸引力

score_*：四象限分數；score_total 為綜合排名

在 Excel 裡加條件格式：

excess_ret_20d > 0、chip_net5_sum > 0 標綠；score_total ≥ 70 標粗體/綠底。

score_risk ≤ 40 標紅；ret_20d < 0 與 chip_net20_sum < 0 標淡紅。

2) 產業視角：誰整體在轉強

用 scores_watchlist.csv 做樞紐表：

列：industry_category

值：score_total 平均、score_tech 平均、檔數（count）

排序：score_total 由高到低

這給你兩個決策：

今天/這週先看分數高的產業裡的高分股

對分數低、但你長期看好的產業，等待「籌碼或技術」回暖再介入

3) scores_breakdown.csv：查分數來源

當某檔 score_total 高，但你想知道「為什麼高」：

找到它的 score_tech_* / score_chip_* / score_fund_* / score_risk_* 欄位

定位主因：例如 score_chip_chip_net20_sum 很高＝中期資金明顯流入；score_fund_pe 高＝估值在同業中偏便宜

如果 score_tech_excess_ret_20d 很高但 score_chip_* 普通，代表技術先行、籌碼未跟；可列為「需觀察續航」

這張表就是檢核/復盤時的依據：你能把每次挑出的名單對照拆分分數，看是否符合原本的邏輯。

4) features_snapshot.csv：加做「監測」與「異常」

把它當作欄位大全，做兩件事：

A. 每日變化（Δ）

在 Excel 加兩欄（手動或用 Power Query）：
Δscore_total = 今天score_total - 昨天score_total
Δrank = 昨天排名 - 今天排名（名次躍升者是訊號）

依 Δscore_total 或 Δrank 由大到小排序，找今天剛轉強的

B. 風險篩子

rolling_vol_20d、（有的話）gap%、vol_mult_20d

設定簡單風險門檻：

rolling_vol_20d > 你群組的 80 分位 → 高波動名單

vol_mult_20d > 2 且 excess_ret_20d < 0 → 放量下跌，列入觀察/排除







### 零超限版流程：粗篩用 TWSE/TPEx，精算才用 FinMind

**A. 由 TaiwanStockInfo 產生全市場名單（只做一次）**
```powershell
python .\tools\make_universe_all.py --input finmind_raw\TaiwanStockInfo.json --out finmind_in --batch-size 200
```

**B. 建立全市場粗篩 features（只用官方來源，近一年）**

```powershell
$since = (Get-Date).AddDays(-90).ToString('yyyy-MM-dd')
$until = (Get-Date).ToString('yyyy-MM-dd')
python -m finmind_etl build-coarse `
  --universe finmind_in\universe_all.csv `
  --since $since `
  --until $until `
  --out-features finmind_scores\features_snapshot_$($until.Replace('-',''))`.csv `
  --sleep-ms 250
```

**C. 粗篩報告（profile=coarse）**

```powershell
python -m finmind_etl scan-market `
  --features finmind_scores\features_snapshot_$($until.Replace('-',''))`.csv `
  --profile coarse `
  --output  finmind_reports\market_scan
```

**D. 取 Top N 生成 watchlist.csv**

```powershell
python - <<'PY'
import pandas as pd
df=pd.read_csv(r"finmind_reports/market_scan/market_scan_scores.csv").dropna(subset=["score_total"])
df.sort_values("score_total", ascending=False).head(50)[["stock_id"]].to_csv("watchlist.csv", index=False, encoding="utf-8")
print("OK -> watchlist.csv")
PY
```

## 精算階段：fetch-fine（含配額追蹤與整點續跑）

`fetch-fine` 專抓 **fine profile** 需要的 FinMind 低頻重欄位（營收、三表、外資持股、融資券…），
並追蹤 600/hr（可改）配額：達上限會寫入 `finmind_raw/_quota/finmind_quota.json` 的 `resume_at`，
下個整點再執行即可「無縫續跑」。

> 需要先把環境變數 `FINMIND_TOKEN` 設好。

### 指令
```powershell
$since = (Get-Date).AddDays(-800).ToString('yyyy-MM-dd')  # 三表TTM/2年
$until = (Get-Date).ToString('yyyy-MM-dd')
python -m finmind_etl fetch-fine \
  --watchlist .\watchlist.csv \
  --since $since \
  --until $until \
  --outdir finmind_raw \
  --sleep-ms 900 \
  --limit-per-hour 600 \
  --max-requests 550
```

### 觀察與續跑

* 每次請求會更新 `finmind_raw/_quota/finmind_quota.json`：

  ```json
  {
    "per_hour": {"2025-09-24T14:00": 548},
    "last": "2025-09-24T14:53:10",
    "hit_limit": true,
    "resume_at": "2025-09-24T15:00:00",
    "used_this_hour": 600,
    "limit_per_hour": 600,
    "dataset": "TaiwanStockShareholding",
    "last_stock": "2454"
  }
  ```
* 若看到 `hit_limit: true`，等到 `resume_at` 後再執行同一條指令即可「接著抓」。
* 由於 raw 以 `(dataset/stock_id/期間)` 落檔，已完成的請求會被跳過，不會重抓。

### 與零超限流程整合

* **粗篩**：`build-coarse` 產出全市場 `features_snapshot_YYYYMMDD.csv`（不使用 FinMind）。
* **取 Top N**：生成 `watchlist.csv`。
* **精算**：執行 `fetch-fine`（可多次、跨小時續跑），完成後再跑：

  ```powershell
  python .\finmind_clean_standardize.py --raw-dir finmind_raw --out-dir finmind_out
  python .\finmind_features_scoring.py --clean-dir finmind_out --raw-dir finmind_raw --out-dir finmind_scores --full-daily
  python -m finmind_etl report-watchlist --features finmind_scores\features_snapshot_YYYYMMDD.csv --watchlist .\watchlist.csv --profile fine --output finmind_reports\watchlist_deep
  ```

**E. 精算（只對 Top N 用 FinMind 抓重欄位 → 清理 → 特徵 → 報告）**

```powershell
# 1) 針對 watchlist 抓 FinMind（低頻重欄位），依上方 fetch-fine 指令（可跨整點續跑）
python -m finmind_etl fetch-fine \
  --watchlist .\watchlist.csv \
  --since $since \
  --until $until \
  --outdir finmind_raw \
  --sleep-ms 900 \
  --limit-per-hour 600 \
  --max-requests 550

# 2) 清理 + 特徵（會把 fund/chip/risk 填齊）
python .\finmind_clean_standardize.py --raw-dir finmind_raw --out-dir finmind_out
python .\finmind_features_scoring.py --clean-dir finmind_out --raw-dir finmind_raw --out-dir finmind_scores --full-daily

# 3) 自選精算報告（profile=fine）
python -m finmind_etl report-watchlist `
  --features  finmind_scores\features_snapshot_$($until.Replace('-',''))`.csv `
  --watchlist .\watchlist.csv `
  --profile   fine `
  --output    finmind_reports\watchlist_deep
```

> 備註：粗篩/精算都會輸出 `_diag_missing_features.csv`，可以快速看到每一面向缺欄與缺值比。

---
