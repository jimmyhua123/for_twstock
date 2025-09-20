

# 📖 FinMind 台股資料自動化流程

這個專案提供一套腳本與模組，協助你以 **FinMind API v4** 為來源，快速抓取、整理並擴充台股資料。輸出的資料可直接用於技術面、基本面、籌碼面與資金面等後續分析。

## ✨ 核心能力

* 一次抓回指定股票、時間區間與多個 dataset 的原始資料。
* 自動合併、清理欄位與日期格式，並將法人長表轉換成寬表。
* 產生完整欄位與精簡欄位兩種 CSV，方便不同分析情境使用。
* 透過額外的指令擴充基本面、財報與市場熱度指標。

## 🔁 工作流程

1. 從 FinMind API 抓取日資料（股價、法人買賣、融資融券、月營收等）。
2. 將原始資料整合為 `_merged.csv`。
3. 進行欄位統一、去重、缺值處理，並輸出：
   * `_clean_daily_wide.csv`：完整寬表資料。
   * `_clean_daily_wide_min.csv`：常用欄位精簡版。
4. （選用）再執行 `finmind_fetch` 模組，新增財報、月營收與市場熱度欄位。

## 📁 主要輸出檔案

| 檔名 | 內容重點 |
| --- | --- |
| `_merged.csv` | FinMind 回傳後的初步合併檔，日期＋股票代號可能有多筆資料。 |
| `_clean_daily_wide.csv` | 清理後的寬表，每個日期＋股票代號唯一一列，包含價量、法人與衍生欄位。 |
| `_clean_daily_wide_min.csv` | 精簡欄位版，保留 `date`, `stock_id`, `open/high/low/close`, `volume`, `turnover`, `inst_foreign`, `inst_investment_trust`, `inst_dealer_self`, `inst_dealer_hedging` 等常用指標。 |

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 建議使用 Python 3.12
pip install pandas requests pyarrow
```

### 2. 抓取資料

```bash
python main.py \
  --token <你的 FinMind token> \
  --stocks 2330,2317,2454 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --datasets TaiwanStockPrice,TaiwanStockInstitutionalInvestorsBuySell \
  --outdir ./output \
  --merge
```

完成後會生成下列檔案：

* `output/_merged.csv`
* `output/_clean_daily_wide.csv`
* `output/_clean_daily_wide_min.csv`

### 3. 擴充基本面與市場熱度欄位（選用）

```bash
python -m finmind_fetch \
  --input finmind_out/_clean_daily_wide.csv \
  --fetch-fundamentals \
  --since 2024-01-01 \
  --finmind-token $FINMIND_TOKEN
```

執行後將透過 FinMind API 取得月營收與財報資料，自動計算 `revenue_yoy`、`revenue_mom`、`eps`、`eps_ttm` 等基本面欄位，以及 `turnover_rank_pct`、`volume_rank_pct`、`volume_ratio`、`turnover_change_5d`、`transactions_change_5d` 等資金熱度指標。結果會回寫至 `_clean_daily_wide.csv` 與 `_clean_daily_wide_min.csv`。

### 4. 檢視摘要

流程結束後，終端機會輸出資料總筆數、股票數量、日期範圍、缺值比例與隨機樣本列，幫助你快速確認資料品質。

## 🧭 後續規劃

* [ ] 加入常見技術指標（MA5、MA20、RSI、MACD…）。
* [ ] 提供法人累計買賣超（3/5/10 日）。
* [ ] 串接更多 FinMind dataset（融資融券、月營收、財報等）。
* [ ] 建立 SQLite / PostgreSQL 等資料庫儲存模式。
* [ ] 自動產生圖表與四大面向的分析報告。

## 📊 進階分析範例

```bash
python -m analysis \
  --input ./finmind_out/_clean_daily_wide.csv \
  --outdir outputs \
  --with-charts \
  --html-report \
  --fetch-fundamentals \
  --since 2024-01-01
```

以上命令會輸出圖表、HTML 報告，並可選擇同時更新基本面資料。
