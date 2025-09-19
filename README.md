

# 📖 README.md

## 專案簡介

這個專案的目的是 **自動化從 FinMind API 抓取台股資料**，並將原始資料清理整理後，輸出成適合進一步分析（例如技術面、基本面、籌碼面、資金面）的格式。

目前流程：

1. 透過 **FinMind API v4** 抓取日資料（每日股價、三大法人買賣、融資融券、月營收等）。
2. 將 raw data 儲存為 **`_merged.csv`**。
3. 進行清理與正規化：

   * 統一欄位名稱與日期格式。
   * 將法人買賣資料（長表）轉換為寬表（外資、投信、自營商自營、避險）。
   * 去重、缺值處理。
4. 輸出兩份清理後的檔案：

   * **`_clean_daily_wide.csv`**：完整寬表（價量＋法人等全部欄位）。
   * **`_clean_daily_wide_min.csv`**：精簡版，僅保留分析最常用欄位。

---

## 檔案說明

* **`_merged.csv`**

  * FinMind 抓回並初步合併的原始檔。
  * 每個日期＋股票代號可能會有多筆（因法人為長表）。

* **`_clean_daily_wide.csv`**

  * 整理後的寬表。
  * 每個日期＋股票代號唯一一行。
  * 欄位包含完整價量、法人買賣、衍生欄位等。

* **`_clean_daily_wide_min.csv`**

  * 精簡版，欄位如下：

    * `date`：日期
    * `stock_id`：股票代號
    * `open, high, low, close`：股價
    * `volume`：成交量
    * `turnover`：成交值
    * `inst_foreign`：外資淨買賣超
    * `inst_investment_trust`：投信淨買賣超
    * `inst_dealer_self`：自營商自營淨買賣超
    * `inst_dealer_hedging`：自營商避險淨買賣超

---

## 使用方法

### 1️⃣ 安裝環境

```bash
# 建議使用 Python 3.12
pip install pandas requests pyarrow
```

### 2️⃣ 抓取資料

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

### 3️⃣ 清理輸出

* 自動產生：

  * `output/_merged.csv`
  * `output/_clean_daily_wide.csv`
  * `output/_clean_daily_wide_min.csv`

### 4️⃣ 檢視摘要

程式會在終端機輸出：

* 資料總筆數
* 股票數量
* 日期範圍
* 缺值比例
* 隨機樣本列

---

## 未來規劃

* [ ] 加入技術指標（MA5、MA20、RSI、MACD…）。
* [ ] 加入法人累計買賣超（3日 / 5日 / 10日）。
* [ ] 串接更多 dataset（融資融券、月營收、財報）。
* [ ] 建立資料庫（SQLite / PostgreSQL）管理。
* [ ] 自動產生圖表與四大面向分析報告。

---

