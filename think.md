我會走「**混合資料源 + 全市場先篩後精算**」這條路，既擴大樣本、又不被 FinMind 流量卡死：

# 簡要結論

* **不要替代 FinMind**，改成：**TWSE / TPEx 官方 OpenData 撐“高流量、全市場”的欄位**（日成交價量、三大法人、融資融券…），**FinMind 補“難抓/需整理”的欄位**（外資持股比、借券成交明細、分點等）。TWSE 有完整 OpenAPI 目錄，TPEx 提供可下載 CSV 的日行情頁面。([openapi.twse.com.tw][1])
* **流程**：先做「全股票粗篩（市場→產業中性化→打分）」，產出一份全市場雷達報告；再對你的**自選清單**做「細緻版四大面向 + 敘事解讀」的第二份報告。
* **流量管理**：把高頻（日更）的都走官方開放資料；FinMind 只在必要欄位上用、並做快取與增量更新。FinMind 的官方文件也提供「使用量查詢」與超限返回格式，便於你程式內做退避。([finmind.github.io][2])

---

# 具體做法（你現在專案可直接加）

## 1) 資料源分工（推薦）

| 面向                    | 推薦來源                                                 | 說明                                                              |
| --------------------- | ---------------------------------------------------- | --------------------------------------------------------------- |
| 日成交價量（全市場批量）          | **TWSE OpenAPI** + **TPEx CSV**                      | 官方開放資料，適合全市場掃描與回補缺資料。([openapi.twse.com.tw][1])                 |
| 三大法人/市場統計             | **TWSE OpenAPI**（含統計）/ TPEx 對應頁                      | 可直接覆蓋你“籌碼面”所需的核心欄位。([openapi.twse.com.tw][1])                   |
| 月營收                   | **TWSE OpenAPI：Monthly Summary of Operating Income** | 有標準化 API，避免自行爬 MOPS。([政府資料開放平臺][3])                             |
| 財報三表、股利/除權息、借券明細、外資持股 | **FinMind**                                          | 這些欄位用 FinMind 最省開發力（官方已彙整），程式內做快取/增量即可。([finmind.github.io][4]) |
| 指數/大盤                 | **TWSE OpenData**（加權指數等）                             | 官方資料集齊全且更新日頻。([政府資料開放平臺][5])                                    |

> 備註：FinMind 文件提供「檢查使用量」與「超限 402 回應」，可在你的 ETL 內做**退避重試**或**改走官方來源**的fallback。([finmind.github.io][2])

## 2) 全市場 → 粗篩 → 精算 的兩層報告

**(A) 全市場雷達報告（每日一次）**

* 流程：取 TWSE/TPEx 全部上市櫃→產業中性化（以產業中位數/MAD 標準化）→在全市場做百分位→出四大面向分數→匯出「Top/N 落選原因」。
* 目的：快速找到“哪個產業在冒頭、哪些個股在產業內領先”。
* 來源：價量/法人/融資融券走 TWSE/TPEx；基本面低頻資料走 TWSE 月營收；需要時再補 FinMind。([openapi.twse.com.tw][1])

**(B) 觀察清單深度報告（同日第二份）**

* 流程：只對你自選的 10–30 檔，拉 FinMind 的細節（外資持股比、借券、分點、事件/除權息）＋文字敘事（異常值解釋、事件調整）。([finmind.github.io][4])
* 目的：把“可交易的下一步”說清楚（進出場區間、風險點、事件校正）。

## 3) ETL 策略（避免被流量限制卡住）

* **增量更新**：以（dataset, stock\_id, date）為鍵快取；每日只補新日期。
* **調度**：把全市場「價量/法人」放在 16:30 後批次（官網落地後），把 FinMind 細節欄位分時段拉。
* **退避機制**：遇到 FinMind 402（超限）就切回官方來源或隔分鐘重試；使用 FinMind 的 `/v2/user_info` 監控餘額。([finmind.github.io][2])
* **缺口回補**：遇到缺資料時，優先用 TWSE/TPEx 歷史 CSV/API 回補（官方頁面明確提供 CSV 下載）。([台北交易所][6])

## 4) 打分口徑（讓全市場與清單版一致）

* 預設用「**產業中性化後的全市場百分位**」做四分項；在 UI/報告保留切換：`產業內%｜全市場%（預設）｜清單內%`。
* 事件校正（減資、除權息）：在**全市場雷達**先用機械化處理（Winsorize/移動平均），**清單深度版**再補人工解釋。

## 5) 你要的兩份輸出長相

* `market_scan_YYYYMMDD.md/csv`：全市場 Top/Bottom 清單 + 產業熱度矩陣 + 因子貢獻。
* `watchlist_deep_YYYYMMDD.md/pdf`：你那 10–30 檔，逐檔四面向 + 事件/風險註解 + 建議操作區間。

---

# 要不要「抓更多資料」？

* **要**：把全市場的**價量/法人/融資券**改走 TWSE/TPEx，就可以**每天掃全市場**（樣本直接擴到全體上市櫃），而 FinMind 留給「基本面/特殊籌碼」即可。這樣**樣本量與穩定度**提升明顯，成本也更可控。([openapi.twse.com.tw][1])
* **不用全都換**：財報三表/分點/借券等 FinMind 已整理好的，繼續用它，並在你的程式內做用量監控與快取。([finmind.github.io][4])

---

如果你要，我可以把你現有 CLI 加上 `sources.yml` 與「官方/FinMind 混合策略」的開關，並新增兩個命令：
`scan-market`（全市場雷達）與 `report-watchlist`（清單深度版）。
這樣你每天就能輸出**兩份報告**：一份全市場、一份你的特定股票。

[1]: https://openapi.twse.com.tw/ "Swagger UI"
[2]: https://finmind.github.io/api_usage_count/ "API 使用次數 - FinMind"
[3]: https://data.gov.tw/en/datasets/18420?utm_source=chatgpt.com "Monthly Summary of Operating Income of Listed Companies"
[4]: https://finmind.github.io/ "FinMind"
[5]: https://data.gov.tw/en/datasets/11755 "Weighted Stock Price Index Historical Data ｜ 政府資料開放平臺"
[6]: https://www.tpex.org.tw/en-us/mainboard/trading/info/stock-pricing.html?utm_source=chatgpt.com "Daily Stock Info - Taipei Exchange"
