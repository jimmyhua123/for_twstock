

# 基本用法（過去一年、RAW JSON）
python main.py

# 指定期間 + 同步輸出 CSV（需 pandas）
python main.py --since 2024-01-01 --until 2025-09-21 --to-csv

# 全市場（較久，建議加點延遲）
python main.py --all-market --sleep 0.1

# 安裝需求
pip install pandas numpy

# 執行（預設讀 finmind_raw/，輸出到 finmind_out/）
python finmind_clean_standardize.py

# 指定路徑
python finmind_clean_standardize.py --raw-dir path/to/finmind_raw --out-dir path/to/finmind_out

# 安裝需求
pip install pandas numpy

# 預設讀 clean 目錄（上一支輸出）：finmind_out/
# 可選：讀 raw 的 TaiwanStockPER.json 來補估值
python finmind_features_scoring.py   --clean-dir finmind_out   --raw-dir finmind_raw   --out-dir finmind_scores

# 指定評分日期（YYYY-MM-DD），不指定則用 price 的最新交易日
python finmind_features_scoring.py --asof 2025-09-19

# 產出整段期間的每日特徵（檔案較大）
python finmind_features_scoring.py --full-daily

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


# 典型條件（和我們先前建議一致）
python filter_watchlist.py --min-total 70 --min-tech 60 --min-chip 60 --min-risk 40 --excess-positive --top 20

# 指定某天（若你想回看 2025-09-19）
python filter_watchlist.py --asof 2025-09-19 --min-total 70 --excess-positive

# 只做去重 + 指定日期，不限制分數
python filter_watchlist.py --asof 2025-09-19 --min-total 0 --min-tech 0 --min-chip 0 --min-risk 0
