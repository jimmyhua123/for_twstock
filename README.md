

# 基本用法（過去一年、RAW JSON）
python main.py

# 指定期間 + 同步輸出 CSV（需 pandas）
python main.py --since 2024-01-01 --until 2025-09-21 --to-csv

# 全市場（較久，建議加點延遲）
python main.py --all-market --sleep 0.1
