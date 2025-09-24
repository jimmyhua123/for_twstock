# 驗證最新 fine 檔是否含原始欄位
$Fine = (Get-ChildItem finmind_scores\features_snapshot_fine_*.csv | Sort-Object LastWriteTime -Desc | Select-Object -First 1).FullName
python - <<'PY'
import pandas as pd, glob, os
p = r"$Fine"
df = pd.read_csv(p, nrows=3); cols=set(df.columns)
print("FILE:", p)
def chk(xs): 
    miss=[c for c in xs if c not in cols]
    print("need:", xs, "missing:", miss)
chk(["rsi_14","breakout_20d","volatility_20d","volume_ratio_20d"])
chk(["revenue_yoy","gross_margin_ttm","roe_ttm","op_margin_ttm"])
chk(["inst_net_buy_5d_ratio","inst_consistency_20d","margin_short_ratio_5d","borrow_balance_chg_5d"])
chk(["drawdown_60d"])
PY
