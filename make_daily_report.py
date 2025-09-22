# -*- coding: utf-8 -*-
"""
make_daily_report.py
讀取 scores_watchlist.csv / scores_breakdown.csv / features_snapshot.csv
輸出：每日 Markdown + HTML 日報（UTF-8-SIG）

功能：
- 自動決定 as-of 日期（或以 --asof 指定）
- 產業摘要（平均分數/檔數）
- Top/Bottom 名單（含每檔 3 條重點訊號）
- 關鍵欄位表格（Top 表 + Optional 全表）
- HTML 版：表格用 DataFrame.to_html；Markdown 版：用簡易 pipe table

作者：你
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def log(m): print(m, flush=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[錯誤] 找不到檔案：{path}")
    return pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")

def resolve_latest(path_str: str, prefix: str) -> Path:
    """
    如果 path_str 是既有檔案就用它；
    若不是檔案：把它視為資料夾，在裡面挑選 prefix_*.csv 的最新一個。
    """
    p = Path(path_str)
    if p.exists() and p.is_file():
        return p
    # 當作資料夾處理
    folder = p if p.exists() and p.is_dir() else p.parent
    if not folder.exists():
        raise SystemExit(f"[錯誤] 找不到檔案或資料夾：{path_str}")
    cands = sorted(folder.glob(f"{prefix}_*.csv"))
    if not cands:
        # 也許是未帶日期的舊檔名
        legacy = folder / f"{prefix}.csv"
        if legacy.exists():
            return legacy
        raise SystemExit(f"[錯誤] 找不到 {prefix}_*.csv 或 {prefix}.csv 於：{folder}")
    return cands[-1]  # 以檔名排序的最後一個視為最新（YYYYMMDD 遞增）

def percent_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    return s.rank(pct=True) * 100.0

def fmt_pct(x, digits=1, sign=False):
    try:
        v = float(x)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    p = f"{v*100:.{digits}f}%"
    if sign and v > 0:
        p = f"+{p}"
    return p

def fmt_num(x, digits=0, sign=False):
    try:
        v = float(x)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    s = f"{v:.{digits}f}"
    if sign and v > 0:
        s = f"+{s}"
    return s

def pick_signals(row, ranks_all):
    """給單檔輸出 3 條重點訊號（正向為主，若不足再補中性/估值/風險）"""
    sig = []
    def add(msg):
        if msg and msg not in sig:
            sig.append(msg)

    # 取分位做參考
    def is_top(col, p=75):
        if col not in ranks_all: return False
        r = ranks_all[col].get(row.name, np.nan)
        return pd.notna(r) and r >= p

    # 技術與相對表現
    if "excess_ret_20d" in row and pd.notna(row["excess_ret_20d"]) and row["excess_ret_20d"] > 0:
        add(f"相對大盤20日 {fmt_pct(row['excess_ret_20d'],1,True)}")
    elif "ret_20d" in row and pd.notna(row["ret_20d"]) and row["ret_20d"] > 0:
        add(f"20日報酬 {fmt_pct(row['ret_20d'],1,True)}")

    # 籌碼
    if "chip_net5_sum" in row and pd.notna(row["chip_net5_sum"]) and row["chip_net5_sum"] > 0:
        add(f"三大法人5日淨買超 {fmt_num(row['chip_net5_sum'],0,True)}")
    if "foreign_ratio_5d_chg" in row and pd.notna(row["foreign_ratio_5d_chg"]) and row["foreign_ratio_5d_chg"] > 0:
        add(f"外資持股比5日 {fmt_num(row['foreign_ratio_5d_chg'],2,True)}%")

    # 基本面
    if "revenue_yoy" in row and pd.notna(row["revenue_yoy"]):
        if row["revenue_yoy"] > 0:
            add(f"月營收YoY {fmt_pct(row['revenue_yoy'],1,True)}")
        # 若 YoY 無法加進前3，之後可能再補

    # 估值（越低越好）
    if "pe" in row and pd.notna(row["pe"]) and row["pe"] > 0:
        add(f"PE≈{fmt_num(row['pe'],1)}")
    if "pb" in row and pd.notna(row["pb"]) and row["pb"] > 0 and len(sig) < 3:
        add(f"PB≈{fmt_num(row['pb'],2)}")

    # 風險（score_risk 高代表低風險）
    if "score_risk" in row and pd.notna(row["score_risk"]) and row["score_risk"] >= 70 and len(sig) < 3:
        add("波動風險低（score_risk≥70）")

    # 若還不到 3 條，補上最高的面向分數
    if len(sig) < 3:
        best = []
        for c in ["score_tech","score_chip","score_fund"]:
            if c in row and pd.notna(row[c]):
                best.append((c, row[c]))
        best = sorted(best, key=lambda x: x[1], reverse=True)[:3-len(sig)]
        for c, v in best:
            name = {"score_tech":"技術強度","score_chip":"籌碼流向","score_fund":"基本體質"}.get(c,c)
            add(f"{name} {int(round(v))} 分")

    return sig[:3]

def build_tables(watch: pd.DataFrame, top_n=10, bottom_n=5):
    # 排序
    if "score_total" in watch.columns:
        watch = watch.sort_values("score_total", ascending=False)
    # 欄位選擇
    cols_pref = [
        "stock_id","stock_name","industry_category","close",
        "ret_5d","ret_20d","excess_ret_20d",
        "chip_net5_sum","chip_net20_sum","foreign_ratio_5d_chg",
        "revenue_yoy","pe","pb",
        "score_tech","score_chip","score_fund","score_risk","score_total"
    ]
    cols = [c for c in cols_pref if c in watch.columns]
    table_all = watch[cols].copy()

    # 產生 ranks 用於訊號（百分位）
    ranks_all = {}
    for c in ["ret_20d","excess_ret_20d","chip_net5_sum","revenue_yoy","pe","pb","score_total"]:
        if c in table_all.columns:
            ranks_all[c] = percent_rank(table_all[c])
        else:
            ranks_all[c] = pd.Series(np.nan, index=table_all.index)

    # Top/Bottom 擷取
    top_df = table_all.head(top_n).copy()
    bottom_df = table_all.tail(bottom_n).copy().sort_values("score_total", ascending=True) if bottom_n>0 else pd.DataFrame()

    return table_all, top_df, bottom_df, ranks_all

def df_to_md_table(df: pd.DataFrame, max_rows=50) -> str:
    """簡易把 DataFrame 轉為 Markdown pipe table（避免額外相依套件）"""
    if df.empty:
        return "_（無資料）_"
    df2 = df.head(max_rows).copy()
    # 簡單格式化
    for c in df2.columns:
        if "ret_" in c or "excess" in c or "yoy" in c:
            df2[c] = df2[c].map(lambda x: fmt_pct(x,1) if pd.notna(x) else "")
        elif c.startswith("score_"):
            df2[c] = df2[c].map(lambda x: f"{int(round(x))}" if pd.notna(x) else "")
        elif c in {"close","pe","pb","chip_net5_sum","chip_net20_sum","foreign_ratio_5d_chg"}:
            df2[c] = df2[c].map(lambda x: fmt_num(x,2) if pd.notna(x) else "")
    # 生成表格
    headers = "| " + " | ".join(df2.columns) + " |"
    sep = "| " + " | ".join(["---"]*len(df2.columns)) + " |"
    rows = ["| " + " | ".join(map(lambda v: "" if pd.isna(v) else str(v), row)) + " |" for row in df2.values]
    return "\n".join([headers, sep] + rows)

def make_report_md(asof_date, watch, top_df, bottom_df, ranks_all, out_md: Path):
    dt = asof_date.strftime("%Y-%m-%d")
    # 產業摘要
    ind_summary = ""
    if "industry_category" in watch.columns and "score_total" in watch.columns:
        grp = (watch.groupby("industry_category", dropna=False)["score_total"]
               .agg(["count","mean"])
               .sort_values("mean", ascending=False))
        ind_summary = grp.head(10)

    # 生成 Top/Bottom 訊號
    def section_for(df, title):
        if df.empty:
            return f"## {title}\n\n_（無資料）_\n"
        lines = [f"## {title}\n"]
        for idx, row in df.iterrows():
            name = f"{row.get('stock_id','')}"
            if pd.notna(row.get("stock_name", np.nan)):
                name += f" {row['stock_name']}"
            if pd.notna(row.get("industry_category", np.nan)):
                name += f"｜{row['industry_category']}"
            lines.append(f"### {name}")
            sigs = pick_signals(row, ranks_all)
            for s in sigs:
                lines.append(f"- {s}")
            # 附上核心數值一行
            snips = []
            if "score_total" in row and pd.notna(row["score_total"]):
                snips.append(f"總分 {int(round(row['score_total']))}")
            if "ret_20d" in row and pd.notna(row["ret_20d"]):
                snips.append(f"20D {fmt_pct(row['ret_20d'],1,True)}")
            if "excess_ret_20d" in row and pd.notna(row["excess_ret_20d"]):
                snips.append(f"Ex20D {fmt_pct(row['excess_ret_20d'],1,True)}")
            if snips:
                lines.append(f"_（{'；'.join(snips)}）_")
            lines.append("")  # blank
        return "\n".join(lines) + "\n"

    md_parts = []
    md_parts.append(f"# 台股觀察日報｜{dt}\n")
    md_parts.append("本報表由 scores_watchlist / breakdown / snapshot 產生，分數為產業內相對分位（0–100）。\n")
    # 名詞解釋
    md_parts.append("## 名詞解釋\n")
    md_parts.append(
        "- **score_total**：四大面向加權總分（預設 技術0.3／籌碼0.3／基本0.3／風險0.1）。\n"
        "- **score_tech／chip／fund／risk**：各面向在**產業內**做百分位分數（0–100）。\n"
        "- **ret_5d／ret_20d**：近 5／20 個交易日收盤報酬率。\n"
        "- **excess_ret_20d**：個股 20 日報酬 − 加權指數 20 日報酬。\n"
        "- **chip_net5_sum／chip_net20_sum**：三大法人 5／20 日滾動淨買賣合計（張/金額，資料源而定）。\n"
        "- **foreign_ratio_5d_chg**：外資持股比 5 日變化（百分點）。\n"
        "- **revenue_yoy**：月營收年增率（公告後**右對齊**沿用至下一次更新）。\n"
        "- **pe／pb**：估值指標（越低越佳，於產業內反向計分）。\n"
        "- **rolling_vol_20d**：以日報酬計算的 20 日波動（越低風險越低）。\n"
        "- **smaX_gap_pct**：收盤價相對 X 日均線差距百分比。\n"
        "- **rsi14**：14 日 RSI 指標；**vol_mult_20d**：相對 20 日均量的量能倍數。\n"
    )
    # 產業摘要
    if isinstance(ind_summary, pd.DataFrame) and not ind_summary.empty:
        md_parts.append("## 產業摘要（均分 Top 10）\n")
        md_parts.append(df_to_md_table(ind_summary.reset_index().rename(columns={"index":"industry"}), max_rows=20))
        md_parts.append("")

    # Top/Bottom
    md_parts.append(section_for(top_df, "Top 名單"))
    if bottom_df is not None and not bottom_df.empty:
        md_parts.append(section_for(bottom_df, "Bottom 名單"))

    # Top 表格（關鍵欄位）
    md_parts.append("## Top 名單（關鍵欄位）\n")
    md_parts.append(df_to_md_table(top_df, max_rows=50))
    md_parts.append("")

    # 全表（可註解）
    md_parts.append("## 全表（節錄）\n")
    md_parts.append(df_to_md_table(watch, max_rows=100))
    md_parts.append("\n_註：百分比欄位為格式化後顯示，詳細數值請見原始 CSV。_\n")

    content = "\n".join(md_parts)
    out_md.write_text(content, encoding="utf-8-sig")
    log(f"[輸出] {out_md}")

def make_report_html(asof_date, watch, top_df, bottom_df, ranks_all, out_html: Path):
    dt = asof_date.strftime("%Y-%m-%d")
    # 簡單 HTML，表格用 pandas.to_html（自帶 <table>）
    def df_html(df, max_rows=50):
        if df.empty: return "<i>（無資料）</i>"
        df2 = df.head(max_rows).copy()
        return df2.to_html(index=False, escape=False)

    # 產業摘要
    ind_html = ""
    if "industry_category" in watch.columns and "score_total" in watch.columns:
        grp = (watch.groupby("industry_category", dropna=False)["score_total"]
               .agg(["count","mean"])
               .sort_values("mean", ascending=False))
        ind_html = df_html(grp.reset_index().rename(columns={"index":"industry"}), 20)

    # Top/Bottom 卡片
    def cards(df, title):
        parts = [f"<h2>{title}</h2>"]
        if df.empty:
            parts.append("<p><i>（無資料）</i></p>")
            return "\n".join(parts)
        for _, row in df.iterrows():
            name = f"{row.get('stock_id','')}"
            if pd.notna(row.get("stock_name", np.nan)):
                name += f" {row['stock_name']}"
            if pd.notna(row.get("industry_category", np.nan)):
                name += f"｜{row['industry_category']}"
            parts.append(f"<h3>{name}</h3>")
            sigs = pick_signals(row, ranks_all)
            parts.append("<ul>" + "".join([f"<li>{s}</li>" for s in sigs]) + "</ul>")
            snips = []
            if "score_total" in row and pd.notna(row["score_total"]):
                snips.append(f"總分 {int(round(row['score_total']))}")
            if "ret_20d" in row and pd.notna(row["ret_20d"]):
                snips.append(f"20D {fmt_pct(row['ret_20d'],1,True)}")
            if "excess_ret_20d" in row and pd.notna(row["excess_ret_20d"]):
                snips.append(f"Ex20D {fmt_pct(row['excess_ret_20d'],1,True)}")
            if snips:
                parts.append(f"<p><i>（{'；'.join(snips)}）</i></p>")
        return "\n".join(parts)

    html = []
    html.append("<meta charset='utf-8'>")
    html.append(f"<h1>台股觀察日報｜{dt}</h1>")
    html.append("<p>本報表由 scores_watchlist / breakdown / snapshot 產生，分數為產業內相對分位（0–100）。</p>")

    # 名詞解釋（HTML）
    html.append("<h2>名詞解釋</h2>")
    html.append("""
    <ul>
      <li><b>score_total</b>：四大面向加權總分（預設 技術0.3／籌碼0.3／基本0.3／風險0.1）。</li>
      <li><b>score_tech／chip／fund／risk</b>：各面向在<strong>產業內</strong>的百分位分數（0–100）。</li>
      <li><b>ret_5d／ret_20d</b>：近 5／20 個交易日收盤報酬率。</li>
      <li><b>excess_ret_20d</b>：個股 20 日報酬 − 加權指數 20 日報酬。</li>
      <li><b>chip_net5_sum／chip_net20_sum</b>：三大法人 5／20 日滾動淨買賣合計（張/金額）。</li>
      <li><b>foreign_ratio_5d_chg</b>：外資持股比 5 日變化（百分點）。</li>
      <li><b>revenue_yoy</b>：月營收年增率（公告後右對齊沿用）。</li>
      <li><b>pe／pb</b>：估值指標（越低越佳；產業內反向計分）。</li>
      <li><b>rolling_vol_20d</b>：以日報酬計算的 20 日波動。</li>
      <li><b>smaX_gap_pct</b>：收盤價相對 X 日均線差距百分比；<b>rsi14</b>：14 日 RSI；<b>vol_mult_20d</b>：量能倍數。</li>
    </ul>
    """)


    if ind_html:
        html.append("<h2>產業摘要（均分 Top 10）</h2>")
        html.append(ind_html)

    html.append(cards(top_df, "Top 名單"))
    if bottom_df is not None and not bottom_df.empty:
        html.append(cards(bottom_df, "Bottom 名單"))

    html.append("<h2>Top 名單（關鍵欄位）</h2>")
    html.append(df_html(top_df, 50))

    html.append("<h2>全表（節錄）</h2>")
    html.append(df_html(watch, 100))
    html.append("<p><i>註：百分比欄位為格式化後顯示，詳細數值請見原始 CSV。</i></p>")

    out_html.write_text("\n".join(html), encoding="utf-8-sig")
    log(f"[輸出] {out_html}")

def main():
    ap = argparse.ArgumentParser(description="台股觀察日報（Markdown + HTML）")
    ap.add_argument("--scores", default="finmind_scores/scores_watchlist.csv")
    ap.add_argument("--breakdown", default="finmind_scores/scores_breakdown.csv")
    ap.add_argument("--snapshot", default="finmind_scores/features_snapshot.csv")
    ap.add_argument("--outdir", default="finmind_reports")
    ap.add_argument("--asof", type=str, default=None, help="評分日期 YYYY-MM-DD（不給則自動抓最新）")
    ap.add_argument("--top", type=int, default=10, help="Top 名單檔數")
    ap.add_argument("--bottom", type=int, default=5, help="Bottom 名單檔數")
    args = ap.parse_args()

        # 支援帶日期檔名：可直接傳資料夾，會自動撈最新的
    scores_path   = resolve_latest(args.scores,   "scores_watchlist")
    breakdown_path= resolve_latest(args.breakdown,"scores_breakdown")
    snapshot_path = resolve_latest(args.snapshot, "features_snapshot")

    scores   = read_csv(scores_path)
    breakdown= read_csv(breakdown_path)
    snap     = read_csv(snapshot_path)

    # 補 date 欄（如果 watchlist 缺）→ 從 snapshot 取各檔最新
    if "date" not in scores.columns and "date" in snap.columns:
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce")
        recent = snap.sort_values(["stock_id","date"]).drop_duplicates("stock_id", keep="last")[["stock_id","date"]]
        scores = scores.merge(recent, on="stock_id", how="left")

    # 去重（同檔保留最新）
    if "date" in scores.columns:
        scores["date"] = pd.to_datetime(scores["date"], errors="coerce")
        scores = scores.sort_values(["stock_id","date"]).drop_duplicates("stock_id", keep="last")

    # 決定 as-of 日期
    if args.asof:
        asof_date = pd.to_datetime(args.asof)
    else:
        if "date" in scores.columns and scores["date"].notna().any():
            asof_date = scores["date"].max()
        else:
            # fallback 今日
            asof_date = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))

    # 只留 as-of 列
    if "date" in scores.columns:
        watch = scores[scores["date"] == asof_date].copy()
    else:
        watch = scores.copy()

    # 若仍有同檔多列，再壓一次
    watch = watch.sort_values(["stock_id"]).drop_duplicates("stock_id", keep="last")

    # 準備表格
    table_all, top_df, bottom_df, ranks_all = build_tables(watch, args.top, args.bottom)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_md   = outdir / f"daily_report_{asof_date.strftime('%Y%m%d')}.md"
    out_html = outdir / f"daily_report_{asof_date.strftime('%Y%m%d')}.html"

    make_report_md(asof_date, table_all, top_df, bottom_df, ranks_all, out_md)
    make_report_html(asof_date, table_all, top_df, bottom_df, ranks_all, out_html)

if __name__ == "__main__":
    main()
