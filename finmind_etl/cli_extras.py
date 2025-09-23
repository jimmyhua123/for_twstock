from __future__ import annotations
import argparse, yaml, pandas as pd
from pathlib import Path
from .reports.market_scan import run_market_scan
from .reports.watchlist_deep import run_watchlist_report
from .official_coarse.build_coarse_features import build_from_official
from .fetch_fine import run_fetch_fine

DEFAULT_FEATURES = "finmind_out/features_snapshot.csv"

def _load_yaml(p: str) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

def _diag_missing(features_df: pd.DataFrame, config: dict, out_dir: str):
    feats_cfg = config.get("features", {}) or config.get("profiles",{}).get("coarse",{}).get("features",{})
    recs=[]
    for pillar, cols in feats_cfg.items():
        for c in cols:
            exists = c in features_df.columns
            miss_rate = float(features_df[c].isna().mean()) if exists else 1.0
            recs.append({"pillar": pillar, "column": c, "exists": exists, "missing_rate": round(miss_rate,4)})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out/"_diag_missing_features.csv", index=False, encoding="utf-8")

def cmd_build_coarse(args: argparse.Namespace):
    out = build_from_official(
        args.universe,
        args.since,
        args.until,
        args.out_features,
        sleep_ms=getattr(args, "sleep_ms", 250),
    )
    print(f"[OK] coarse features -> {out}")

def cmd_scan_market(args: argparse.Namespace):
    cfg_all = _load_yaml(args.scoring)
    profile = args.profile or "coarse"
    cfg = (cfg_all.get("profiles", {}).get(profile, {})) | {"industry_col": cfg_all.get("industry_col", "industry")}
    feats_path = Path(args.features or DEFAULT_FEATURES)
    if not feats_path.exists():
        raise SystemExit(f"features 檔不存在：{feats_path}")
    feats = pd.read_csv(feats_path)
    _diag_missing(feats, cfg, args.output)
    run_market_scan(feats, cfg, cfg.get("universe","market_neutralized_by_industry"), args.output)

def cmd_report_watchlist(args: argparse.Namespace):
    cfg_all = _load_yaml(args.scoring)
    profile = args.profile or "fine"
    cfg = (cfg_all.get("profiles", {}).get(profile, {})) | {"industry_col": cfg_all.get("industry_col", "industry")}
    feats = pd.read_csv(args.features)
    wl = set(pd.read_csv(args.watchlist)["stock_id"].astype(str))
    feats = feats[feats["stock_id"].astype(str).isin(wl)].copy()
    _diag_missing(feats, cfg, args.output)
    run_watchlist_report(feats, cfg, args.output)


def cmd_fetch_fine(args: argparse.Namespace):
    datasets = None
    if args.datasets:
        datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    res = run_fetch_fine(
        watchlist_csv=args.watchlist,
        since=args.since,
        until=args.until,
        outdir=args.outdir,
        sleep_ms=args.sleep_ms,
        limit_per_hour=args.limit_per_hour,
        max_requests=args.max_requests,
        state_file=args.state_file,
        datasets=datasets,
    )
    print("[FETCH-FINE DONE]", res)

def register_subcommands(subparsers):
    sp0 = subparsers.add_parser("build-coarse", help="用 TWSE/TPEx 建全市場粗篩 features")
    sp0.add_argument("--universe", required=True, help="universe_all.csv 路徑（需含 stock_id, stock_name, industry/industry_category, market）")
    sp0.add_argument("--since", required=True)
    sp0.add_argument("--until", required=True)
    sp0.add_argument("--out-features", required=True)
    sp0.add_argument("--sleep-ms", type=int, default=250)
    sp0.set_defaults(func=cmd_build_coarse)

    sp = subparsers.add_parser("scan-market", help="全市場粗篩報告")
    sp.add_argument("--features", default=DEFAULT_FEATURES)
    sp.add_argument("--scoring", default="scoring.yml")
    sp.add_argument("--profile", default="coarse", choices=["coarse","fine"])
    sp.add_argument("--output", default="finmind_out/market_scan")
    sp.set_defaults(func=cmd_scan_market)

    sp2 = subparsers.add_parser("report-watchlist", help="自選清單深度報告")
    sp2.add_argument("--features", required=True)
    sp2.add_argument("--watchlist", required=True)  # CSV with stock_id column
    sp2.add_argument("--scoring", default="scoring.yml")
    sp2.add_argument("--profile", default="fine", choices=["coarse","fine"])
    sp2.add_argument("--output", default="finmind_out/watchlist_deep")
    sp2.set_defaults(func=cmd_report_watchlist)

    sp3 = subparsers.add_parser("fetch-fine", help="抓精算用 FinMind 資料（含配額追蹤、整點續跑）")
    sp3.add_argument("--watchlist", required=True)
    sp3.add_argument("--since", required=True)
    sp3.add_argument("--until", required=True)
    sp3.add_argument("--outdir", default="finmind_raw")
    sp3.add_argument("--sleep-ms", type=int, default=900)
    sp3.add_argument("--limit-per-hour", type=int, default=600)
    sp3.add_argument("--max-requests", type=int, default=550)
    sp3.add_argument("--state-file", default="finmind_raw/_quota/finmind_quota.json")
    sp3.add_argument("--datasets", help="僅抓這些 dataset，逗號分隔；預設抓內建 plan")
    sp3.set_defaults(func=cmd_fetch_fine)
