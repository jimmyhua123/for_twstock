from __future__ import annotations
import argparse, yaml, pandas as pd
from pathlib import Path
from .reports.market_scan import run_market_scan
from .reports.watchlist_deep import run_watchlist_report
from .official_coarse.build_coarse_features import build_from_official
from .fetch_fine import run_fetch_fine

DEFAULT_SCORES_DIR = Path("finmind_scores")
COARSE_PATTERN = "features_snapshot_coarse_*.csv"
FINE_PATTERN = "features_snapshot_fine_*.csv"
LEGACY_COARSE = Path("finmind_out/features_snapshot.csv")

def _load_yaml(p: str) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _latest_features(pattern: str, candidates: list[Path]) -> Path | None:
    for base in candidates:
        base = base.expanduser()
        if not base.exists():
            continue
        matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _resolve_features_path(profile: str, provided: str | None) -> Path:
    pattern = COARSE_PATTERN if profile == "coarse" else FINE_PATTERN
    if provided:
        path = Path(provided)
        if path.is_dir():
            found = _latest_features(pattern, [path])
            if found:
                return found
        if path.exists():
            return path
    search_dirs = []
    if provided:
        search_dirs.append(Path(provided).parent)
    search_dirs.extend([DEFAULT_SCORES_DIR, Path(".")])
    found = _latest_features(pattern, search_dirs)
    if found:
        return found
    if profile == "coarse" and LEGACY_COARSE.exists():
        return LEGACY_COARSE
    raise SystemExit(f"找不到 {profile} features 檔案，請使用 --features 指定正確的 CSV。")

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
    stamp = pd.to_datetime(args.until).strftime("%Y%m%d")
    desired = f"features_snapshot_coarse_{stamp}.csv"
    out_arg = Path(args.out_features)
    if out_arg.is_dir() or not out_arg.suffix:
        target = out_arg / desired
    elif out_arg.name != desired:
        target = out_arg.parent / desired
    else:
        target = out_arg
    target.parent.mkdir(parents=True, exist_ok=True)
    out = build_from_official(
        args.universe,
        args.since,
        args.until,
        str(target),
        sleep_ms=getattr(args, "sleep_ms", 250),
    )
    print(f"[OK] coarse features -> {out}")

def cmd_scan_market(args: argparse.Namespace):
    cfg_all = _load_yaml(args.scoring)
    profile = args.profile or "coarse"
    cfg = (cfg_all.get("profiles", {}).get(profile, {})) | {"industry_col": cfg_all.get("industry_col", "industry")}
    feats_path = _resolve_features_path(profile, args.features)
    print(f"[INFO] 使用 features 檔案：{feats_path}")
    feats = pd.read_csv(feats_path)
    _diag_missing(feats, cfg, args.output)
    run_market_scan(feats, cfg, cfg.get("universe","market_neutralized_by_industry"), args.output)

# def cmd_report_watchlist(args: argparse.Namespace):
#     cfg_all = _load_yaml(args.scoring)
#     profile = args.profile or "fine"
#     cfg = (cfg_all.get("profiles", {}).get(profile, {})) | {"industry_col": cfg_all.get("industry_col", "industry")}
#     feats = pd.read_csv(args.features)
#     wl = set(pd.read_csv(args.watchlist)["stock_id"].astype(str))
#     feats = feats[feats["stock_id"].astype(str).isin(wl)].copy()
#     _diag_missing(feats, cfg, args.output)
#     run_watchlist_report(feats, cfg, args.output)

def cmd_report_watchlist(args):
    cfg_all = _load_yaml(args.scoring)
    profile = args.profile or "fine"
    cfg = (cfg_all.get("profiles", {}).get(profile, {})) | {
        "industry_col": cfg_all.get("industry_col", "industry")
    }
    feats_path = _resolve_features_path(profile, args.features)
    print(f"[INFO] 使用 features 檔案：{feats_path}")
    feats = pd.read_csv(feats_path)
    wl = set(pd.read_csv(args.watchlist)["stock_id"].astype(str))
    feats = feats[feats["stock_id"].astype(str).isin(wl)].copy()
    _diag_missing(feats, cfg, args.output)
    run_watchlist_report(feats, cfg, args.output)


def _cmd_fetch_fine(args):
        ds = [s for s in args.datasets.split(",") if s.strip()] or None
        stat = run_fetch_fine(args.watchlist, args.since, args.until, args.outdir,
                            sleep_ms=args.sleep_ms, limit_per_hour=args.limit_per_hour,
                            max_requests=args.max_requests, datasets=ds)
        print("[FETCH-FINE DONE]", stat)
        # 若設定 auto-resume 且被 quota 擋住，就睡到時間到再繼續
        if args.auto_resume and stat.get("stopped") and "resume at" in (stat.get("reason") or ""):
            import time, datetime as dt
            ra_txt = (stat.get("reason") or "").split("resume at")[-1].strip()
            try:
                ra = dt.datetime.fromisoformat(ra_txt)
                now = dt.datetime.now(ra.tzinfo)
                wait = max(0, (ra - now).total_seconds())
                print(f"[AUTO-RESUME] sleeping {int(wait)}s until {ra.isoformat()} ...")
                time.sleep(wait+3)
                # 再跑一次
                stat2 = run_fetch_fine(args.watchlist, args.since, args.until, args.outdir,
                                    sleep_ms=args.sleep_ms, limit_per_hour=args.limit_per_hour,
                                    max_requests=args.max_requests, datasets=ds)
                print("[FETCH-FINE DONE 2nd]", stat2)
            except Exception:
                pass

def register_subcommands(subparsers):
    sp0 = subparsers.add_parser("build-coarse", help="用 TWSE/TPEx 建全市場粗篩 features")
    sp0.add_argument("--universe", required=True, help="universe_all.csv 路徑（需含 stock_id, stock_name, industry/industry_category, market）")
    sp0.add_argument("--since", required=True)
    sp0.add_argument("--until", required=True)
    sp0.add_argument("--out-features", required=True)
    sp0.add_argument("--sleep-ms", type=int, default=250)
    sp0.set_defaults(func=cmd_build_coarse)

    sp = subparsers.add_parser("scan-market", help="全市場粗篩報告")
    sp.add_argument("--features")
    sp.add_argument("--scoring", default="scoring.yml")
    sp.add_argument("--profile", default="coarse", choices=["coarse","fine"])
    sp.add_argument("--output", default="finmind_out/market_scan")
    sp.set_defaults(func=cmd_scan_market)

    sp2 = subparsers.add_parser("report-watchlist", help="自選清單深度報告")
    sp2.add_argument("--features")
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
    sp3.add_argument("--auto-resume", action="store_true", help="若 quota 尚未到期則睡到 resume_at 再繼續")

    sp3.set_defaults(func=_cmd_fetch_fine)
