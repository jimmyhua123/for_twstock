from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:  # pragma: no cover - optional dependency guard
    import yaml
except ImportError:  # pragma: no cover - runtime check
    yaml = None  # type: ignore[assignment]

from .reports.market_scan import run_market_scan
from .reports.watchlist_deep import run_watchlist_report

DEFAULT_FEATURES = "finmind_out/features_snapshot.csv"


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("請先安裝 pyyaml：pip install pyyaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_scan_market(args: argparse.Namespace) -> None:
    config = _load_yaml(args.scoring)
    feats_path = Path(args.features or DEFAULT_FEATURES)
    if not feats_path.exists():
        raise SystemExit(f"features 檔不存在：{feats_path}")
    feats = pd.read_csv(feats_path)
    universe = args.universe or config.get("universe", {}).get("default", "market_neutralized_by_industry")
    result = run_market_scan(feats, config, universe, args.output)
    print(f"已產出全市場粗篩報告：{result}")


def cmd_report_watchlist(args: argparse.Namespace) -> None:
    config = _load_yaml(args.scoring)
    feats = pd.read_csv(args.features)
    watchlist_path = Path(args.watchlist)
    if not watchlist_path.exists():
        raise SystemExit(f"watchlist 檔不存在：{watchlist_path}")
    watchlist_df = pd.read_csv(watchlist_path)
    if "stock_id" not in watchlist_df.columns:
        raise SystemExit("watchlist.csv 需要包含 stock_id 欄位")
    wl = set(watchlist_df["stock_id"].astype(str))
    feats = feats[feats["stock_id"].astype(str).isin(wl)].copy()
    if feats.empty:
        raise SystemExit("features 內無符合 watchlist 的股票")
    result = run_watchlist_report(feats, config, args.output)
    print(f"已產出自選清單報告：{result}")


def register_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sp = subparsers.add_parser("scan-market", help="全市場粗篩報告")
    sp.add_argument("--features", default=DEFAULT_FEATURES)
    sp.add_argument("--scoring", default="scoring.yml")
    sp.add_argument(
        "--universe",
        default="market_neutralized_by_industry",
        choices=["market_neutralized_by_industry", "industry", "watchlist", "market"],
    )
    sp.add_argument("--output", default="finmind_out/market_scan")
    sp.set_defaults(func=cmd_scan_market)

    sp2 = subparsers.add_parser("report-watchlist", help="自選清單深度報告")
    sp2.add_argument("--features", required=True)
    sp2.add_argument("--watchlist", required=True)
    sp2.add_argument("--scoring", default="scoring.yml")
    sp2.add_argument("--output", default="finmind_out/watchlist_deep")
    sp2.set_defaults(func=cmd_report_watchlist)
