from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from .reports.market_scan import run_market_scan
from .reports.watchlist_deep import run_watchlist_report

try:  # optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_FEATURES = "finmind_out/features_snapshot.csv"


def _parse_scalar(value: str):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    # numbers
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_inline_list(value: str):
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [_parse_scalar(p) for p in parts]


def _minimal_yaml_load(text: str) -> dict:
    lines = text.splitlines()

    def parse_block(index: int, indent: int):
        data = {}
        i = index
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            current_indent = len(raw) - len(raw.lstrip(" "))
            if current_indent < indent:
                break
            if stripped.startswith("- "):
                raise ValueError("Unexpected list item without key context")
            if ":" not in stripped:
                i += 1
                continue
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest:
                if rest.startswith("[") and rest.endswith("]"):
                    data[key] = _parse_inline_list(rest)
                else:
                    data[key] = _parse_scalar(rest)
                i += 1
                continue
            # lookahead for list or nested block
            i += 1
            # skip blank/comment lines when determining next item
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("#")):
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith("- "):
                lst, i = parse_list(i, indent + 2)
                data[key] = lst
            else:
                sub, i = parse_block(i, indent + 2)
                data[key] = sub
        return data, i

    def parse_list(index: int, indent: int):
        items = []
        i = index
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            current_indent = len(raw) - len(raw.lstrip(" "))
            if current_indent < indent:
                break
            if not stripped.startswith("- "):
                break
            item = stripped[2:].strip()
            if item.startswith("[") and item.endswith("]"):
                items.append(_parse_inline_list(item))
                i += 1
            elif item and ":" in item:
                # list item with inline dict ("- key: value")
                sub_lines = [" " * (indent + 2) + item]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    nxt_strip = nxt.strip()
                    nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                    if nxt_indent <= indent or (nxt_strip and nxt_strip.startswith("- ")):
                        break
                    sub_lines.append(nxt)
                    j += 1
                sub_text = "\n".join(sub_lines)
                items.append(_minimal_yaml_load(sub_text))
                i = j
            else:
                items.append(_parse_scalar(item))
                i += 1
        return items, i

    data, _ = parse_block(0, 0)
    return data


def _load_yaml(p: str) -> dict:
    text = Path(p).read_text(encoding="utf-8")
    if yaml is not None:  # pragma: no branch
        return yaml.safe_load(text)
    return _minimal_yaml_load(text)


def _diag_missing(features_df: pd.DataFrame, config: dict, out_dir: str):
    feats_cfg = config.get("features", {})
    recs = []
    for pillar, cols in feats_cfg.items():
        for c in cols:
            exists = c in features_df.columns
            miss_rate = float(features_df[c].isna().mean()) if exists else 1.0
            recs.append({"pillar": pillar, "column": c, "exists": exists, "missing_rate": round(miss_rate, 4)})
    if recs:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_csv(out/"_diag_missing_features.csv", index=False, encoding="utf-8")


def cmd_scan_market(args: argparse.Namespace):
    config = _load_yaml(args.scoring)
    feats_path = Path(args.features or DEFAULT_FEATURES)
    if not feats_path.exists():
        raise SystemExit(f"features 檔不存在：{feats_path}")
    feats = pd.read_csv(feats_path)
    _diag_missing(feats, config, args.output)
    run_market_scan(feats, config, args.universe, args.output)


def cmd_report_watchlist(args: argparse.Namespace):
    config = _load_yaml(args.scoring)
    feats = pd.read_csv(args.features)
    wl = set(pd.read_csv(args.watchlist)["stock_id"].astype(str))
    feats = feats[feats["stock_id"].astype(str).isin(wl)].copy()
    _diag_missing(feats, config, args.output)
    run_watchlist_report(feats, config, args.output)


def register_subcommands(subparsers):
    sp = subparsers.add_parser("scan-market", help="全市場粗篩報告")
    sp.add_argument("--features", default=DEFAULT_FEATURES)
    sp.add_argument("--scoring", default="scoring.yml")
    sp.add_argument("--universe", default="market_neutralized_by_industry", choices=["market_neutralized_by_industry","industry","watchlist","market"])
    sp.add_argument("--output", default="finmind_out/market_scan")
    sp.set_defaults(func=cmd_scan_market)

    sp2 = subparsers.add_parser("report-watchlist", help="自選清單深度報告")
    sp2.add_argument("--features", required=True)
    sp2.add_argument("--watchlist", required=True)  # CSV with stock_id column
    sp2.add_argument("--scoring", default="scoring.yml")
    sp2.add_argument("--output", default="finmind_out/watchlist_deep")
    sp2.set_defaults(func=cmd_report_watchlist)
