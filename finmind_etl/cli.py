from __future__ import annotations

import argparse
from typing import Sequence

from .cli_extras import register_subcommands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finmind_etl",
        description="Utilities for generating market scan and watchlist reports",
    )
    subparsers = parser.add_subparsers(dest="cmd")

    # 新增子命令
    register_subcommands(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
