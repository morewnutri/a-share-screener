from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

from . import __version__
from .backtest import Backtester
from .calendar import expected_complete_session
from .config import load_config
from .reporting import print_run_summary
from .scanner import DailyScanner


def _as_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share main-board end-of-day setup scanner")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument("--data-dir", help="Override persistent data directory")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the latest complete-session scan")
    run_parser.add_argument(
        "--as-of",
        help="ISO datetime used to determine the latest complete session (testing/replay)",
    )
    run_parser.add_argument(
        "--print-top",
        type=int,
        default=20,
        help="Number of candidates printed for each signal",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Backtest using cached histories")
    backtest_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    backtest_parser.add_argument("--end", required=True, help="YYYY-MM-DD")

    date_parser = subparsers.add_parser("expected-date", help="Print the expected complete session")
    date_parser.add_argument("--as-of", help="ISO datetime")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config(Path(args.config))
    if args.command == "run":
        output = DailyScanner(config, args.data_dir).run(_as_datetime(args.as_of))
        print_run_summary(output, args.print_top)
    elif args.command == "backtest":
        output = Backtester(config, args.data_dir).run(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )
        print(output)
    else:
        print(
            expected_complete_session(
                _as_datetime(args.as_of),
                config.data.close_buffer_minutes,
            ).isoformat()
        )
    return 0
