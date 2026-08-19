"""
paper CLI — default workflow is model-first (`decide`).
`forecast` (blind human) is optional. No edit/revise forecast command.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="paper",
        description="Paper trading: decide (model-first) → mark → report",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dec = sub.add_parser(
        "decide",
        help="MODEL-FIRST: show model verdict, then trade or skip (recommended)",
    )
    p_dec.add_argument("--ticker", required=True)
    p_dec.add_argument("--horizon", type=int, default=30)
    p_dec.add_argument("--auto-open", action="store_true",
                       help="If model says TRADE and you accept, open without a second prompt")

    p_fc = sub.add_parser("forecast", help="Optional: blind human forecast (no model shown)")
    p_fc.add_argument("--ticker", required=True)

    p_ev = sub.add_parser("evaluate", help="Re-run model on an existing forecast_id")
    p_ev.add_argument("--forecast-id", required=True)

    p_open = sub.add_parser("open", help="Open a trade from a logged forecast")
    p_open.add_argument("--forecast-id", required=True)
    p_open.add_argument("--override", action="store_true")
    p_open.add_argument("--override-reason", default="")

    p_mark = sub.add_parser("mark", help="Mark open trades from chain_history; auto-close exits")
    p_mark.add_argument("--asof", default=None)

    sub.add_parser("report", help="Model calibration scorecard; P&L last and noisy")

    args = parser.parse_args(argv)

    if args.cmd == "decide":
        from paper.entry import decide
        decide(args.ticker, horizon_days=args.horizon, auto_open=args.auto_open)
        return 0

    if args.cmd == "forecast":
        from paper.entry import prompt_forecast
        prompt_forecast(args.ticker)
        return 0

    if args.cmd == "evaluate":
        from paper.entry import run_evaluate
        run_evaluate(args.forecast_id)
        return 0

    if args.cmd == "open":
        from paper.entry import open_trade
        open_trade(args.forecast_id, override=args.override, override_reason=args.override_reason)
        return 0

    if args.cmd == "mark":
        from paper.mark import run_mark
        run_mark(asof=args.asof)
        return 0

    if args.cmd == "report":
        from paper.score import report
        report()
        return 0

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
