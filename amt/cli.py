"""Command line for the walking skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from amt.envutil import load_env
from amt.orchestrator import run_brief
from amt.status import format_status


def main(argv: list[str] | None = None) -> int:
    load_env(Path.cwd())
    parser = argparse.ArgumentParser(prog="amt")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a brief until the ledger row is written")
    run_parser.add_argument("brief", type=Path)
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument(
        "--approval-mode",
        default=None,
        choices=["none", "every_stage", "round_shortlist"],
    )

    status_parser = sub.add_parser("status", help="print run status")
    status_parser.add_argument("--run-id", default="toy")
    status_parser.add_argument("--audit", action="store_true")

    stop_parser = sub.add_parser("stop", help="ask a run to stop at the next stage")
    stop_parser.add_argument("--run-id", default="toy")

    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "run":
        state = run_brief(
            args.brief,
            root=root,
            fresh=args.fresh,
            run_id=args.run_id,
            approval_mode=args.approval_mode,
        )
        print(format_status(root, state.run_id, audit=state.stage == "REPORT"))
        return 0 if state.stage == "REPORT" else 1
    if args.command == "status":
        print(format_status(root, args.run_id, audit=args.audit))
        return 0
    stop_path = root / "runs" / args.run_id / "STOP"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop\n")
    print(f"wrote {stop_path}")
    return 0
