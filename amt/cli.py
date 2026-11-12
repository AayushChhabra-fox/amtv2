"""Command line for the walking skeleton, the gates, and the scale-up runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from amt.envutil import load_env
from amt.ladder import build_ladder_report, ladder_jobs
from amt.ladder import load_cells as load_ladder_cells
from amt.noise import GateDataError, build_noise_report, round0_jobs
from amt.noise import load_cells as load_noise_cells
from amt.orchestrator import run_brief
from amt.scale import (
    full_budget_block_reason,
    g3_is_current,
    sign_g3,
    write_agent_stats,
)
from amt.schemas import RunState, load_brief
from amt.signoff import (
    SignoffError,
    g1_is_current,
    g2_is_current,
    search_block_reason,
    sign_g1,
    sign_g2,
    write_ladder_report,
    write_noise_report,
)
from amt.signoff import _resolve as resolve_under
from amt.signoff import _spec_path as spec_path
from amt.status import audit_rows, format_status
from harness.data.contract import load_spec


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
    run_parser.add_argument(
        "--scale",
        default=None,
        choices=["search", "calibration", "small", "full"],
        help="calibration is G3, small is the K=4 run, full records Phase 2 entry",
    )

    status_parser = sub.add_parser("status", help="print run status")
    status_parser.add_argument("--run-id", default="toy")
    status_parser.add_argument("--audit", action="store_true")

    stop_parser = sub.add_parser("stop", help="ask a run to stop at the next stage")
    stop_parser.add_argument("--run-id", default="toy")

    approve_parser = sub.add_parser("approve", help="unblock a run waiting on approval_mode")
    approve_parser.add_argument("--run-id", required=True)

    noise_parser = sub.add_parser("noise", help="write the round-0 noise report for gate G1")
    noise_parser.add_argument("--brief", type=Path, required=True)
    noise_parser.add_argument("--observations", type=Path)
    noise_parser.add_argument("--plan", action="store_true")
    noise_parser.add_argument("--root", type=Path, default=None)

    ladder_parser = sub.add_parser("ladder", help="write the ladder report for gate G2")
    ladder_parser.add_argument("--dataset", required=True)
    ladder_parser.add_argument("--observations", type=Path)
    ladder_parser.add_argument("--plan", action="store_true")
    ladder_parser.add_argument("--skip-r1", action="append", default=[])
    ladder_parser.add_argument("--r1-train-fraction", type=float, default=None)
    ladder_parser.add_argument("--root", type=Path, default=None)

    sign_parser = sub.add_parser("sign", help="sign gate G1 or G2")
    sign_parser.add_argument("gate", choices=["g1", "g2", "g3"])
    sign_parser.add_argument("--signer", required=True)
    sign_parser.add_argument("--brief", type=Path)
    sign_parser.add_argument("--dataset", default=None)
    sign_parser.add_argument("--manifest", type=Path)
    sign_parser.add_argument("--run-id", default=None)
    sign_parser.add_argument("--root", type=Path, default=None)

    stats_parser = sub.add_parser("stats", help="print agent statistics for a run")
    stats_parser.add_argument("--run-id", required=True)
    stats_parser.add_argument("--root", type=Path, default=None)

    gates_parser = sub.add_parser("gates", help="show whether G1 and G2 are signed")
    gates_parser.add_argument("--brief", type=Path, required=True)
    gates_parser.add_argument("--root", type=Path, default=None)

    args = parser.parse_args(argv)
    root = args.root if getattr(args, "root", None) else Path.cwd()
    if args.command == "noise":
        return _noise(root, args)
    if args.command == "ladder":
        return _ladder(root, args)
    if args.command == "sign":
        return _sign(root, args)
    if args.command == "gates":
        return _gates(root, args)
    if args.command == "stats":
        return _stats(root, args)
    if args.command == "run":
        state = run_brief(
            args.brief,
            root=root,
            fresh=args.fresh,
            run_id=args.run_id,
            approval_mode=args.approval_mode,
            scale=args.scale,
        )
        print(format_status(root, state.run_id, audit=state.stage == "REPORT"))
        return 0 if state.stage == "REPORT" else 1
    if args.command == "status":
        print(format_status(root, args.run_id, audit=args.audit))
        if args.audit and audit_rows(root, args.run_id):
            return 1
        return 0
    if args.command == "approve":
        approve_path = root / "runs" / args.run_id / "APPROVE"
        approve_path.parent.mkdir(parents=True, exist_ok=True)
        approve_path.write_text("approve\n", encoding="utf-8")
        print(f"wrote {approve_path}")
        return 0
    stop_path = root / "runs" / args.run_id / "STOP"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop\n")
    print(f"wrote {stop_path}")
    return 0


def _noise(root: Path, args: argparse.Namespace) -> int:
    brief_path = _under_root(root, args.brief)
    brief = load_brief(brief_path)
    try:
        spec = load_spec(spec_path(root, brief))
    except (OSError, SignoffError, ValueError) as exc:
        print(str(exc))
        return 1
    if args.plan:
        print(json.dumps([job.params() for job in round0_jobs(spec.reference_scores)], indent=2))
        if args.observations is None:
            return 0
    if args.observations is None:
        print("pass --observations or --plan")
        return 1
    try:
        cells = load_noise_cells(json.loads(_under_root(root, args.observations).read_text()))
        report = build_noise_report(
            cells,
            brief,
            dataset=spec.name,
            reference_scores=list(spec.reference_scores),
        )
    except (OSError, GateDataError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    path = write_noise_report(root, brief, brief_path, report)
    print(f"{report.status} {path}")
    return 0 if report.status == "pass" else 2


def _ladder(root: Path, args: argparse.Namespace) -> int:
    dataset, fraction = _dataset(root, args.dataset)
    if args.r1_train_fraction is not None:
        fraction = args.r1_train_fraction
    if args.plan:
        print(json.dumps([job.params() for job in ladder_jobs()], indent=2))
        if args.observations is None:
            return 0
    if args.observations is None:
        print("pass --observations or --plan")
        return 1
    try:
        cells = load_ladder_cells(json.loads(_under_root(root, args.observations).read_text()))
        report = build_ladder_report(
            cells,
            dataset=dataset,
            skip_r1=list(args.skip_r1),
            r1_train_fraction=fraction,
        )
    except (OSError, GateDataError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    path = write_ladder_report(root, report)
    print(f"{report.status} {path}")
    return 0 if report.status == "pass" else 2


def _sign(root: Path, args: argparse.Namespace) -> int:
    try:
        if args.gate == "g1":
            if args.brief is None or args.manifest is None:
                print("g1 requires --brief and --manifest")
                return 1
            record = sign_g1(
                root,
                _under_root(root, args.brief),
                args.signer,
                _under_root(root, args.manifest),
            )
        elif args.gate == "g3":
            if args.brief is None or not args.run_id:
                print("g3 requires --brief and --run-id")
                return 1
            record = sign_g3(
                root,
                _under_root(root, args.brief),
                args.signer,
                args.run_id,
            )
        else:
            if not args.dataset:
                print("g2 requires --dataset")
                return 1
            dataset, _fraction = _dataset(root, args.dataset)
            record = sign_g2(root, dataset, args.signer)
    except (OSError, SignoffError, ValueError) as exc:
        print(str(exc))
        return 1
    print(f"signed {record.gate} as {record.signer} at {record.signed_at}")
    return 0


def _gates(root: Path, args: argparse.Namespace) -> int:
    brief_path = _under_root(root, args.brief)
    brief = load_brief(brief_path)
    if brief.candidate is not None:
        print("walking skeleton; G1 and G2 are not required")
        return 0
    reason = search_block_reason(root, brief, brief_path)
    if reason:
        print(reason)
        return 1
    g1 = "current" if g1_is_current(root, brief_path) else "missing"
    dataset = brief.resolved_dataset_spec() or ""
    try:
        name, _fraction = _dataset(root, dataset)
    except (OSError, ValueError):
        name = dataset
    g2 = "current" if name and g2_is_current(root, name) else "missing"
    g3 = "current" if g3_is_current(root) else "missing"
    ready = full_budget_block_reason(root)
    full = "ready" if ready is None else "not ready"
    print(f"G1 {g1}; G2 {g2}; G3 {g3}; full budget {full}")
    return 0


def _stats(root: Path, args: argparse.Namespace) -> int:
    path = root / "runs" / args.run_id / "state.json"
    if not path.is_file():
        print(f"no state for {args.run_id}")
        return 1
    state = RunState.model_validate(json.loads(path.read_text(encoding="utf-8")))
    print(write_agent_stats(root, state).model_dump_json(indent=2))
    return 0


def _dataset(root: Path, text: str) -> tuple[str, float | None]:
    path = resolve_under(root, text)
    if path.exists():
        spec = load_spec(path)
        return spec.name, float(spec.sampling.rungs["R1"].train_fraction)
    return text, None


def _under_root(root: Path, path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    rooted = root / path
    if rooted.exists():
        return rooted
    return path
