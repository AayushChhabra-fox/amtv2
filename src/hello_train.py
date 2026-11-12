"""Phase 0 dummy training task.

Logs one metric and the AMT tag set. The score is a placeholder so the
orchestrator can prove Jobs polling and MLflow search. It is not a model.
"""

from __future__ import annotations

import argparse

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_id", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--git_sha", required=True)
    parser.add_argument("--dbu_est", required=True)
    parser.add_argument("--amt_run_id", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--parent_id", default="")
    parser.add_argument("--operator", default="draft")
    parser.add_argument("--split", default="toy")
    parser.add_argument("--node_id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as active:
        # Keys match amt.schemas.MLFLOW_TAG_KEYS. This file runs on serverless
        # without the amt package, so the convention is inlined here.
        node_id = args.node_id or args.candidate_id
        mlflow.set_tags(
            {
                "run_id": args.amt_run_id,
                "node_id": node_id,
                "parent_id": args.parent_id,
                "family": args.family,
                "operator": args.operator,
                "rung": args.rung,
                "seed": args.seed,
                "batch": args.batch,
                "split": args.split,
                "git_sha": args.git_sha,
                "candidate_id": args.candidate_id,
                "dbu_est": args.dbu_est,
            }
        )
        mlflow.log_metric("ndcg_at_10", 0.5)
        print(f"MLFLOW_RUN_ID={active.info.run_id}")


if __name__ == "__main__":
    main()
