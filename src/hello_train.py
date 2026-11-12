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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as active:
        mlflow.set_tags(
            {
                "candidate_id": args.candidate_id,
                "family": args.family,
                "rung": args.rung,
                "batch": args.batch,
                "git_sha": args.git_sha,
                "dbu_est": args.dbu_est,
                "seed": args.seed,
                "run_id": args.amt_run_id,
                "node_id": args.candidate_id,
                "operator": "draft",
                "split": "toy",
            }
        )
        mlflow.log_metric("ndcg_at_10", 0.5)
        print(f"MLFLOW_RUN_ID={active.info.run_id}")


if __name__ == "__main__":
    main()
