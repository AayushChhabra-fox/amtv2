import json
from pathlib import Path

from amt.context import build_context
from amt.ledger import append
from amt.schemas import Brief, LedgerRow


def _row(index: int) -> LedgerRow:
    family = "linear" if index % 2 == 0 else "tree"
    return LedgerRow(
        run_id="r",
        node_id=f"n{index}",
        operator="draft",
        family=family,
        claim=f"claim {index}",
        rung="R1",
        seed=0,
        status="kept",
        git_sha="abc",
        metric=float(index),
    )


def test_context_keeps_newest_nodes_siblings_and_the_fix_chain(tmp_path: Path) -> None:
    for index in range(13):
        append(tmp_path, _row(index))
    state = {
        "run_id": "r",
        "round": 1,
        "nodes": [
            {
                "node_id": "alpha",
                "operator": "draft",
                "family": "linear",
                "status": "pending",
                "search_round": 1,
                "claim": "first draft",
            },
            {
                "node_id": "beta",
                "operator": "draft",
                "family": "tree",
                "status": "pending",
                "search_round": 1,
                "claim": "second draft",
            },
        ],
    }
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps(state), encoding="utf-8")
    log = tmp_path / "candidates" / "n12" / "debug_log.md"
    log.parent.mkdir(parents=True)
    log.write_text("# debug log\n\n## attempt 1\n\nboom\n", encoding="utf-8")
    brief = Brief.model_validate({"name": "nfy", "budget": {"total_dbu": 10}, "k": 2})

    debug = build_context(
        "implementer",
        brief,
        tmp_path,
        run_id="r",
        node_id="n12",
        operator="debug",
    )
    assert "n0 " not in debug
    assert "n12 " in debug
    assert "Best node per family" in debug
    assert "n12 " in debug.split("Best node per family", 1)[1]
    assert "boom" in debug
    assert "Siblings" not in debug

    draft = build_context(
        "planner",
        brief,
        tmp_path,
        run_id="r",
        node_id="alpha",
        operator="draft",
    )
    assert "Siblings" in draft
    assert "beta " in draft
    assert "second draft" in draft
    assert "simple" in draft
