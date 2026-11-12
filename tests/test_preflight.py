from pathlib import Path

from amt.agents import MODEL_PY
from amt.preflight import preflight_candidate, static_check


def test_logistic_candidate_passes_static_and_smoke(tmp_path: Path) -> None:
    directory = tmp_path / "candidates" / "c00"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(MODEL_PY, encoding="utf-8")
    smoke = preflight_candidate(directory)
    assert smoke["log_loss"] < smoke["baseline_log_loss"]


def test_static_rejects_spark_holdout_network_and_outside_paths(tmp_path: Path) -> None:
    directory = tmp_path / "candidates" / "bad"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(
        "import pyspark\n"
        "import subprocess\n"
        "TABLE = 'pnr_prod.silver.news_impressions'\n"
        "SPLIT = 'select_holdout'\n"
        "open('/tmp/leak')\n"
        "spark.table(TABLE)\n",
        encoding="utf-8",
    )
    violations = static_check(directory)
    text = "\n".join(violations)
    assert "pyspark" in text
    assert "subprocess" in text
    assert "select_holdout" in text
    assert "pnr_prod.silver.news_impressions" in text
    assert "/tmp/leak" in text
    assert "spark" in text
