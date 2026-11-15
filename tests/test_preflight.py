from pathlib import Path

from amt.agents import MODEL_PY
from amt.preflight import preflight_candidate, static_check


def test_ranker_passes_static_and_smoke(tmp_path: Path) -> None:
    directory = tmp_path / "candidates" / "c01"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(MODEL_PY, encoding="utf-8")
    smoke = preflight_candidate(directory)
    assert smoke["rows"] == 120
    assert smoke["finite"] == 1.0
    assert smoke["deterministic"] == 1.0
    assert "class Ranker" in MODEL_PY
    assert "LogisticRanker" not in MODEL_PY


def test_static_rejects_hub_downloads_without_a_volume_cache(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "candidates" / "hub"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(
        "def load():\n"
        "    model.from_pretrained('org/name')\n"
        "    hf_hub_download('org/name', 'weights.bin')\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AMT_HF_OFFLINE_CACHE", raising=False)
    denied = "\n".join(static_check(directory))
    assert "from_pretrained" in denied
    assert "hf_hub_download" in denied
    monkeypatch.setenv("AMT_HF_OFFLINE_CACHE", "/tmp/hf")
    still = "\n".join(static_check(directory))
    assert "from_pretrained" in still
    monkeypatch.setenv("AMT_HF_OFFLINE_CACHE", "/Volumes/catalog/schema/amt_artifacts/hf")
    assert static_check(directory) == []


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
