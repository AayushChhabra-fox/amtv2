"""Pinned candidate requirements: parse, constraints, preflight, and train."""

import sys
from pathlib import Path

import pytest

from amt.agents import MODEL_PY
from amt.preflight import static_check
from harness.deps import (
    RequirementsError,
    constraints_text,
    parse_requirements,
    requirements_hash,
)
from harness.train import main as train_main


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_accepts_exact_pins(tmp_path: Path) -> None:
    path = _write(tmp_path / "requirements.txt", "# ranker\ncatboost==1.2.5\n\nimplicit==0.7.2\n")
    pins = parse_requirements(path)
    assert [(item.name, item.version) for item in pins] == [
        ("catboost", "1.2.5"),
        ("implicit", "0.7.2"),
    ]
    assert len(requirements_hash(pins)) == 16


@pytest.mark.parametrize(
    "text",
    [
        "catboost>=1.2",
        "git+https://github.com/example/ranker.git",
        "https://example.com/pkg.whl",
        "-e .",
        "-r other.txt",
        "catboost==1.2 --index-url https://pypi.org/simple",
        "numpy==1.26.4",
        "scikit-learn==1.4.0",
        "requests==2.32.3",
        "databricks-sdk==0.40.0",
        "catboost==1.2.5\ncatboost==1.2.6\n",
        "\n".join(f"pkg{index}==1.0.0" for index in range(11)),
    ],
)
def test_parse_rejects_unpinned_and_base_packages(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path / "requirements.txt", text + "\n")
    with pytest.raises(RequirementsError):
        parse_requirements(path)


def test_constraints_pin_the_installed_base() -> None:
    text = constraints_text()
    assert "numpy==" in text


def test_static_check_allows_a_declared_import(tmp_path: Path) -> None:
    directory = tmp_path / "candidates" / "c00"
    _write(
        directory / "model.py",
        "import catboost\nimport numpy\n",
    )
    denied = "\n".join(static_check(directory))
    assert "catboost" in denied
    allowed = static_check(directory, extra_imports=frozenset({"catboost"}))
    assert allowed == []


def test_preflight_installs_declared_requirements_before_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "candidates" / "c00"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(MODEL_PY, encoding="utf-8")
    (directory / "requirements.txt").write_text("catboost==1.2.5\n", encoding="utf-8")
    calls: list[str] = []

    def fake_install(candidate_dir: Path, target_root: Path) -> Path:
        calls.append(str(candidate_dir))
        dest = target_root / "env"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr("amt.preflight.install", fake_install)
    monkeypatch.setattr("amt.preflight.import_names", lambda target: frozenset())
    monkeypatch.setenv("AMT_ENV_ROOT", str(tmp_path / "envs"))
    from amt.preflight import preflight_candidate

    smoke = preflight_candidate(directory)
    assert calls == [str(directory)]
    assert smoke["log_loss"] < smoke["baseline_log_loss"]


def test_train_installs_before_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "candidates" / "c00" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "class Ranker:\n"
        "    def fit(self, train, valid, ctx):\n"
        "        return None\n"
        "    def score(self, frame):\n"
        "        import numpy as np\n"
        "        return np.zeros(len(frame))\n"
        "    def save(self, path):\n"
        "        return None\n"
        "    @classmethod\n"
        "    def load(cls, path):\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    (model.parent / "requirements.txt").write_text("catboost==1.2.5\n", encoding="utf-8")
    seen: list[Path] = []

    def fake_activate(model_path: Path) -> tuple[str, float]:
        seen.append(model_path)
        assert "catboost" not in sys.modules
        return "abc123abc123abcd", 0.25

    monkeypatch.setattr("harness.train.activate_candidate_requirements", fake_activate)
    monkeypatch.setattr(
        "harness.train.make_ranker",
        lambda baseline, model_path, reference: _Tiny(),
    )
    monkeypatch.setattr(
        "harness.train.load_dataset",
        lambda dataset_dir: (
            {"samples": {}},
            {"train": [], "search_val": []},
        ),
    )

    class _Frame:
        def __len__(self) -> int:
            return 0

        rows: list = []

    def fake_frames(*args, **kwargs):
        return _Frame(), _Frame()

    monkeypatch.setattr("harness.train.frames_for_rung", fake_frames)
    monkeypatch.setattr(
        "harness.train.train_ranker",
        lambda **kwargs: __import__("harness.train", fromlist=["TrainResult"]).TrainResult(
            status="ok", metrics={"ndcg@10_graded": 0.4}, tags=dict(kwargs["tags"])
        ),
    )
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text("", encoding="utf-8")
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    spec = tmp_path / "dataset.yaml"
    spec.write_text("name: toy\noutput:\n  catalog: c\n  schema: s\n", encoding="utf-8")
    class _Output:
        catalog = "c"
        schema_name = "s"

    class _Spec:
        name = "toy"
        output = _Output()

    monkeypatch.setattr("harness.train.load_spec", lambda path: _Spec())
    code = train_main(
        [
            "--candidate_id",
            "c00",
            "--rung",
            "R1",
            "--seed",
            "0",
            "--split",
            "search_val",
            "--batch",
            "b",
            "--spec",
            str(spec),
            "--dataset-dir",
            str(dataset),
            "--model",
            str(model),
            "--run-id",
            "r",
            "--family",
            "linear",
            "--git-sha",
            "abc",
        ]
    )
    assert code is None
    assert seen == [model]


class _Tiny:
    pass
