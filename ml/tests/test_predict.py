import json
import shutil
import subprocess
import sys

import pytest

from ml import predict as P
from ml.common import ML_DIR, ROOT
from ml.tests.synthetic import make_csv
from ml.train import main as train_main

FIXTURES = ML_DIR / "fixtures"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "synthetic.csv"
    make_csv(csv)
    out = d / "models"
    train_main(
        ["--data", str(csv), "--dataset-name", "SYNTHETIC", "--plumbing-only",
         "--split", "group", "--group-col", "Source IP", "--out-dir", str(out)]
    )
    return out


def valid():
    return json.loads((FIXTURES / "valid_flow.json").read_text())


def test_cleaning_ran(model_dir):
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["cleaning"]["dropped_missing_or_non_finite"] >= 1
    assert meta["cleaning"]["dropped_below_min"] >= 1
    assert meta["cleaning"]["dropped_unknown_category"] >= 1
    assert meta["cleaning"]["dropped_duplicates"] >= 1


def test_valid_single(model_dir):
    out = P.predict(valid(), model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert set(out) == {
        "model_version", "feature_schema_version", "predicted_class", "is_attack",
        "attack_score", "probabilities", "probabilities_calibrated",
    }
    assert out["model_version"] == meta["model_version"]
    assert out["predicted_class"] in meta["classes"]
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-4


def test_key_order_does_not_matter(model_dir):
    r = valid()
    assert P.predict(r, model_dir) == P.predict(dict(reversed(list(r.items()))), model_dir)


def test_batch(model_dir):
    batch = json.loads((FIXTURES / "batch.json").read_text())
    out = P.predict_batch(batch, model_dir)
    assert len(out) == len(batch)
    assert out[0] == P.predict(batch[0], model_dir)


def test_missing_field(model_dir):
    r = valid()
    del r["protocol"]
    with pytest.raises(P.InputValidationError, match="missing fields"):
        P.predict(r, model_dir)


def test_unknown_field(model_dir):
    r = valid()
    r["dst_port"] = 80
    with pytest.raises(P.InputValidationError, match="unknown fields"):
        P.predict(r, model_dir)


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("flow_duration_us", "5", "must be a number"),
        ("flow_duration_us", True, "must be a number"),
        ("flow_duration_us", float("nan"), "finite"),
        ("flow_bytes_per_s", float("inf"), "finite"),
        ("total_fwd_packets", -1, "below minimum"),
        ("total_fwd_packets", 2.5, "integer"),
        ("protocol", 99, "unknown category"),
    ],
)
def test_bad_value(model_dir, field, value, fragment):
    r = valid()
    r[field] = value
    with pytest.raises(P.InputValidationError, match=fragment):
        P.predict(r, model_dir)


@pytest.mark.parametrize("bad", [[], {}, "x", [[1, 2, 3]]])
def test_bad_batch(model_dir, bad):
    with pytest.raises(P.InputValidationError):
        P.predict_batch(bad, model_dir)


def test_tampered_artifact_rejected(model_dir, tmp_path):
    bad = tmp_path / "models"
    shutil.copytree(model_dir, bad)
    with open(bad / "classifier.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(P.ModelIntegrityError):
        P.predict(valid(), bad)


def test_fresh_process_matches(model_dir):
    expected = P.predict(valid(), model_dir)
    r = subprocess.run(
        [sys.executable, "-m", "ml.predict", "--input", str(FIXTURES / "valid_flow.json"),
         "--model-dir", str(model_dir)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)[0] == expected


def test_cli_rejects_invalid(model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "ml.predict", "--input", str(FIXTURES / "invalid_missing_field.json"),
         "--model-dir", str(model_dir)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "missing fields" in r.stderr