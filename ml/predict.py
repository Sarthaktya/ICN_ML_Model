import argparse
import json
import math
import numbers
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.common import DEFAULT_MODEL_DIR, METADATA_FILE, MODEL_FILE, sha256_file


class InputValidationError(ValueError):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems))


class ModelIntegrityError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load(model_dir):
    d = Path(model_dir)
    meta = json.loads((d / METADATA_FILE).read_text(encoding="utf-8"))
    path = d / MODEL_FILE
    if sha256_file(path) != meta["artifact_sha256"]:
        raise ModelIntegrityError(f"{path} does not match the hash in {METADATA_FILE}; refusing to load")
    if meta["packages"]["scikit-learn"] != sklearn.__version__:
        warnings.warn(
            f"model trained with scikit-learn {meta['packages']['scikit-learn']}, running {sklearn.__version__}"
        )
    return joblib.load(path), meta


def _validate(record, i, schema):
    tag = f"record[{i}]"
    if not isinstance(record, dict):
        return [f"{tag}: must be an object mapping feature name to value"]
    spec = {f["name"]: f for f in schema["numeric_features"] + schema["categorical_features"]}
    problems = []
    missing = sorted(set(spec) - set(record))
    extra = sorted(set(record) - set(spec))
    if missing:
        problems.append(f"{tag}: missing fields {missing}")
    if extra:
        problems.append(f"{tag}: unknown fields {extra}")
    for name, f in spec.items():
        if name not in record:
            continue
        v = record[name]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            problems.append(f"{tag}.{name}: must be a number, got {type(v).__name__}")
        elif not math.isfinite(v):
            problems.append(f"{tag}.{name}: must be finite, got {v}")
        elif "allowed" in f:
            if v not in f["allowed"]:
                problems.append(f"{tag}.{name}: unknown category {v!r}, allowed {f['allowed']}")
        else:
            if v < f["min"]:
                problems.append(f"{tag}.{name}: {v} is below minimum {f['min']}")
            if f["dtype"] == "int" and v != int(v):
                problems.append(f"{tag}.{name}: must be an integer, got {v}")
    return problems


def predict_batch(records, model_dir=DEFAULT_MODEL_DIR):
    pipe, meta = _load(str(model_dir))
    if not isinstance(records, list) or not records:
        raise InputValidationError(["records must be a non-empty list of objects"])
    problems = [p for i, r in enumerate(records) for p in _validate(r, i, meta["feature_schema"])]
    if problems:
        raise InputValidationError(problems)
    X = pd.DataFrame(records)[meta["features"]]
    proba = pipe.predict_proba(X)
    classes = [str(c) for c in pipe.classes_]
    benign = meta["benign_label"]
    results = []
    for row in proba:
        label = classes[int(row.argmax())]
        results.append(
            {
                "model_version": meta["model_version"],
                "feature_schema_version": meta["feature_schema_version"],
                "predicted_class": label,
                "is_attack": label != benign,
                "attack_score": round(float(1 - row[classes.index(benign)]), 6),
                "probabilities": {c: round(float(p), 6) for c, p in zip(classes, row)},
                "probabilities_calibrated": False,
            }
        )
    return results


def predict(record, model_dir=DEFAULT_MODEL_DIR):
    return predict_batch([record], model_dir)[0]


def main():
    ap = argparse.ArgumentParser(prog="python -m ml.predict")
    ap.add_argument("--input", default="-", help="JSON file with one flow object or a list of them; - for stdin")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    a = ap.parse_args()
    raw = sys.stdin.read() if a.input == "-" else Path(a.input).read_text(encoding="utf-8")
    data = json.loads(raw)
    try:
        out = predict_batch(data if isinstance(data, list) else [data], a.model_dir)
    except InputValidationError as e:
        print(json.dumps({"error": "invalid_input", "problems": e.problems}, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()