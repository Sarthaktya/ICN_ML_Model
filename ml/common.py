import hashlib
import json
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
ROOT = ML_DIR.parent
SCHEMA_PATH = ML_DIR / "feature_schema.json"
DEFAULT_MODEL_DIR = ROOT / "models"
MODEL_FILE = "classifier.joblib"
METADATA_FILE = "metadata.json"


def load_schema(path=SCHEMA_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_names(schema):
    return [f["name"] for f in schema["numeric_features"]] + [
        f["name"] for f in schema["categorical_features"]
    ]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_path(path):
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(f.relative_to(p).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()