import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from ml.common import (
    DEFAULT_MODEL_DIR,
    METADATA_FILE,
    MODEL_FILE,
    feature_names,
    load_schema,
    sha256_file,
    sha256_path
)

PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}
LEAK_THRESHOLD = 0.95


def parse(argv):
    p = argparse.ArgumentParser(prog="python -m ml.train")
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--data", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dataset-version", default="unspecified")
    p.add_argument("--licence", default="unspecified")
    p.add_argument("--source-url", default="unspecified")
    p.add_argument("--label-col", default="Label")
    p.add_argument("--benign-label", default="BENIGN")
    p.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    p.add_argument("--split", choices=["stratified", "group", "time"], default="stratified")
    p.add_argument("--group-col")
    p.add_argument("--time-col")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--column-map")
    p.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--plumbing-only", action="store_true")
    a = p.parse_args(argv)
    if a.split == "group" and not a.group_col:
        p.error("--split group needs --group-col")
    if a.split == "time" and not a.time_col:
        p.error("--split time needs --time-col")
    if a.sample_frac < 1 and not a.plumbing_only:
        p.error("--sample-frac is for dry runs; add --plumbing-only")
    return a


def load(a, schema):
    cols = {f["name"]: f["source_column"] for f in schema["numeric_features"] + schema["categorical_features"]}
    if a.column_map:
        with open(a.column_map, encoding="utf-8") as f:
            cols.update(json.load(f))
    needed = list(cols.values()) + [a.label_col] + [c for c in (a.group_col, a.time_col) if c]
    path = Path(a.data)
    if path.is_dir():
        df = pd.concat([pd.read_parquet(p, columns=needed) for p in sorted(path.rglob("*.parquet"))], ignore_index=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=needed)
    else:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
    if a.sample_frac < 1:
        df = df.sample(frac=a.sample_frac, random_state=a.seed).reset_index(drop=True)
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(f"columns not found in data: {absent}")
    X = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})[feature_names(schema)]
    y = df[a.label_col].astype(str).str.strip().str.replace(r"[^\x20-\x7E]+", "-", regex=True)
    groups = df[a.group_col].astype(str) if a.group_col else None
    times = pd.to_datetime(df[a.time_col], errors="coerce") if a.time_col else None
    return X, y, groups, times


def sub(s, mask):
    return None if s is None else s[mask]


def clean(X, y, groups, times, schema, a):
    stats = {"rows_raw": int(len(X))}
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    if times is not None:
        keep &= times.notna()
    stats["dropped_missing_or_non_finite"] = int((~keep).sum())

    bad = pd.Series(False, index=X.index)
    for f in schema["numeric_features"]:
        bad |= X[f["name"]] < f["min"]
    bad &= keep
    stats["dropped_below_min"] = int(bad.sum())
    keep &= ~bad

    bad = pd.Series(False, index=X.index)
    for f in schema["numeric_features"]:
        if f["dtype"] == "int":
            bad |= (X[f["name"]] % 1) != 0
    bad &= keep
    stats["dropped_non_integer"] = int(bad.sum())
    keep &= ~bad

    ok_cat = pd.Series(True, index=X.index)
    for f in schema["categorical_features"]:
        ok_cat &= X[f["name"]].isin(f["allowed"])
    bad = keep & ~ok_cat
    stats["dropped_unknown_category"] = int(bad.sum())
    keep &= ok_cat

    X, y, groups, times = X[keep], y[keep], sub(groups, keep), sub(times, keep)
    for f in schema["numeric_features"] + schema["categorical_features"]:
        if f["dtype"] == "int":
            X[f["name"]] = X[f["name"]].astype("int64")

    if a.task == "binary":
        y = y.where(y == a.benign_label, "ATTACK")

    dup = ~X.assign(_label=y).duplicated()
    stats["dropped_duplicates"] = int((~dup).sum())
    X, y, groups, times = X[dup], y[dup], sub(groups, dup), sub(times, dup)

    counts = y.value_counts()
    rare = counts[counts < a.min_class_count].index
    stats["dropped_rare_classes"] = {str(c): int(counts[c]) for c in rare}
    keep = ~y.isin(rare)
    X, y, groups, times = X[keep], y[keep], sub(groups, keep), sub(times, keep)

    if a.benign_label not in set(y):
        sys.exit(f"benign label {a.benign_label!r} not present after cleaning; check --benign-label")
    stats["rows_clean"] = int(len(X))
    stats["class_counts"] = {str(k): int(v) for k, v in y.value_counts().items()}
    return (
        X.reset_index(drop=True),
        y.to_numpy(),
        None if groups is None else groups.to_numpy(),
        None if times is None else times.to_numpy(),
        stats,
    )


def take(arr, idx):
    return None if arr is None else arr[idx]


def split(y, groups, times, method, size, seed):
    idx = np.arange(len(y))
    if method == "stratified":
        a, b = train_test_split(idx, test_size=size, stratify=y, random_state=seed)
    elif method == "group":
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(idx, groups=groups))
    else:
        order = np.argsort(times, kind="stable")
        cut = int(len(order) * (1 - size))
        a, b = order[:cut], order[cut:]
    return np.sort(a), np.sort(b)


def build_pipeline(schema, seed):
    num = [f["name"] for f in schema["numeric_features"]]
    cat = schema["categorical_features"]
    prep = ColumnTransformer(
        [
            ("num", "passthrough", num),
            (
                "cat",
                OneHotEncoder(categories=[f["allowed"] for f in cat], handle_unknown="error"),
                [f["name"] for f in cat],
            ),
        ]
    )
    return Pipeline([("prep", prep), ("rf", RandomForestClassifier(random_state=seed, n_jobs=-1, **PARAMS))])


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0, c - h)), float(min(1, c + h))


def metrics(y_true, y_pred, classes, benign):
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    t, p = y_true != benign, y_pred != benign
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    fp, tn = int((~t & p).sum()), int((~t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    fpr, lo, hi = wilson(fp, fp + tn) or (None, None, None)
    return {
        "n": int(len(y_true)),
        "classes": classes,
        "per_class": {
            c: {
                "precision": float(rep[c]["precision"]),
                "recall": float(rep[c]["recall"]),
                "f1": float(rep[c]["f1-score"]),
                "support": int(rep[c]["support"]),
            }
            for c in classes
        },
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "attack_vs_benign": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "fpr_ci95": [lo, hi],
            "false_alarms_per_100k_benign": None if fpr is None else fpr * 1e5,
        },
        "confusion_matrix": cm.tolist(),
    }


def single_feature_scores(Xtr, ytr, Xva, yva):
    out = {}
    for c in Xtr.columns:
        t = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(Xtr[[c]], ytr)
        out[c] = float(balanced_accuracy_score(yva, t.predict(Xva[[c]])))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def row_overlap(Xa, Xb):
    ha = set(pd.util.hash_pandas_object(Xa, index=False).to_numpy())
    hb = pd.util.hash_pandas_object(Xb, index=False).to_numpy()
    return int(sum(h in ha for h in hb))


def plot_cm(cm, classes, path, title):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * n, 1.2 + 0.8 * n), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "#1a1a1a", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (shade = share of the actual class)")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pct(v):
    return "n/a" if v is None else f"{100 * v:.2f}%"


def report_md(a, meta, res, stats, leak, overlap, importances, notes):
    t = res["test"]
    b = t["attack_vs_benign"]
    L = []
    if a.plumbing_only:
        L += ["> **PLUMBING TEST ONLY (synthetic or sampled data). These numbers are not a measured result.**", ""]
    L += [
        "# Evaluation report", "",
        f"- Model: `{meta['model_version']}`  |  trained {meta['trained_at_utc']}",
        f"- Dataset: {a.dataset_name} ({a.dataset_version}), licence: {a.licence}, source: {a.source_url}",
        f"- Task: {a.task}  |  split: {a.split}  |  seed: {a.seed}",
        f"- Rows train/val/test: {meta['split']['rows_train']} / {meta['split']['rows_val']} / {meta['split']['rows_test']}", "",
        "## Cleaning", "",
        "| Step | Rows |", "|---|---|",
    ]
    for k, v in stats.items():
        if k not in ("class_counts", "dropped_rare_classes"):
            L.append(f"| {k} | {v} |")
    L += [f"| dropped_rare_classes | {stats['dropped_rare_classes'] or 'none'} |", "",
          f"## Held-out TEST: attack vs benign ({a.dataset_name}, {a.split} split)", "",
          "| Precision | Recall | F1 | FPR | FPR 95% CI | Benign flows | False positives |", "|---|---|---|---|---|---|---|",
          f"| {pct(b['precision'])} | {pct(b['recall'])} | {pct(b['f1'])} | {pct(b['fpr'])} | "
          f"{pct(b['fpr_ci95'][0])} - {pct(b['fpr_ci95'][1])} | {b['fp'] + b['tn']} | {b['fp']} |", ""]
    if b["false_alarms_per_100k_benign"] is not None:
        L += [f"At this FPR the model would raise about {b['false_alarms_per_100k_benign']:.0f} false alarms per 100,000 benign flows.", ""]
    L += ["## Per-class (TEST)", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, m in t["per_class"].items():
        L.append(f"| {c} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['support']} |")
    L += ["", f"Macro F1 (test): {pct(t['macro_f1'])}  |  Macro F1 (validation): {pct(res['validation']['macro_f1'])}", "",
          "![confusion matrix](confusion_matrix_test.png)", "",
          "## Leakage checks", "",
          f"- Identical feature rows in both train and test: {overlap}",
          f"- Best single-feature balanced accuracy on validation (depth-3 tree), flag if > {LEAK_THRESHOLD}:"]
    for k, v in list(leak.items())[:5]:
        L.append(f"  - {k}: {v:.3f}{'  **FLAG**' if v > LEAK_THRESHOLD else ''}")
    L += ["", "## Top feature importances (impurity-based, unreliable for correlated features)", ""]
    for k, v in importances[:8]:
        L.append(f"- {k}: {v:.3f}")
    L += ["", "## Operational meaning", "",
          "- False positive: a legitimate source gets blocked. Cost is lost availability and a locked-out user.",
          "- False negative: an attack passes unnoticed. Cost is the damage the attack was meant to do.",
          "- Blocking thresholds, whitelists and TTLs belong to the policy engine (Laptop 3), not this model.", "",
          "## Caveats", ""]
    L += [f"- {n}" for n in notes]
    L += ["- Hyperparameters are fixed defaults; no tuning was done against the test set.",
          "- Random Forest probabilities are not calibrated."]
    return "\n".join(L) + "\n"


def main(argv=None):
    a = parse(argv)
    schema = load_schema()
    names = feature_names(schema)
    X, y, groups, times = load(a, schema)
    X, y, groups, times, stats = clean(X, y, groups, times, schema, a)

    tr, te = split(y, groups, times, a.split, a.test_size, a.seed)
    tr2, va = split(y[tr], take(groups, tr), take(times, tr), a.split, a.val_size, a.seed)
    tr_idx, va_idx = tr[tr2], tr[va]

    classes_all = sorted(set(y))
    notes = []
    for c in classes_all:
        if c not in set(y[tr_idx]):
            sys.exit(f"class {c!r} has no rows in the training split; change the split or drop the class")
        if c not in set(y[te]):
            notes.append(f"class {c!r} is absent from the test split; its test metrics are meaningless")
    if a.split == "stratified":
        notes.append("Stratified row-level split: flows from the same session or attacker can land on both sides, so scores are optimistic. Prefer --split group or time when the CSV has IP or timestamp columns.")

    pipe = build_pipeline(schema, a.seed)
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    classes = [str(c) for c in pipe.classes_]

    res = {
        "validation": metrics(y[va_idx], pipe.predict(X.iloc[va_idx]), classes, a.benign_label),
        "test": metrics(y[te], pipe.predict(X.iloc[te]), classes, a.benign_label),
    }

    leak = single_feature_scores(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx])
    overlap = row_overlap(X.iloc[tr_idx], X.iloc[te])
    imp = sorted(
        zip(pipe.named_steps["prep"].get_feature_names_out(), pipe.named_steps["rf"].feature_importances_),
        key=lambda kv: -kv[1],
    )
    imp = [(k, float(v)) for k, v in imp]

    out = Path(a.out_dir)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    model_path = out / MODEL_FILE
    joblib.dump(pipe, model_path)

    data_sha = sha256_path(a.data)
    version = "rf-" + sha256(
    json.dumps([data_sha, PARAMS, a.seed, schema["schema_version"], a.split, a.task, a.sample_frac, a.min_class_count], sort_keys=True).encode()
    ).hexdigest()[:10]

    meta = {
        "model_version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_schema_version": schema["schema_version"],
        "feature_schema": schema,
        "features": names,
        "classes": classes,
        "label_mapping": {c: i for i, c in enumerate(classes)},
        "benign_label": a.benign_label,
        "task": a.task,
        "params": {**PARAMS, "random_state": a.seed},
        "seed": a.seed,
        "plumbing_only": a.plumbing_only,
        "dataset": {
            "name": a.dataset_name, "version": a.dataset_version, "licence": a.licence,
            "source_url": a.source_url, "file_sha256": data_sha,
        },
        "cleaning": stats,
        "split": {
            "method": a.split, "group_col": a.group_col, "time_col": a.time_col,
            "test_size": a.test_size, "val_size": a.val_size,
            "rows_train": int(len(tr_idx)), "rows_val": int(len(va_idx)), "rows_test": int(len(te)),
        },
        "packages": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "artifact_sha256": sha256_file(model_path),
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "eval" / "metrics.json").write_text(json.dumps({"validation": res["validation"], "test": res["test"], "leakage_single_feature": leak, "train_test_identical_rows": overlap}, indent=2), encoding="utf-8")
    plot_cm(res["test"]["confusion_matrix"], classes, out / "eval" / "confusion_matrix_test.png",
            "Test confusion matrix" + (" (SYNTHETIC)" if a.plumbing_only else ""))
    (out / "eval" / "report.md").write_text(report_md(a, meta, res, stats, leak, overlap, imp, notes), encoding="utf-8")
    print(f"saved {model_path} ({version})")
    print(f"report: {out / 'eval' / 'report.md'}")
    return meta


if __name__ == "__main__":
    main()