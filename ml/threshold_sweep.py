import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIXED = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
ALARM_TARGETS = (25, 50, 100, 200, 500, 1000)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def precision_at(tpr, fpr, prev):
    tp, fp = tpr * prev, fpr * (1 - prev)
    return tp / (tp + fp) if tp + fp else float("nan")


def threshold_for(score, is_attack, alarms_per_100k):
    benign = np.sort(score[~is_attack])[::-1]
    k = int(len(benign) * alarms_per_100k / 1e5)
    t = benign[min(k, len(benign) - 1)]
    return float(min(1.0, np.ceil(t * 1e4) / 1e4))


def flags(score, pred, benign, t):
    return pred != benign if t is None else score > t


def operating_points(score_va, y_va, benign):
    is_attack = y_va != benign
    ops = [("argmax rule", None)]
    ops += [(f"score > {t}", t) for t in FIXED]
    ops += [(f"<= {n} alarms/100k (val)", threshold_for(score_va, is_attack, n)) for n in ALARM_TARGETS]
    return ops


def table(score, pred, y, benign, ops):
    is_attack = y != benign
    n_att = int(is_attack.sum())
    n_ben = len(is_attack) - n_att
    rows = []
    for name, t in ops:
        flag = flags(score, pred, benign, t)
        tp = int((flag & is_attack).sum())
        fp = int((flag & ~is_attack).sum())
        tpr, fpr = tp / n_att, fp / n_ben
        lo, hi = wilson(fp, n_ben)
        rows.append({
            "rule": name,
            "threshold": t,
            "attack_recall": tpr,
            "false_positives": fp,
            "alarms_per_100k_benign": fpr * 1e5,
            "fpr_ci_low_per_100k": lo * 1e5,
            "fpr_ci_high_per_100k": hi * 1e5,
            "precision_1pct_attacks": precision_at(tpr, fpr, 0.01),
            "precision_0.1pct_attacks": precision_at(tpr, fpr, 0.001),
            "false_alarms_per_real_attack_0.1pct": fpr * 0.999 / (tpr * 0.001) if tpr else float("nan"),
        })
    return pd.DataFrame(rows)


def per_class(score, pred, y, benign, ops):
    out = {}
    for name, t in ops:
        flag = flags(score, pred, benign, t)
        out[name] = {c: float((flag & (y == c)).sum() / (y == c).sum()) for c in sorted(set(y) - {benign})}
    return pd.DataFrame(out)[[n for n, _ in ops]]


def md(df, pct_cols=(), ndigits=2):
    d = df.copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: f"{100 * v:.2f}%")
        elif d[c].dtype.kind == "f":
            n = 4 if c == "threshold" else ndigits
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{n}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in d.itertuples(index=False))


def plot(score_va, y_va, benign, val_table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    is_attack = y_va != benign
    fpr, tpr, _ = roc_curve(is_attack, score_va)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    keep = fpr > 0
    a1.plot(fpr[keep] * 1e5, tpr[keep], color="tab:blue")
    pts = val_table[val_table.threshold.notna()]
    a1.scatter(pts.alarms_per_100k_benign.clip(lower=1), pts.attack_recall, color="tab:red", zorder=3)
    for r in pts.itertuples():
        a1.annotate(f"{r.threshold:g}", (max(r.alarms_per_100k_benign, 1), r.attack_recall), textcoords="offset points", xytext=(4, -10), fontsize=8)
    a1.set_xscale("log")
    a1.set_xlabel("false alarms per 100,000 benign flows (validation)")
    a1.set_ylabel("share of attack flows with attack_score above threshold")
    a1.set_title("Recall against false alarms; labels are thresholds")
    a1.grid(alpha=0.3)
    grid = np.linspace(0, 0.99, 100)
    for c in sorted(set(y_va) - {benign}):
        s = score_va[y_va == c]
        a2.plot(grid, [(s > t).mean() for t in grid], label=c)
    a2.set_xlabel("threshold on attack_score")
    a2.set_ylabel("recall for the class (validation)")
    a2.set_title("Per-class recall as the threshold rises")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser(prog="python -m ml.threshold_sweep")
    ap.add_argument("--data", default="data/raw/cicids2017")
    ap.add_argument("--model-dir", default="models/multiclass")
    ap.add_argument("--min-class-count", default="100")
    args = ap.parse_args()

    import joblib

    from ml import train
    from ml.common import METADATA_FILE, MODEL_FILE, load_schema, sha256_file

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    want = meta.get("artifact_sha256")
    if want and sha256_file(model_dir / MODEL_FILE) != want:
        sys.exit("model file does not match metadata.json; refusing to load it")

    d, s = meta["dataset"], meta["split"]
    a = train.parse([
        "--data", args.data, "--dataset-name", d["name"], "--dataset-version", d["version"],
        "--licence", d["licence"], "--source-url", d["source_url"], "--benign-label", meta["benign_label"],
        "--split", s["method"], "--task", meta["task"], "--seed", str(meta["seed"]),
        "--test-size", str(s["test_size"]), "--val-size", str(s["val_size"]),
        "--min-class-count", args.min_class_count, "--out-dir", str(model_dir),
    ])
    schema = load_schema()
    X, y, groups, times = train.load(a, schema)
    X, y, groups, times, _ = train.clean(X, y, groups, times, schema, a)
    y = np.asarray(y)
    tr, te = train.split(y, groups, times, a.split, a.test_size, a.seed)
    tr2, va = train.split(y[tr], train.take(groups, tr), train.take(times, tr), a.split, a.val_size, a.seed)
    tr_idx, va_idx = tr[tr2], tr[va]
    if (len(tr_idx), len(va_idx), len(te)) != (s["rows_train"], s["rows_val"], s["rows_test"]):
        sys.exit("split sizes differ from the saved model; the data or arguments are not the same as in training")

    pipe = joblib.load(model_dir / MODEL_FILE)
    classes = [str(c) for c in pipe.classes_]
    benign = meta["benign_label"]
    b = classes.index(benign)

    def scores(idx):
        p = pipe.predict_proba(X.iloc[idx])
        return np.round(1 - p[:, b], 6), np.asarray(classes, dtype=object)[p.argmax(1)]

    sv, pv = scores(va_idx)
    st, pt = scores(te)
    yv, yt = y[va_idx], y[te]

    ops = operating_points(sv, yv, benign)
    tv = table(sv, pv, yv, benign, ops)
    tt = table(st, pt, yt, benign, ops)
    pc = per_class(sv, pv, yv, benign, ops)

    out = model_dir / "eval"
    tv.to_csv(out / "threshold_sweep_validation.csv", index=False)
    tt.to_csv(out / "threshold_sweep_test_confirmation.csv", index=False)
    pc.to_csv(out / "threshold_per_class_validation.csv")
    plot(sv, yv, benign, tv, out / "threshold_sweep.png")

    cols = ["rule", "threshold", "attack_recall", "alarms_per_100k_benign", "precision_1pct_attacks", "false_alarms_per_real_attack_0.1pct"]
    pct = ("attack_recall", "precision_1pct_attacks")
    report = (
        f"# Threshold analysis, model {meta['model_version']}\n\n"
        f"attack_score = 1 - P(Benign). An alert fires when attack_score is above the threshold.\n"
        f"Thresholds were chosen on the validation split ({len(va_idx)} flows). The test split ({len(te)} flows) is confirmation only.\n"
        f"Precision and false-alarms-per-real-attack columns assume attacks are 1% or 0.1% of flows, not the dataset's mix.\n\n"
        f"## Validation\n\n{md(tv[cols], pct)}\n\n"
        f"## Test confirmation, same thresholds\n\n{md(tt[cols], pct)}\n\n"
        f"## Per-class share of flows flagged as attack (validation)\n\n{md(pc.reset_index().rename(columns={'index': 'class'}), pc.columns)}\n\n"
        f"![thresholds](threshold_sweep.png)\n"
    )
    (out / "threshold_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()