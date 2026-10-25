# ICN_ML_Model: ML component of the AI-powered open-source firewall

Random Forest classifier that labels a network flow as Benign or one of 11 attack types, from 19 flow features. It is the machine-learning part (Laptop 2) of an end-semester project built on OPNsense, Suricata and CICFlowMeter, with an LLM layer that explains decisions.

The model scores flows. It never blocks anything itself: blocking is decided by the policy engine (Laptop 3).

## Results at a glance

Held-out test split of CIC-IDS2017, stratified random split, seed 42, about 17% attacks.

| Model | Version | Attack recall | False-positive rate | Macro F1 |
|---|---|---|---|---|
| Multiclass (default) | `rf-d64f3a43de` | 99.74% | 0.35% | 87.39% |
| Binary (reference) | `rf-580816bcd7` | 99.60% | 0.23% | 99.54% |

These numbers are optimistic: the dataset has no IPs or timestamps, so flows from the same attack session can land in both train and test. Web Attack - XSS, Web Attack - Brute Force and Bot are weak classes and must not be used to block on their own. Full tables are in `models/multiclass/eval/report.md` and `models/binary/eval/report.md`.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/Sarthaktya/ICN_ML_Model.git
cd ICN_ML_Model
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt
python -m pytest ml/tests -v
```

Python 3.11 was used. All 20 tests should pass in about 10 seconds and need no dataset or model file.

## Get the model file

`classifier.joblib` (61 MB) is not in git. Get it from the repo owner and place it at `models/classifier.joblib`.

Expected SHA-256: `PASTE_HASH_HERE`

```powershell
Get-FileHash models\classifier.joblib
```

Only load a `.joblib` file from a trusted source. `ml.predict` checks the hash against `models/metadata.json` and refuses a mismatch.

## Predict

```powershell
python -m ml.predict --input ml/fixtures/valid_flow.json
python -m ml.predict --input ml/fixtures/batch.json
```

From Python:

```python
from ml.predict import predict, predict_batch, InputValidationError
result = predict(flow_dict)
```

Input is a JSON object with exactly the 19 fields in `ml/feature_schema.json` (or a list of such objects). Output per flow: `model_version`, `feature_schema_version`, `predicted_class`, `is_attack`, `attack_score`, `probabilities`, `probabilities_calibrated` (always false). Use `attack_score` for thresholds, not the label. Invalid input exits with code 2 and lists every faulty field.

## Features

18 numeric features plus `protocol` (1, 6 or 17): flow duration, packet and byte counts per direction, packet length mean and standard deviation, rates, inter-arrival time mean, standard deviation and max, and SYN, ACK, RST, FIN and PSH flag counts. Units follow CICFlowMeter: times in microseconds, rates per second, lengths in bytes. Ports, IP addresses and timestamps are deliberately excluded. Schema version: `1.0-draft`.

## Train

Download the Kaggle dataset [dhoogla/cicids2017](https://www.kaggle.com/datasets/dhoogla/cicids2017) (cleaned parquet) and place the files in `data/raw/cicids2017/`. The data is not in git.

```powershell
$meta = @("--data","data/raw/cicids2017","--dataset-name","CIC-IDS2017 (Kaggle dhoogla, cleaned parquet)","--dataset-version","Kaggle V3, downloaded 2026-10-25","--licence","CC BY-NC-SA 4.0 (Kaggle dhoogla/cicids2017; original data: Canadian Institute for Cybersecurity)","--source-url","https://www.kaggle.com/datasets/dhoogla/cicids2017","--benign-label","Benign","--split","stratified")
python -m ml.train @meta --task multiclass --min-class-count 100 --out-dir models/multiclass
python -m ml.train @meta --task binary --out-dir models/binary
```

Each run writes `classifier.joblib`, `metadata.json` and an `eval/` folder (metrics, confusion matrix, report). A retrain of the multiclass model gave byte-identical metrics and an identical model file hash. To make the multiclass model the default, copy its `classifier.joblib` and `metadata.json` into `models/`.

## Repository layout

| Path | Contents |
|---|---|
| `ml/` | `train.py`, `predict.py`, `common.py`, `feature_schema.json` |
| `ml/fixtures/` | Example flows (hand-written values, not measured) |
| `ml/tests/` | Tests and a synthetic data generator |
| `models/` | Metadata and evaluation reports (model binaries are not in git) |
| `scripts/` | Dataset inspection helper |

## Limitations

- Scores come from CIC-IDS2017 with a random row split and are an upper bound for our network.
- Infiltration, Web Attack - Sql Injection and Heartbleed were dropped for lack of data; the model cannot name them.
- Probabilities are not calibrated.
- Real traffic is mostly benign, so even a 0.35% false-positive rate causes many false alarms; require repeated evidence before blocking.

## Data licence and citation

CIC-IDS2017 is by the Canadian Institute for Cybersecurity, licence CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike).
Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.