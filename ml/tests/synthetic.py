import numpy as np
import pandas as pd

from ml.common import load_schema

SPECS = {
    "BENIGN": dict(dur=(13.5, 1.2), fwd=8, bwd=8, fm=120, bm=900, sd=350, syn=1, ack=12, rst=0.05, fin=1, psh=5, tcp=0.8),
    "PortScan": dict(dur=(6.0, 0.8), fwd=0, bwd=0.3, fm=5, bm=0, sd=5, syn=1, ack=0.1, rst=0.8, fin=0.01, psh=0, tcp=0.98),
    "SSH-Patator": dict(dur=(14.5, 0.6), fwd=20, bwd=20, fm=60, bm=80, sd=40, syn=1, ack=38, rst=0.2, fin=1, psh=18, tcp=1.0),
}


def make_csv(path, n=600, seed=0):
    rng = np.random.default_rng(seed)
    schema = load_schema()
    src = {f["name"]: f["source_column"] for f in schema["numeric_features"] + schema["categorical_features"]}
    frames = []
    for cid, (label, s) in enumerate(SPECS.items()):
        dur = np.maximum(rng.lognormal(s["dur"][0], s["dur"][1], n), 1.0).round()
        fwd = rng.poisson(s["fwd"], n) + 1
        bwd = rng.poisson(s["bwd"], n)
        fm = np.clip(rng.normal(s["fm"], 40, n), 0, None)
        bm = np.clip(rng.normal(s["bm"], 40, n), 0, None) * (bwd > 0)
        fb, bb = (fwd * fm).round(), (bwd * bm).round()
        total = fwd + bwd
        iat = dur / total
        frames.append(
            pd.DataFrame(
                {
                    "flow_duration_us": dur.astype("int64"),
                    "total_fwd_packets": fwd,
                    "total_bwd_packets": bwd,
                    "total_fwd_bytes": fb.astype("int64"),
                    "total_bwd_bytes": bb.astype("int64"),
                    "fwd_pkt_len_mean": fm,
                    "bwd_pkt_len_mean": bm,
                    "pkt_len_std": np.clip(rng.normal(s["sd"], 20, n), 0, None),
                    "flow_bytes_per_s": (fb + bb) / (dur / 1e6),
                    "flow_pkts_per_s": total / (dur / 1e6),
                    "flow_iat_mean": iat,
                    "flow_iat_std": iat * rng.uniform(0.2, 2, n),
                    "flow_iat_max": iat * rng.uniform(2, 8, n),
                    "syn_flag_count": rng.poisson(s["syn"], n),
                    "ack_flag_count": rng.poisson(s["ack"], n),
                    "rst_flag_count": rng.poisson(s["rst"], n),
                    "fin_flag_count": rng.poisson(s["fin"], n),
                    "psh_flag_count": rng.poisson(s["psh"], n),
                    "protocol": rng.choice([6, 17], n, p=[s["tcp"], 1 - s["tcp"]]),
                    "Label": label,
                    "Source IP": [f"10.{cid}.0.{i}" for i in rng.integers(1, 31, n)],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    dirty = df.iloc[:3].copy()
    dirty.loc[dirty.index[0], "flow_bytes_per_s"] = np.inf
    dirty.loc[dirty.index[1], "flow_duration_us"] = -5
    dirty.loc[dirty.index[2], "protocol"] = 0
    df = pd.concat([df, dirty, df.iloc[[10]]], ignore_index=True)
    df = df.sample(frac=1, random_state=seed)
    df.columns = [" " + src.get(c, c) for c in df.columns]
    df.to_csv(path, index=False)