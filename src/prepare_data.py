"""Build the Kazakh profanity dataset from KazSAnDRA reviews.

Steps: download -> mask personal info -> remove duplicates -> label (rules + manual review)
-> sample negatives -> group near-duplicates -> split 80/10/10 with a fixed seed.

Run:  python src/prepare_data.py
"""
import argparse
import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from lexicon import rule_hits
from text_utils import mask_pii, near_dup_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
BASE_URL = "https://raw.githubusercontent.com/IS2AI/KazSAnDRA/main/dataset/"
FILES = ["01_pc_train_ib", "04_pc_valid", "05_pc_test"]  # polarity files, together = full corpus

# mild insults / rude words that are NOT profanity under our rules (hard negatives)
MILD = re.compile(r"(?<!\w)(?:мал|малдар|ешак|ешек|ақымақ|акымак|жынды|тупой|қоқыс|коқыс|мусор|есек|шошқа|ит)(?!\w)", re.I)


def download():
    os.makedirs(RAW, exist_ok=True)
    for name in FILES:
        csv_path = os.path.join(RAW, name + ".csv")
        if os.path.exists(csv_path):
            continue
        print("downloading", name)
        data = urllib.request.urlopen(BASE_URL + name + ".zip").read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(RAW)


def load_corpus():
    df = pd.concat([pd.read_csv(os.path.join(RAW, f + ".csv")) for f in FILES], ignore_index=True)
    df = df.rename(columns={"label": "sentiment"})[["custom_id", "text", "sentiment", "domain"]]
    df["domain"] = df["domain"].fillna("appstore")  # a few rows have no domain; ids show they are appstore
    df["raw_text"] = df["text"].astype(str)
    df["text"] = df["raw_text"].map(mask_pii)
    return df


def label(df):
    hits = df["raw_text"].map(rule_hits)
    df["strong"] = hits.map(lambda h: ",".join(h[0]))
    df["weak"] = hits.map(lambda h: ",".join(h[1]))
    df["label"] = (df["strong"] != "").astype(int)
    df["label_source"] = "rule"

    manual = pd.read_csv(os.path.join(ROOT, "data", "manual_labels.csv"))
    m = dict(zip(manual["text"], manual["manual_label"]))
    has = df["raw_text"].isin(m)
    df.loc[has, "label"] = df.loc[has, "raw_text"].map(m).astype(int)
    df.loc[has, "label_source"] = "manual"
    # weak-only texts that were never reviewed are dropped (unclear)
    unclear = (df["strong"] == "") & (df["weak"] != "") & ~has
    return df[~unclear].copy()


def split_groups(df, seed):
    """Assign whole near-duplicate groups to train/val/test, stratified by label."""
    rng = np.random.default_rng(seed)
    g = df.groupby("group").label.max()
    split = {}
    for lab in (0, 1):
        groups = g[g == lab].index.to_numpy()
        rng.shuffle(groups)
        n = len(groups)
        a, b = int(0.8 * n), int(0.9 * n)
        split.update({x: "train" for x in groups[:a]})
        split.update({x: "val" for x in groups[a:b]})
        split.update({x: "test" for x in groups[b:]})
    return df["group"].map(split)


def main(seed=42, n_neg=20000):
    download()
    df = load_corpus()
    n_raw = len(df)

    # exact duplicates (after lowercasing and masking)
    df["dedup"] = df["text"].str.lower().str.strip()
    df = df.drop_duplicates("dedup").drop(columns="dedup")
    df = df[df["text"].str.len() > 0]
    n_dedup = len(df)

    df = label(df)

    # keep all positives; sample negatives half from negative reviews, half from positive
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    hard = neg[(neg.label_source == "manual") | neg.raw_text.str.contains(MILD)]
    rest = neg.drop(hard.index)
    k = max(0, n_neg - len(hard)) // 2
    neg_bad = rest[rest.sentiment == 0].sample(k, random_state=seed)
    neg_good = rest[rest.sentiment == 1].sample(k, random_state=seed)
    data = pd.concat([pos, hard, neg_bad, neg_good]).sample(frac=1, random_state=seed)

    data["group"] = data["text"].map(near_dup_key)
    # texts with an empty key (only emoji / digits) form their own group
    empty = data["group"] == ""
    data.loc[empty, "group"] = "_" + data.loc[empty, "custom_id"]
    # a near-duplicate group must have one label; take the majority
    data["label"] = data.groupby("group").label.transform(lambda s: int(s.mean() >= 0.5))
    data["split"] = split_groups(data, seed)

    os.makedirs(OUT, exist_ok=True)
    cols = ["custom_id", "text", "label", "label_source", "sentiment", "domain", "group", "split"]
    for s in ("train", "val", "test"):
        data.loc[data.split == s, cols].to_csv(os.path.join(OUT, f"{s}.csv"), index=False)

    print(f"raw reviews: {n_raw}, after exact dedup: {n_dedup}, final dataset: {len(data)}")
    print("near-duplicate groups:", data.group.nunique())
    print(data.groupby("split").label.agg(["count", "sum"]).rename(columns={"sum": "profane"}))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_neg", type=int, default=20000)
    a = ap.parse_args()
    main(a.seed, a.n_neg)
