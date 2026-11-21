"""Train the char-CNN. Settings are chosen on the validation set; the test set is not used here.

Run:  python src/train.py
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from model import CharCNN, CharVocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")

DEFAULTS = dict(seed=42, max_len=300, min_freq=3, emb_dim=64, n_filters=128, kernel_sizes=[2, 3, 4, 5],
                dropout=0.3, lr=1e-3, weight_decay=1e-5, batch_size=128, epochs=15)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def to_tensor(texts, vocab, max_len):
    return torch.tensor([vocab.encode(t, max_len) for t in texts], dtype=torch.long)


def length_batches(lengths, batch_size, g):
    """Shuffle, then sort inside big chunks by length so a batch has similar lengths (less padding)."""
    perm = torch.randperm(len(lengths), generator=g)
    chunk = batch_size * 20
    batches = []
    for i in range(0, len(perm), chunk):
        part = perm[i:i + chunk]
        part = part[torch.argsort(lengths[part], stable=True)]
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    order = torch.randperm(len(batches), generator=g)
    return [batches[k] for k in order]


def trim(xb):
    """Cut the padding columns that no text in the batch uses (faster, same result)."""
    n = int((xb != 0).sum(1).max())
    return xb[:, :max(n, 5)]


@torch.no_grad()
def predict_proba(model, X, batch_size=512):
    model.eval()
    return torch.cat([torch.sigmoid(model(trim(X[i:i + batch_size]))) for i in range(0, len(X), batch_size)]).numpy()


def best_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y, p >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main(cfg):
    set_seed(cfg["seed"])
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    val = pd.read_csv(os.path.join(DATA, "val.csv"))

    vocab = CharVocab.build(train.text, cfg["min_freq"])  # training data only
    Xtr, ytr = to_tensor(train.text, vocab, cfg["max_len"]), torch.tensor(train.label.values, dtype=torch.float)
    Xva, yva = to_tensor(val.text, vocab, cfg["max_len"]), val.label.values
    print(f"vocab size {len(vocab)}, train {len(train)}, val {len(val)}")

    model = CharCNN(len(vocab), cfg["emb_dim"], cfg["n_filters"], tuple(cfg["kernel_sizes"]), cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    lengths = (Xtr != 0).sum(1)

    os.makedirs(MODELS, exist_ok=True)
    history, best = [], (-1, None, None)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total = time.time(), 0.0
        for idx in length_batches(lengths, cfg["batch_size"], g):
            opt.zero_grad()
            loss = loss_fn(model(trim(Xtr[idx])), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        p = predict_proba(model, Xva)
        thr, f1 = best_threshold(yva, p)
        history.append(dict(epoch=epoch, train_loss=total / len(Xtr), val_f1_at_0_5=f1_score(yva, p >= 0.5),
                            val_best_f1=f1, val_best_thr=thr))
        print(f"epoch {epoch:2d} loss {total / len(Xtr):.4f} val F1@0.5 {history[-1]['val_f1_at_0_5']:.4f} "
              f"best val F1 {f1:.4f} (thr {thr}) {time.time() - t0:.0f}s", flush=True)
        if f1 > best[0]:
            best = (f1, epoch, thr)
            torch.save(model.state_dict(), os.path.join(MODELS, "char_cnn.pt"))

    vocab.save(os.path.join(MODELS, "vocab.json"))
    cfg = dict(cfg, vocab_size=len(vocab), best_epoch=best[1], threshold=best[2], val_f1=best[0])
    json.dump(cfg, open(os.path.join(MODELS, "config.json"), "w"), indent=2)
    pd.DataFrame(history).to_csv(os.path.join(ROOT, "results", "train_history.csv"), index=False)
    print(f"saved best model from epoch {best[1]} (val F1 {best[0]:.4f}, threshold {best[2]})")
    return cfg, pd.DataFrame(history)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if k != "kernel_sizes":
            ap.add_argument("--" + k, type=type(v), default=v)
    main(dict(DEFAULTS, **vars(ap.parse_args())))
