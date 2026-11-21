"""Evaluate the saved model ONCE on the held-out test set.

Run:  python src/evaluate.py
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

from predict import ProfanityDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def main():
    det = ProfanityDetector()
    test = pd.read_csv(os.path.join(ROOT, "data", "processed", "test.csv"))
    preds = det.predict(test.text.tolist())
    test["probability"] = [p["probability"] for p in preds]
    test["pred"] = [p["label"] for p in preds]

    p, r, f1, _ = precision_recall_fscore_support(test.label, test.pred, average="binary", pos_label=1)
    cm = confusion_matrix(test.label, test.pred, labels=[0, 1])
    metrics = dict(n_test=len(test), n_profane=int(test.label.sum()), threshold=det.threshold,
                   precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                   tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))
    os.makedirs(RES, exist_ok=True)
    json.dump(metrics, open(os.path.join(RES, "test_metrics.json"), "w"), indent=2)
    test[["custom_id", "text", "label", "pred", "probability", "domain"]].to_csv(
        os.path.join(RES, "test_predictions.csv"), index=False)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["0 clean", "1 profane"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Test set (n={len(test)})")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=150)
    print(json.dumps(metrics, indent=2))
    return metrics, test, fig


if __name__ == "__main__":
    main()
