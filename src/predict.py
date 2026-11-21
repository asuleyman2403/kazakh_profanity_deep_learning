"""Load the saved model and predict on new Kazakh comments (no retraining).

Run:  python src/predict.py "Өте жақсы тауар, рахмет"
      python src/predict.py            # interactive mode
"""
import json
import os
import sys

import torch

from model import CharCNN, CharVocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


class ProfanityDetector:
    def __init__(self, model_dir=MODELS):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.vocab = CharVocab.load(os.path.join(model_dir, "vocab.json"))
        c = self.cfg
        self.model = CharCNN(len(self.vocab), c["emb_dim"], c["n_filters"], tuple(c["kernel_sizes"]), c["dropout"])
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "char_cnn.pt"), map_location="cpu"))
        self.model.eval()
        self.threshold = c["threshold"]

    @torch.no_grad()
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = torch.tensor([self.vocab.encode(t, self.cfg["max_len"]) for t in texts], dtype=torch.long)
        probs = torch.sigmoid(self.model(X)).tolist()
        return [dict(text=t, probability=round(p, 4), label=int(p >= self.threshold)) for t, p in zip(texts, probs)]


if __name__ == "__main__":
    det = ProfanityDetector()
    texts = sys.argv[1:]
    if texts:
        for r in det.predict(texts):
            print(f"label={r['label']}  p={r['probability']:.3f}  {r['text']}")
    else:
        print(f"threshold = {det.threshold}. Type a comment (empty line to quit).")
        while (t := input("> ").strip()):
            r = det.predict(t)[0]
            print(f"label={r['label']}  p={r['probability']:.3f}")
