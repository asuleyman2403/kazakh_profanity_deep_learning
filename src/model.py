"""Character-level CNN for binary profanity detection."""
import json
from collections import Counter

import torch
import torch.nn as nn

from text_utils import model_normalize

PAD, UNK = 0, 1


class CharVocab:
    def __init__(self, chars):
        self.itos = ["<pad>", "<unk>"] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=3):
        """Build from TRAINING texts only."""
        cnt = Counter(c for t in texts for c in model_normalize(t))
        return cls(sorted(c for c, n in cnt.items() if n >= min_freq))

    def encode(self, text, max_len):
        t = model_normalize(text)
        if len(t) > max_len:  # long comment: keep the start and the end
            half = max_len // 2
            t = t[:half] + t[-(max_len - half):]
        ids = [self.stoi.get(c, UNK) for c in t] or [UNK]
        return ids + [PAD] * (max_len - len(ids))

    def save(self, path):
        json.dump(self.itos[2:], open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf-8")))

    def __len__(self):
        return len(self.itos)


class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, n_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x):                      # x: (batch, max_len) char ids
        mask = (x != PAD).unsqueeze(1)          # (batch, 1, len)
        e = self.emb(x).transpose(1, 2)         # (batch, emb, len)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)      # padding cannot win the max-pool
            feats.append(h.max(dim=2).values)   # strongest match of each filter anywhere in the text
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(1)  # logit
