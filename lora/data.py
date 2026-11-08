"""Tiny synthetic sentiment dataset, so nothing has to be downloaded.

Negation flips the label ("the movie was not great" is negative), so the model
can't just look for a single positive or negative word.
"""
import random

import torch

PAD, BOS = 0, 1

POS = ["great", "good", "wonderful", "amazing", "fun", "brilliant", "lovely", "excellent"]
NEG = ["terrible", "bad", "awful", "boring", "dull", "horrible", "poor", "dreadful"]
NOUNS = ["movie", "film", "show", "book", "game", "story", "play", "album",
         "song", "meal", "trip", "class"]
INTENS = ["really", "very", "so", "quite", "truly"]


def build_examples():
    ex = []
    for n in NOUNS:
        for words, label in ((POS, 1), (NEG, 0)):
            for a in words:
                flip = 1 - label
                ex.append((f"the {n} was {a}", label))
                for i in INTENS:
                    ex.append((f"the {n} was {i} {a}", label))
                ex.append((f"i thought the {n} was {a}", label))
                ex.append((f"what a {a} {n}", label))
                ex.append((f"the {n} was not {a}", flip))
                ex.append((f"i did not think the {n} was {a}", flip))
    return ex


def split_data(seed=0, test_frac=0.2):
    ex = build_examples()
    random.Random(seed).shuffle(ex)
    n_test = int(len(ex) * test_frac)
    return ex[n_test:], ex[:n_test]  # train, test (no overlap)


def build_vocab(examples):
    words = sorted({w for text, _ in examples for w in text.split()})
    return {"<pad>": PAD, "<bos>": BOS, **{w: i + 2 for i, w in enumerate(words)}}


def batchify(examples, stoi, device):
    seqs = [[BOS] + [stoi[w] for w in text.split()] for text, _ in examples]
    lengths = torch.tensor([len(s) for s in seqs])
    idx = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s)
    labels = torch.tensor([y for _, y in examples])
    return idx.to(device), lengths.to(device), labels.to(device)
