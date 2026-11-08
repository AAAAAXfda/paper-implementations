"""LoRA demo, no pretrained weights anywhere.

1. Pretrain a small GPT-style model as a language model on the toy sentences.
2. Freeze it, add LoRA to the attention q/v projections, and train a sentiment
   classifier (LoRA + a new linear head).
3. Compare against training only the head (frozen model, no LoRA).
4. Check that merging the LoRA weights gives the same outputs.

    python lora/lora_demo.py
"""
import argparse
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD, batchify, build_vocab, split_data
from gpt import GPT
from lora import count_params, inject_lora, merge_lora


class SentimentClassifier(nn.Module):
    def __init__(self, gpt, d_model):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(d_model, 2)

    def forward(self, idx, lengths):
        h = self.gpt(idx, return_hidden=True)
        last = h[torch.arange(idx.size(0), device=idx.device), lengths - 1]
        return self.head(last)


def pretrain(model, train, stoi, steps, bs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        model.train()
        idx, _, _ = batchify(random.sample(train, bs), stoi, device)
        logits = model(idx)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"  pretrain step {step:5d}  lm loss {loss.item():.3f}")


@torch.no_grad()
def accuracy(clf, data, stoi, device):
    clf.eval()
    idx, lengths, labels = batchify(data, stoi, device)
    return (clf(idx, lengths).argmax(-1) == labels).float().mean().item()


def finetune(clf, train, test, stoi, steps, bs, lr, device, name):
    params = [p for p in clf.parameters() if p.requires_grad]
    trainable, total = count_params(clf)
    print(f"\n{name}: training {trainable:,} of {total:,} params ({100 * trainable / total:.1f}%)")
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(1, steps + 1):
        clf.train()
        idx, lengths, labels = batchify(random.sample(train, bs), stoi, device)
        loss = F.cross_entropy(clf(idx, lengths), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.item():.3f}  test acc {accuracy(clf, test, stoi, device):.3f}")
    return accuracy(clf, test, stoi, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_steps", type=int, default=1500)
    ap.add_argument("--finetune_steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )

    train, test = split_data(args.seed)
    stoi = build_vocab(train + test)
    print(f"{len(train)} train / {len(test)} test sentences, vocab {len(stoi)}, device {device}")

    d_model = 64
    base = GPT(len(stoi), d_model=d_model).to(device)
    print("pretraining the base model (language modelling)...")
    pretrain(base, train, stoi, args.pretrain_steps, args.batch_size, 3e-3, device)

    # baseline: frozen base model, only the new head trains
    head_only_gpt = copy.deepcopy(base)
    for p in head_only_gpt.parameters():
        p.requires_grad = False
    head_only = SentimentClassifier(head_only_gpt, d_model).to(device)
    acc_head = finetune(head_only, train, test, stoi, args.finetune_steps,
                        args.batch_size, 3e-3, device, "head only")

    # LoRA: frozen base + low-rank updates on q/v + new head
    lora_gpt = inject_lora(copy.deepcopy(base), r=args.rank, alpha=args.alpha)
    lora_clf = SentimentClassifier(lora_gpt, d_model).to(device)
    acc_lora = finetune(lora_clf, train, test, stoi, args.finetune_steps,
                        args.batch_size, 3e-3, device, f"LoRA (r={args.rank})")

    # merging the update into the weights should not change the outputs
    lora_clf.eval()
    idx, lengths, _ = batchify(test, stoi, device)
    with torch.no_grad():
        before = lora_clf(idx, lengths)
    merge_lora(lora_clf.gpt)
    with torch.no_grad():
        after = lora_clf(idx, lengths)
    print(f"\nmerge check, max abs diff in logits: {(before - after).abs().max().item():.2e}")

    print(f"\nhead only : {acc_head:.4f}")
    print(f"LoRA      : {acc_lora:.4f}")


if __name__ == "__main__":
    main()
