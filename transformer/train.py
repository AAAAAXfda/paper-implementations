"""Train the Transformer on a toy copy task: the target is the source sequence.

    python transformer/train.py
    python transformer/train.py --steps 5000 --device cuda
"""
import argparse
import random

import torch
import torch.nn.functional as F

from model import BOS, EOS, PAD, Transformer


def make_batch(bs, vocab, min_len, max_len, device):
    lens = torch.randint(min_len, max_len + 1, (bs,))
    T = int(lens.max())
    src = torch.full((bs, T), PAD, dtype=torch.long)
    tgt = torch.full((bs, T + 2), PAD, dtype=torch.long)
    tgt[:, 0] = BOS
    for i, l in enumerate(lens.tolist()):
        seq = torch.randint(3, vocab, (l,))  # 0, 1, 2 are PAD, BOS, EOS
        src[i, :l] = seq
        tgt[i, 1 : l + 1] = seq
        tgt[i, l + 1] = EOS
    return src.to(device), tgt.to(device)


def evaluate(model, vocab, min_len, max_len, device, n=1000):
    model.eval()
    src, _ = make_batch(n, vocab, min_len, max_len, device)
    pred = model.greedy_decode(src, max_len=max_len + 2)
    correct = 0
    for s, p in zip(src.tolist(), pred.tolist()):
        want = [t for t in s if t != PAD]
        got = []
        for t in p:
            if t == EOS:
                break
            got.append(t)
        correct += int(got == want)
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=30)
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=15)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )

    model = Transformer(args.vocab, args.vocab, d_model=args.d_model).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device: {device}")

    # Noam schedule from the paper: lr = d^-0.5 * min(step^-0.5, step * warmup^-1.5)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    factor = 0.5
    noam = lambda s: factor * args.d_model ** -0.5 * min((s + 1) ** -0.5, (s + 1) * args.warmup ** -1.5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam)

    for step in range(1, args.steps + 1):
        model.train()
        src, tgt = make_batch(args.batch_size, args.vocab, args.min_len, args.max_len, device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
            ignore_index=PAD, label_smoothing=0.1,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
        if step % 500 == 0 or step == args.steps:
            acc = evaluate(model, args.vocab, args.min_len, args.max_len, device)
            print(f"  exact-match accuracy: {acc:.3f}")

    final = evaluate(model, args.vocab, args.min_len, args.max_len, device)
    print(f"\nfinal exact-match accuracy on 1000 fresh sequences: {final:.4f}")


if __name__ == "__main__":
    main()
