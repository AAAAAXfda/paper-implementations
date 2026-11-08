"""Train seq2seq + additive attention on a toy task: output the input reversed.

Reversal is a nice test for attention, because the decoder has to look at the
source positions in the opposite order. The script prints the attention weights
for one example at the end so you can see that.

    python bahdanau_attention/train.py
"""
import argparse
import random

import torch
import torch.nn.functional as F

from model import BOS, EOS, PAD, Seq2Seq


def make_batch(bs, vocab, min_len, max_len, device):
    lens = torch.randint(min_len, max_len + 1, (bs,))
    T = int(lens.max())
    src = torch.full((bs, T), PAD, dtype=torch.long)
    tgt = torch.full((bs, T + 2), PAD, dtype=torch.long)
    tgt[:, 0] = BOS
    for i, l in enumerate(lens.tolist()):
        seq = torch.randint(3, vocab, (l,))
        src[i, :l] = seq
        tgt[i, 1 : l + 1] = seq.flip(0)
        tgt[i, l + 1] = EOS
    return src.to(device), lens.to(device), tgt.to(device)


def evaluate(model, vocab, min_len, max_len, device, n=1000):
    model.eval()
    src, lens, _ = make_batch(n, vocab, min_len, max_len, device)
    pred = model.greedy_decode(src, lens, max_len=max_len + 2)
    correct = 0
    for s, l, p in zip(src.tolist(), lens.tolist(), pred.tolist()):
        want = s[:l][::-1]
        got = []
        for t in p:
            if t == EOS:
                break
            got.append(t)
        correct += int(got == want)
    return correct / n


def show_attention(model, vocab, device, length=8):
    model.eval()
    src = torch.randint(3, vocab, (1, length)).to(device)
    lens = torch.tensor([length]).to(device)
    out, attn = model.greedy_decode(src, lens, max_len=length + 1, return_attn=True)
    print("\nsource:", src[0].tolist())
    print("output:", [t for t in out[0].tolist() if t not in (PAD, EOS)])
    print("attention (rows = output steps, columns = source positions):")
    for row in attn[0, :length].cpu():
        print(" ".join(f"{v:.2f}" for v in row.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=30)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )

    model = Seq2Seq(args.vocab).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        model.train()
        src, lens, tgt = make_batch(args.batch_size, args.vocab, args.min_len, args.max_len, device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, lens, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
        if step % 500 == 0 or step == args.steps:
            print(f"  exact-match accuracy: {evaluate(model, args.vocab, args.min_len, args.max_len, device):.3f}")

    final = evaluate(model, args.vocab, args.min_len, args.max_len, device)
    print(f"\nfinal exact-match accuracy on 1000 fresh sequences: {final:.4f}")
    show_attention(model, args.vocab, device)


if __name__ == "__main__":
    main()
