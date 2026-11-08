"""Encoder-decoder Transformer, following "Attention Is All You Need".

Masks are boolean tensors where True means "this position can be attended to".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = self.q(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v(v).view(B, Tk, self.h, self.dk).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.drop(F.relu(self.l1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        # post-norm, like the original paper
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
        x = self.n3(x + self.drop(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d_model, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def src_mask(src):
        return (src != PAD)[:, None, None, :]  # (B, 1, 1, Ts)

    @staticmethod
    def tgt_mask(tgt):
        T = tgt.size(1)
        pad = (tgt != PAD)[:, None, None, :]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
        return pad & causal[None, None]  # (B, 1, Tt, Tt)

    def encode(self, src, src_mask):
        x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.drop(self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.out(x)

    def forward(self, src, tgt_in):
        src_mask = self.src_mask(src)
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, src_mask, self.tgt_mask(tgt_in))

    @torch.no_grad()
    def greedy_decode(self, src, max_len):
        src_mask = self.src_mask(src)
        memory = self.encode(src, src_mask)
        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_mask, self.tgt_mask(ys))
            nxt = logits[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS)
            if done.all():
                break
        return ys[:, 1:]
