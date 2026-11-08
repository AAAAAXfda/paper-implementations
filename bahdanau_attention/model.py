"""Seq2seq with additive attention, after Bahdanau et al. (2015).

Bidirectional GRU encoder. At each decoder step the attention score for encoder
position j is  v^T tanh(W_h h_j + W_s s_{t-1}), the context is the weighted sum of
encoder states, and the decoder GRU takes [embedding, context] as input.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, BOS, EOS = 0, 1, 2


class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, dec_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid_dim, dec_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, lengths):
        emb = self.drop(self.emb(src))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # initial decoder state from the last forward and first backward states
        s0 = torch.tanh(self.fc(torch.cat([h[0], h[1]], dim=-1)))
        return out, s0  # out: (B, T, 2H)


class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s, enc_out, mask):
        scores = self.v(torch.tanh(self.W_h(enc_out) + self.W_s(s).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        a = F.softmax(scores, dim=-1)  # (B, T)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)  # (B, 2H)
        return ctx, a


class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, enc_dim, dec_dim, attn_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.attn = AdditiveAttention(enc_dim, dec_dim, attn_dim)
        self.gru = nn.GRUCell(emb_dim + enc_dim, dec_dim)
        self.out = nn.Linear(dec_dim + enc_dim + emb_dim, vocab)
        self.drop = nn.Dropout(dropout)

    def step(self, y_prev, s, enc_out, mask):
        emb = self.drop(self.emb(y_prev))
        ctx, a = self.attn(s, enc_out, mask)
        s = self.gru(torch.cat([emb, ctx], dim=-1), s)
        logits = self.out(torch.cat([s, ctx, emb], dim=-1))
        return logits, s, a


class Seq2Seq(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid_dim=128, attn_dim=64, dropout=0.1):
        super().__init__()
        dec_dim = hid_dim
        self.encoder = Encoder(vocab, emb_dim, hid_dim, dec_dim, dropout)
        self.decoder = Decoder(vocab, emb_dim, 2 * hid_dim, dec_dim, attn_dim, dropout)

    def forward(self, src, lengths, tgt_in):
        """Teacher forcing: feed the true previous token at every step."""
        enc_out, s = self.encoder(src, lengths)
        mask = src != PAD
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, s, _ = self.decoder.step(tgt_in[:, t], s, enc_out, mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)  # (B, T, V)

    @torch.no_grad()
    def greedy_decode(self, src, lengths, max_len, return_attn=False):
        enc_out, s = self.encoder(src, lengths)
        mask = src != PAD
        B = src.size(0)
        y = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        outs, attns = [], []
        for _ in range(max_len):
            logits, s, a = self.decoder.step(y, s, enc_out, mask)
            y = logits.argmax(-1).masked_fill(done, PAD)
            outs.append(y)
            attns.append(a)
            done = done | (y == EOS)
            if done.all():
                break
        outs = torch.stack(outs, dim=1)
        if return_attn:
            return outs, torch.stack(attns, dim=1)  # attn: (B, T_out, T_in)
        return outs
