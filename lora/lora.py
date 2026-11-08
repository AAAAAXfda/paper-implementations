"""LoRA (Hu et al., 2021): freeze W, learn a low-rank update  W + (alpha/r) * B @ A."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        # A gets a random init, B starts at zero, so the update is zero at step 0
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        update = self.drop(x) @ self.A.T @ self.B.T
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merge(self):
        """Fold the update into the base weight and return the plain Linear."""
        self.base.weight += (self.B @ self.A) * self.scaling
        return self.base


def inject_lora(model, targets=("q_proj", "v_proj"), r=4, alpha=8, dropout=0.0):
    """Freeze every parameter, then swap target Linear layers for LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
    return model


def merge_lora(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
