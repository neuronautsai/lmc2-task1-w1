"""Modest 2D U-Net: one transmitted-light input channel, four fluorescence output channels in [0, 1]."""
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    """n_mod > 0 adds modality-aware conditioning: a learned embedding of the transmitted-light modality (BF, DIC, PC and an
    unknown token at index n_mod - 1) produces a per-channel scale and shift after every encoder and decoder block (FiLM).
    The scale and shift start at zero, so the network starts as the unconditioned U-Net. n_mod = 0 gives the Week 2 model."""

    def __init__(self, in_ch=1, out_ch=4, base=32, depth=4, n_mod=0, emb_dim=32):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([block(in_ch, chs[0])] + [block(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([block(2 * chs[i], chs[i]) for i in range(depth)])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.factor = 2 ** depth
        self.n_mod = n_mod
        if n_mod:
            self.emb = nn.Embedding(n_mod, emb_dim)
            self.film_enc = nn.ModuleList([nn.Linear(emb_dim, 2 * c) for c in chs])
            self.film_dec = nn.ModuleList([nn.Linear(emb_dim, 2 * chs[i]) for i in range(depth)])
            for m in list(self.film_enc) + list(self.film_dec):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _film(x, layer, e):
        g, b = layer(e).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, mod=None):
        """mod: (B,) long tensor of modality ids, or None. With n_mod > 0 a missing label uses the unknown token."""
        e = None
        if self.n_mod:
            if mod is None:
                mod = torch.full((x.shape[0],), self.n_mod - 1, dtype=torch.long, device=x.device)
            e = self.emb(mod.to(x.device))
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else self.pool(x))
            if e is not None:
                x = self._film(x, self.film_enc[i], e)
            skips.append(x)
        x = skips.pop()
        for j, (up, dec) in enumerate(zip(reversed(self.up), reversed(self.dec))):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
            if e is not None:
                x = self._film(x, self.film_dec[len(self.dec) - 1 - j], e)
        return torch.sigmoid(self.head(x))
