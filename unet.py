"""Modest 2D U-Net: one transmitted-light input channel, four fluorescence output channels in [0, 1]."""
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([block(in_ch, chs[0])] + [block(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([block(2 * chs[i], chs[i]) for i in range(depth)])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.factor = 2 ** depth

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(reversed(self.up), reversed(self.dec)):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return torch.sigmoid(self.head(x))
