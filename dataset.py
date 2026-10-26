"""Dataset for LMC2 Task 1: one transmitted-light plane to four fluorescence channels.

Normalization rule (declared): every image is scaled on its own by the 1st and 99.5th percentile and clipped to [0, 1].
The input rule uses only the input image, so it also works at inference.
Targets are normalized the same way, so models predict normalized fluorescence and absolute intensity is not evaluated.
Training items hold crops_per_image random crops of one image (shape K, C, P, P), evaluation items hold the full image.
Missing channels get a finite placeholder (zeros) and observed=False, they must be excluded from the loss.
"""
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, get_worker_info

from config import ORGANELLES, ChallengeConfig


def normalize(a, p_low, p_high):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_pairs(cfg, split):
    df = pd.read_csv(cfg.pairs_csv)
    df = df[(df.split == split) & ~df.excluded]
    has_target = df[[f"{o}_obs" for o in ORGANELLES]].to_numpy().any(axis=1)
    return df[has_target].reset_index(drop=True)


class LMC2Dataset(Dataset):
    def __init__(self, cfg: ChallengeConfig, split: str, train: bool, seed: int = 0):
        self.cfg = cfg
        self.train = train
        self.df = load_pairs(cfg, split)
        self.seed = seed
        self.epoch = 0
        self._rng = None
        self._rng_key = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def rng(self):
        info = get_worker_info()
        key = (self.epoch, info.id if info else -1)
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng(self.seed + 1000 * self.epoch + (info.id if info else 0))
            self._rng_key = key
        return self._rng

    def __len__(self):
        return len(self.df)

    def _read(self, study, name):
        a = tifffile.imread(self.cfg.data_root / f"Study_{study:02d}" / name)
        return normalize(a, self.cfg.p_low, self.cfg.p_high)

    def _augment(self, x, y):
        size = self.cfg.patch_size
        h, w = x.shape[-2:]
        ph, pw = max(0, size - h), max(0, size - w)
        if ph or pw:
            pad = ((0, 0), (0, ph), (0, pw))
            x, y = np.pad(x, pad, mode="reflect"), np.pad(y, pad, mode="reflect")
            h, w = x.shape[-2:]
        i = int(self.rng.integers(0, h - size + 1))
        j = int(self.rng.integers(0, w - size + 1))
        x, y = x[:, i:i + size, j:j + size], y[:, i:i + size, j:j + size]
        if self.rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        if self.rng.random() < 0.5:
            x, y = x[:, ::-1, :], y[:, ::-1, :]
        k = int(self.rng.integers(0, 4))
        return np.rot90(x, k, axes=(1, 2)).copy(), np.rot90(y, k, axes=(1, 2)).copy()

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x = self._read(int(r.study), r.in_file)[None]
        y = np.zeros((len(ORGANELLES),) + x.shape[1:], dtype=np.float32)
        observed = np.zeros(len(ORGANELLES), dtype=bool)
        for k, o in enumerate(ORGANELLES):
            if bool(r[f"{o}_obs"]):
                t = self._read(int(r.study), r[f"{o}_file"])
                assert t.shape == x.shape[1:], f"shape mismatch {r.study} {r.image_id} {o}"
                y[k] = t
                observed[k] = True
        if self.train:
            crops = [self._augment(x, y) for _ in range(self.cfg.crops_per_image)]
            x = np.stack([c[0] for c in crops])
            y = np.stack([c[1] for c in crops])
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(np.ascontiguousarray(y)),
            "observed": torch.from_numpy(observed),
            "study": int(r.study),
            "image_id": int(r.image_id),
            "modality": r.modality,
        }
