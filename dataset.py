"""Dataset for LMC2 Task 1: one transmitted-light plane to four fluorescence channels.

Normalization rule (declared): every image is scaled on its own by the 1st and 99.5th percentile and clipped to [0, 1].
Optional training target rule (cfg.train_target_*): training targets only can use other percentiles and no clipping.
Inputs and evaluation targets always use the declared rule.
The input rule uses only the input image, so it also works at inference.
Targets are normalized the same way, so models predict normalized fluorescence and absolute intensity is not evaluated.
Training items hold crops_per_image random crops of one image (shape K, C, P, P), evaluation items hold the full image.
Missing channels get a finite placeholder (zeros) and observed=False, they must be excluded from the loss.
With cfg.prepared_dir set, items come from the arrays written by prepare_dataset.py (float16, already normalized,
channel 0 input and channels 1 to 4 targets) and training crops are read from memory-mapped files.
"""
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, get_worker_info

import json

from config import ORGANELLES, ChallengeConfig, declared_rule, modality_id, train_target_rule


def normalize(a, p_low, p_high, clip=True):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(a)
    a = (a - lo) / (hi - lo)
    return (np.clip(a, 0.0, 1.0) if clip else a).astype(np.float32)


def prepared_rule(pdir):
    """Target rule a prepared folder was built with (prepared_info.json). Folders built before the file existed use the
    declared default rule (1, 99.5, clipped)."""
    f = pdir / "prepared_info.json"
    if not f.exists():
        return (1.0, 99.5, True)
    r = json.load(open(f))["target_rule"]
    return (float(r["p_low"]), float(r["p_high"]), bool(r["clip"]))


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
        self.target_rule = train_target_rule(cfg) if train else declared_rule(cfg)
        self.pdir = cfg.train_prepared_dir if (train and cfg.train_prepared_dir is not None) else cfg.prepared_dir
        if self.pdir is not None:
            if not self.pdir.is_dir():
                raise FileNotFoundError(f"prepared folder {self.pdir} not found (restore it, or pass --prepared-dir none)")
            built = prepared_rule(self.pdir)
            if built != self.target_rule:
                raise ValueError(f"{self.pdir} was built with target rule {built}, this dataset needs {self.target_rule}. "
                                 "Build a prepared folder with that rule (prepare_dataset.py --target-*) and pass it as "
                                 "train_prepared_dir, or read the raw TIFFs.")
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

    def _read(self, study, name, rule=None):
        a = tifffile.imread(self.cfg.data_root / f"Study_{study:02d}" / name)
        lo, hi, clip = declared_rule(self.cfg) if rule is None else rule
        return normalize(a, lo, hi, clip)

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
        return self._flip_rotate(x, y)

    def _flip_rotate(self, x, y):
        if self.rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        if self.rng.random() < 0.5:
            x, y = x[:, ::-1, :], y[:, ::-1, :]
        k = int(self.rng.integers(0, 4))
        return np.rot90(x, k, axes=(1, 2)).copy(), np.rot90(y, k, axes=(1, 2)).copy()

    def _augment_prepared(self, arr, extra=None):
        """Same random draws as _augment, but only the crop is read from the memory-mapped array. extra: optional
        (H, W) extra input channel (BM3D), cropped and transformed together with the input."""
        size = self.cfg.patch_size
        h, w = arr.shape[-2:]
        if h < size or w < size:
            x = arr[:1].astype(np.float32)
            if extra is not None:
                x = np.concatenate([x, np.asarray(extra, dtype=np.float32)[None]])
            return self._augment(x, arr[1:].astype(np.float32))
        i = int(self.rng.integers(0, h - size + 1))
        j = int(self.rng.integers(0, w - size + 1))
        crop = np.asarray(arr[:, i:i + size, j:j + size], dtype=np.float32)
        x = crop[:1]
        if extra is not None:
            x = np.concatenate([x, np.asarray(extra[i:i + size, j:j + size], dtype=np.float32)[None]])
        return self._flip_rotate(x, crop[1:])

    def _extra(self, stem):
        if not self.cfg.extra_input:
            return None
        if self.cfg.extra_input_dir is None:
            raise ValueError("extra_input needs extra_input_dir (bm3d_input.py build)")
        f = self.cfg.extra_input_dir / f"{stem}_{self.cfg.extra_input}.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing, run bm3d_input.py build for every prepared image")
        return np.load(f, mmap_mode="r")

    def _prepared_item(self, r):
        stem = f"{int(r.study):02d}_{int(r.image_id):05d}"
        arr = np.load(self.pdir / f"{stem}.npy", mmap_mode="r")
        assert arr.ndim == 3 and arr.shape[0] == 1 + len(ORGANELLES), f"bad prepared array {arr.shape}"
        extra = self._extra(stem)
        observed = np.array([bool(r[f"{o}_obs"]) for o in ORGANELLES])
        if self.train:
            crops = [self._augment_prepared(arr, extra) for _ in range(self.cfg.crops_per_image)]
            x = np.stack([c[0] for c in crops])
            y = np.stack([c[1] for c in crops])
        else:
            x, y = arr[:1].astype(np.float32), arr[1:].astype(np.float32)
            if extra is not None:
                x = np.concatenate([x, np.asarray(extra, dtype=np.float32)[None]])
        return x, y, observed

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        if self.pdir is not None:
            x, y, observed = self._prepared_item(r)
            return {
                "input": torch.from_numpy(np.ascontiguousarray(x)),
                "target": torch.from_numpy(np.ascontiguousarray(y)),
                "observed": torch.from_numpy(observed),
                "study": int(r.study),
                "image_id": int(r.image_id),
                "modality": r.modality,
                "modality_id": modality_id(r.modality),
            }
        if self.train and self.cfg.extra_input:
            raise ValueError("extra_input training needs the prepared dataset (--prepared-dir) and bm3d_input.py build")
        x = self._read(int(r.study), r.in_file)[None]
        y = np.zeros((len(ORGANELLES),) + x.shape[1:], dtype=np.float32)
        observed = np.zeros(len(ORGANELLES), dtype=bool)
        for k, o in enumerate(ORGANELLES):
            if bool(r[f"{o}_obs"]):
                t = self._read(int(r.study), r[f"{o}_file"], self.target_rule)
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
            "modality_id": modality_id(r.modality),
        }


def eval_dataset(cfg, split="val", heldout_json=None):
    """Full-image evaluation dataset. split "val" is the local validation split, "heldout" the training studies listed
    in a run's heldout_studies.json (written by train.py --select-on heldout), for choices that must not look at the
    reported validation images."""
    if split == "val":
        return LMC2Dataset(cfg, "val", train=False)
    if split != "heldout" or not heldout_json:
        raise ValueError("split must be val, or heldout with heldout_json")
    studies = {int(s) for s in json.load(open(heldout_json))["studies"]}
    ds = LMC2Dataset(cfg, "train", train=False)
    ds.df = ds.df[ds.df.study.isin(studies)].reset_index(drop=True)
    if not len(ds.df):
        raise ValueError(f"no training images in the held-out studies {sorted(studies)}")
    return ds
