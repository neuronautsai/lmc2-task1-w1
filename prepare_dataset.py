"""Build and restore the prepared dataset used for fast training.

build:   one .npy per image, shape (5, H, W) float16. Channel 0 is the transmitted-light input and channels 1 to 4
         are Nucleus, Mitochondria, Tubulin, Actin (zeros where the target is missing). Every channel is already
         scaled by its own 1st and 99.5th percentile and clipped to [0, 1] with dataset.normalize (the declared rule).
         Only the best-z transmitted plane and the best-z targets listed in pairs.csv are read. Existing files are
         skipped, so an interrupted build can be restarted. Optional --shards-dir packs the files into tar shards.
restore: extract the tar shards from a folder (for example on Drive) into a local folder.

Usage:
  python prepare_dataset.py build --out /content/prepared [--shards-dir DIR --n-shards 8]
  python prepare_dataset.py restore --shards-dir DIR --out /content/prepared
"""
import argparse
import dataclasses
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from config import ORGANELLES, ChallengeConfig
from dataset import load_pairs, normalize


def build_one(cfg, r, out):
    path = out / f"{int(r.study):02d}_{int(r.image_id):05d}.npy"
    if path.exists():
        return path.stat().st_size
    base = cfg.data_root / f"Study_{int(r.study):02d}"
    x = normalize(tifffile.imread(base / r.in_file), cfg.p_low, cfg.p_high)
    arr = np.zeros((1 + len(ORGANELLES),) + x.shape, dtype=np.float16)
    arr[0] = x
    for k, o in enumerate(ORGANELLES):
        if bool(r[f"{o}_obs"]):
            t = normalize(tifffile.imread(base / r[f"{o}_file"]), cfg.p_low, cfg.p_high)
            assert t.shape == x.shape, f"shape mismatch {r.study} {r.image_id} {o}"
            arr[1 + k] = t
    assert np.isfinite(arr).all()
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return path.stat().st_size


def make_shards(out, shards_dir, n_shards):
    files = sorted(out.glob("*.npy"))
    shards_dir.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        with tarfile.open(shards_dir / f"prepared_shard_{s:02d}.tar", "w") as tar:
            for f in files[s::n_shards]:
                tar.add(f, arcname=f.name)
    index = out / "prepared_index.csv"
    if index.exists():
        (shards_dir / "prepared_index.csv").write_bytes(index.read_bytes())
    sizes = [p.stat().st_size for p in sorted(shards_dir.glob("*.tar"))]
    print(f"{n_shards} shards, total {sum(sizes) / 1e9:.2f} GB in {shards_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "restore"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--n-shards", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "restore":
        t0 = time.time()
        for shard in sorted(Path(a.shards_dir).glob("*.tar")):
            with tarfile.open(shard) as tar:
                tar.extractall(out, filter="data")
        idx = Path(a.shards_dir) / "prepared_index.csv"
        if idx.exists():
            (out / "prepared_index.csv").write_bytes(idx.read_bytes())
        n = len(list(out.glob("*.npy")))
        print(f"restored {n} files in {time.time() - t0:.0f} s")
        return
    over = {}
    if a.data_root:
        over["data_root"] = Path(a.data_root)
    if a.pairs_csv:
        over["pairs_csv"] = Path(a.pairs_csv)
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    frames = []
    for split in ("train", "val"):
        frames.append(load_pairs(cfg, split))
    df = pd.concat(frames, ignore_index=True)
    rows = [r for _, r in df.iterrows()]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        sizes = list(ex.map(lambda r: build_one(cfg, r, out), rows))
    keep = ["image_id", "study", "split", "modality", "in_shape"] + [f"{o}_obs" for o in ORGANELLES]
    df[[c for c in keep if c in df.columns]].to_csv(out / "prepared_index.csv", index=False)
    print(f"images {len(rows)} | total {sum(sizes) / 1e9:.2f} GB | mean {np.mean(sizes) / 1e6:.1f} MB "
          f"| max {max(sizes) / 1e6:.1f} MB | {time.time() - t0:.0f} s")
    if a.shards_dir:
        make_shards(out, Path(a.shards_dir), a.n_shards)


if __name__ == "__main__":
    main()
