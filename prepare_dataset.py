"""Build and restore the prepared dataset used for fast training.

build:   one .npy per image, shape (5, H, W) float16. Channel 0 is the transmitted-light input and channels 1 to 4
         are Nucleus, Mitochondria, Tubulin, Actin (zeros where the target is missing). Every channel is already
         scaled by its own 1st and 99.5th percentile and clipped to [0, 1] with dataset.normalize (the declared rule).
         Only the best-z transmitted plane and the best-z targets listed in pairs.csv are read. Existing files are
         skipped, so an interrupted build can be restarted. Optional --shards-dir packs the files into tar shards.
         --target-p-low, --target-p-high and --no-target-clip build the targets with another rule (the input keeps the
         declared rule), for train.py --train-prepared-dir. The rule is written to prepared_info.json, and a build into
         a folder made with another rule is refused. --splits limits the build to some splits (default train and val).
restore: extract the tar shards from a folder (for example on Drive) into a local folder.

Usage:
  python prepare_dataset.py build --out /content/prepared [--shards-dir DIR --n-shards 8]
  python prepare_dataset.py build --out /content/prepared_p2 --splits train --target-p-low 2 --target-p-high 99.8 --no-target-clip
  python prepare_dataset.py restore --shards-dir DIR --out /content/prepared
"""
import argparse
import dataclasses
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from config import ORGANELLES, ChallengeConfig
from dataset import load_pairs, normalize, prepared_rule


def build_one(cfg, r, out, rule):
    path = out / f"{int(r.study):02d}_{int(r.image_id):05d}.npy"
    if path.exists():
        return path.stat().st_size
    base = cfg.data_root / f"Study_{int(r.study):02d}"
    x = normalize(tifffile.imread(base / r.in_file), cfg.p_low, cfg.p_high)
    arr = np.zeros((1 + len(ORGANELLES),) + x.shape, dtype=np.float16)
    arr[0] = x
    for k, o in enumerate(ORGANELLES):
        if bool(r[f"{o}_obs"]):
            t = normalize(tifffile.imread(base / r[f"{o}_file"]), *rule)
            assert t.shape == x.shape, f"shape mismatch {r.study} {r.image_id} {o}"
            arr[1 + k] = t
    assert np.isfinite(arr).all(), f"non-finite value in {path.name} (float16 overflow of an unclipped target?)"
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
    for name in ("prepared_index.csv", "prepared_info.json"):
        if (out / name).exists():
            (shards_dir / name).write_bytes((out / name).read_bytes())
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
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--target-p-low", type=float, default=None)
    ap.add_argument("--target-p-high", type=float, default=None)
    ap.add_argument("--no-target-clip", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "restore":
        t0 = time.time()
        for shard in sorted(Path(a.shards_dir).glob("*.tar")):
            with tarfile.open(shard) as tar:
                tar.extractall(out, filter="data")
        for name in ("prepared_index.csv", "prepared_info.json"):
            src = Path(a.shards_dir) / name
            if src.exists():
                (out / name).write_bytes(src.read_bytes())
        n = len(list(out.glob("*.npy")))
        print(f"restored {n} files in {time.time() - t0:.0f} s")
        return
    over = {}
    if a.data_root:
        over["data_root"] = Path(a.data_root)
    if a.pairs_csv:
        over["pairs_csv"] = Path(a.pairs_csv)
    cfg = dataclasses.replace(ChallengeConfig(), **over)
    rule = (float(cfg.p_low if a.target_p_low is None else a.target_p_low),
            float(cfg.p_high if a.target_p_high is None else a.target_p_high), not a.no_target_clip)
    info_path = out / "prepared_info.json"
    if info_path.exists() or any(out.glob("*.npy")):
        if prepared_rule(out) != rule:
            raise SystemExit(f"{out} holds files built with target rule {prepared_rule(out)}, not {rule}. Use another folder.")
    json.dump({"target_rule": {"p_low": rule[0], "p_high": rule[1], "clip": rule[2]},
               "input_rule": {"p_low": float(cfg.p_low), "p_high": float(cfg.p_high), "clip": True},
               "splits": a.splits}, open(info_path, "w"), indent=2)
    frames = []
    for split in a.splits:
        frames.append(load_pairs(cfg, split))
    df = pd.concat(frames, ignore_index=True)
    rows = [r for _, r in df.iterrows()]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        sizes = list(ex.map(lambda r: build_one(cfg, r, out, rule), rows))
    keep = ["image_id", "study", "split", "modality", "in_shape"] + [f"{o}_obs" for o in ORGANELLES]
    df[[c for c in keep if c in df.columns]].to_csv(out / "prepared_index.csv", index=False)
    print(f"images {len(rows)} | total {sum(sizes) / 1e9:.2f} GB | mean {np.mean(sizes) / 1e6:.1f} MB "
          f"| max {max(sizes) / 1e6:.1f} MB | {time.time() - t0:.0f} s")
    if a.shards_dir:
        make_shards(out, Path(a.shards_dir), a.n_shards)


if __name__ == "__main__":
    main()
