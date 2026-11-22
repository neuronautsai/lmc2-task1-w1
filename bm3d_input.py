"""BM3D-denoised copy of the transmitted-light input as a second input channel (#333 item 11).

The denoised channel is computed on the normalized input (1st to 99.5th percentile, clipped, the training input rule)
with the bm3d package, noise level from skimage.restoration.estimate_sigma on the same image. predict.py adds it at
inference for a model trained with --extra-input bm3d. The container needs the bm3d package in that case
(requirements-submission.txt does not list it).

  time   seconds per image, on given TIFF files (--image) or on a synthetic image (--size), the cost added to every case
  build  BM3D channel for every array of a prepared folder, written as <stem>_bm3d.npy (float16) to --out, resumable

Usage:
  python bm3d_input.py time --image IMG1.tif IMG2.tif
  python bm3d_input.py time --size 2048
  python bm3d_input.py build --prepared-dir /content/prepared --out DRIVE/bm3d --workers 8
  python train.py ... --extra-input bm3d --extra-input-dir DRIVE/bm3d
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STAGES = ("all", "hard")


def bm3d_denoise(xn, stage="all"):
    """xn: 2D normalized input in [0, 1]. Returns the BM3D estimate in [0, 1] as float32."""
    import bm3d
    from skimage.restoration import estimate_sigma
    xn = np.asarray(xn, dtype=np.float32)
    sigma = max(float(estimate_sigma(xn)), 1e-3)
    st = bm3d.BM3DStages.ALL_STAGES if stage == "all" else bm3d.BM3DStages.HARD_THRESHOLDING
    return np.clip(bm3d.bm3d(xn, sigma_psd=sigma, stage_arg=st), 0.0, 1.0).astype(np.float32)


def _build_one(args):
    src, dst, stage = args
    if dst.exists():
        return 0.0
    t0 = time.time()
    x = np.load(src, mmap_mode="r")[0].astype(np.float32)
    d = bm3d_denoise(x, stage)
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    np.save(tmp, d.astype(np.float16))
    os.replace(tmp, dst)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument("--image", nargs="*", default=[])
    t.add_argument("--size", type=int, default=2048)
    t.add_argument("--stage", choices=STAGES, default="all")
    b = sub.add_parser("build")
    b.add_argument("--prepared-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=os.cpu_count())
    b.add_argument("--stage", choices=STAGES, default="all")
    a = ap.parse_args()
    if a.cmd == "time":
        from config import ChallengeConfig
        from dataset import normalize
        import tifffile
        cfg = ChallengeConfig()
        imgs = [(p, np.squeeze(tifffile.imread(p))) for p in a.image] or \
               [(f"synthetic {a.size}x{a.size}", np.random.default_rng(0).integers(0, 65535, (a.size, a.size)))]
        print(f"cpu count {os.cpu_count()} | stage {a.stage}", flush=True)
        for name, raw in imgs:
            xn = normalize(raw.astype(np.float32), cfg.p_low, cfg.p_high)
            t0 = time.time()
            bm3d_denoise(xn, a.stage)
            dt = time.time() - t0
            print(f"{name} {raw.shape}: {dt:.1f} s, {dt / (raw.size / 2 ** 20):.1f} s per megapixel "
                  f"({dt * 2048 * 2048 / raw.size:.1f} s scaled to 2048 x 2048)", flush=True)
        return
    src = sorted(p for p in Path(a.prepared_dir).glob("*.npy") if "_" in p.stem and not p.stem.endswith("_bm3d"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / f"{p.stem}_bm3d.npy", a.stage) for p in src]
    todo = [j for j in jobs if not j[1].exists()]
    print(f"{len(jobs)} prepared arrays, {len(todo)} to denoise with {a.workers} workers", flush=True)
    t0, times = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, dt in enumerate(ex.map(_build_one, todo), 1):
            times.append(dt)
            if n % 20 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} | {np.mean(times):.1f} s per image per worker | {(time.time() - t0) / 3600:.2f} h",
                      flush=True)
    info = {"stage": a.stage, "sigma": "skimage estimate_sigma per image", "n": len(jobs),
            "mean_s_per_image_per_worker": float(np.mean(times)) if times else None, "workers": a.workers}
    json.dump(info, open(out / "bm3d_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
