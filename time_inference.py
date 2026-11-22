"""Time the container prediction on the largest case before choosing the ensemble and TTA setting.

Runs the same code path as inference.py (normalization, Hann tiling, TTA, ensemble mean, uint8 conversion) on one
synthetic uint16 image of --size x --size pixels and reports seconds per case for every TTA setting asked for, plus the
peak GPU memory and the process peak RAM. Weights: the given checkpoints, else --n-models random-weight U-Nets (the
run time does not depend on the weight values).

Usage:
  python time_inference.py --checkpoint W3/base_s42/best.pt W3/base_s123/best.pt W3/base_s456/best.pt --tta 1 4 8
  python time_inference.py --n-models 3 --tta 1 4 8 --device cpu
"""
import argparse
import platform
import resource
import time

import numpy as np
import torch

from config import ChallengeConfig
from dataset import normalize
from predict import predict_tta
from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--n-models", type=int, default=3, help="random-weight models when no checkpoint is given")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--tta", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--window", default="hann")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    cfg = ChallengeConfig()
    device = torch.device(a.device)
    models = []
    if a.checkpoint:
        for c in a.checkpoint:
            ck = torch.load(c, map_location=device, weights_only=False)
            m = UNet(base=int(ck.get("cfg", {}).get("base_channels", 32)), n_mod=0).to(device)
            m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
            models.append(m.eval())
    else:
        for seed in range(a.n_models):
            torch.manual_seed(seed)
            models.append(UNet(base=32, n_mod=0).to(device).eval())
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu"
    print(f"device {device} ({name}) | torch {torch.__version__} | cpu threads {torch.get_num_threads()} | "
          f"models {len(models)} | size {a.size} | window {a.window} | overlap {a.overlap}", flush=True)
    raw = np.random.default_rng(0).integers(0, 65535, size=(a.size, a.size)).astype(np.uint16)
    for n_tta in a.tta:
        times = []
        for _ in range(a.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.time()
            x = torch.from_numpy(normalize(raw.astype(np.float32), cfg.p_low, cfg.p_high))[None, None].to(device)
            acc = None
            for m in models:
                p = predict_tta(m, x, cfg.tile_size, a.overlap, window=a.window, n_tta=n_tta)
                acc = p if acc is None else acc + p
            out = np.round(np.clip(acc / len(models), 0, 1) * 255).astype(np.uint8)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.time() - t0)
        gpu = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else float("nan")
        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # kB on Linux
        print(f"tta {n_tta}: {np.mean(times):.1f} s per case (min {min(times):.1f}, {len(times)} runs) | "
              f"peak GPU {gpu:.2f} GB | peak RAM {ram:.2f} GB | output {out.shape} {out.dtype}", flush=True)


if __name__ == "__main__":
    main()
