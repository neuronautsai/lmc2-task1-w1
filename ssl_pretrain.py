"""Masked-image pretraining of the pretrained-encoder U-Net's encoder (#333 Week 2, Run B).

Data: every transmitted-light slice (all focal planes) of the training-split studies, taken from the data manifest
(kind == "transmitted") and results/split_manifest.csv. Validation studies and challenge inputs are never read.
--exclude-heldout RUN/heldout_studies.json also leaves out the studies a fine-tuning run holds out for selection.
Each sample: one random slice, normalized by its own 1st and 99.5th percentile and clipped (the training input rule),
--crops-per-file random crops of --patch px (flips and 90-degree rotations), --mask-ratio of the 32 px patches set to 0.
An smp.Unet with the same encoder and one output channel reconstructs the crop, the loss is L1 on the masked pixels only.
The encoder starts from --init (imagenet by default, the Run A starting point, or none).

Output: OUT/ssl_encoder.pt ({"encoder": state_dict, "info": {...}}) for train.py --ssl-checkpoint, OUT/history.csv,
OUT/last.pt for --resume, OUT/ssl_info.json with the settings, the number of slices and the pretraining time.

Usage:
  python ssl_pretrain.py --manifest data_manifest_full.csv --out OUT/ssl_r34 --encoder resnet34 --steps 10000 --resume
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from config import ChallengeConfig
from dataset import normalize

SPLIT_MANIFEST = Path(os.environ.get("LMC2_SPLIT_MANIFEST", Path(__file__).resolve().parent / "results" / "split_manifest.csv"))


def slice_list(manifest, data_root, exclude_heldout=None):
    man = pd.read_csv(manifest)
    tl = man[man.kind == "transmitted"]
    if "on_drive" in tl:
        tl = tl[tl.on_drive.astype(bool)]
    sm = pd.read_csv(SPLIT_MANIFEST)
    train_studies = set(sm[sm.split == "train"].study.astype(int))
    val_studies = set(sm[sm.split != "train"].study.astype(int))
    assert not train_studies & val_studies, "a study is in both splits"
    if exclude_heldout:
        train_studies -= {int(s) for s in json.load(open(exclude_heldout))["studies"]}
    tl = tl[tl.study.astype(int).isin(train_studies)]
    paths = [str(Path(data_root) / f"Study_{int(s):02d}" / f) for s, f in zip(tl.study, tl.filename)]
    return paths, sorted(train_studies)


class SliceCrops(Dataset):
    def __init__(self, paths, n_items, patch, crops, p_low, p_high, seed):
        self.paths, self.n, self.patch, self.crops = paths, n_items, patch, crops
        self.p_low, self.p_high, self.seed = p_low, p_high, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        for _ in range(5):  # a slice smaller than the patch or a failed Drive read: draw another file
            try:
                a = np.squeeze(tifffile.imread(self.paths[int(rng.integers(len(self.paths)))]))
            except OSError:
                continue
            if a.ndim == 2 and min(a.shape) >= self.patch:
                break
        else:
            raise RuntimeError("no readable slice of the patch size after 5 draws")
        a = normalize(a, self.p_low, self.p_high)
        out = []
        for _ in range(self.crops):
            y = int(rng.integers(0, a.shape[0] - self.patch + 1))
            x = int(rng.integers(0, a.shape[1] - self.patch + 1))
            c = a[y:y + self.patch, x:x + self.patch]
            c = np.rot90(c[:, ::-1] if rng.random() < 0.5 else c, int(rng.integers(4)))
            out.append(np.ascontiguousarray(c))
        return torch.from_numpy(np.stack(out)[:, None])


def random_mask(b, size, cell, ratio, gen, device):
    g = size // cell
    n_mask = int(round(ratio * g * g))
    scores = torch.rand(b, g * g, generator=gen)
    m = torch.zeros(b, g * g)
    m.scatter_(1, scores.argsort(dim=1)[:, :n_mask], 1.0)
    m = m.view(b, 1, g, g).repeat_interleave(cell, 2).repeat_interleave(cell, 3)
    return m.to(device)


def main():
    import segmentation_models_pytorch as smp
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="data_manifest_full.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--init", default="imagenet", help="imagenet or none")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32, help="crops per step")
    ap.add_argument("--crops-per-file", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--mask-cell", type=int, default=32)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-heldout", default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert a.batch % a.crops_per_file == 0, "--batch must be a multiple of --crops-per-file"
    cfg = ChallengeConfig(**({"data_root": Path(a.data_root)} if a.data_root else {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths, studies = slice_list(a.manifest, cfg.data_root, a.exclude_heldout)
    print(f"{len(paths)} transmitted-light slices from {len(studies)} training studies", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    init = None if a.init in ("none", "None") else a.init
    model = smp.Unet(encoder_name=a.encoder, encoder_weights=init, in_channels=1, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    step, hist, elapsed = 0, [], 0.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, hist, elapsed = ck["step"], ck["history"], ck["elapsed_s"]
        print(f"resumed at step {step}", flush=True)
    files_per_step = a.batch // a.crops_per_file
    ds = SliceCrops(paths, (a.steps - step) * files_per_step, a.patch, a.crops_per_file, cfg.p_low, cfg.p_high,
                    a.seed + step)
    loader = DataLoader(ds, batch_size=files_per_step, num_workers=a.workers, persistent_workers=a.workers > 0)
    gen = torch.Generator().manual_seed(a.seed + step)

    def save(final=False):
        info = {"encoder": a.encoder, "init": a.init, "steps": step, "batch": a.batch, "patch": a.patch,
                "mask_cell": a.mask_cell, "mask_ratio": a.mask_ratio, "lr": a.lr, "n_slices": len(paths),
                "train_studies": studies, "exclude_heldout": a.exclude_heldout, "pretraining_hours": elapsed / 3600,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": hist,
                    "elapsed_s": elapsed}, out / "last.pt")
        torch.save({"encoder": model.encoder.state_dict(), "info": info}, out / "ssl_encoder.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if final:
            json.dump(info, open(out / "ssl_info.json", "w"), indent=2)

    model.train()
    t0, run = time.time(), []
    for batch in loader:
        x = batch.view(-1, 1, a.patch, a.patch).to(device, non_blocking=True)
        m = random_mask(x.shape[0], a.patch, a.mask_cell, a.mask_ratio, gen, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            rec = model(x * (1 - m))
        loss = ((rec.float() - x).abs() * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append(loss.item())
        step += 1
        if step % 100 == 0 or step == a.steps:
            elapsed += time.time() - t0
            t0 = time.time()
            hist.append({"step": step, "masked_l1": float(np.mean(run)), "elapsed_h": elapsed / 3600})
            run = []
            print(f"step {step}/{a.steps} masked L1 {hist[-1]['masked_l1']:.4f} | {elapsed / 3600:.2f} h", flush=True)
        if step % a.save_every == 0:
            save()
        if step >= a.steps:
            break
    elapsed += time.time() - t0
    save(final=True)
    print(f"done: {step} steps, {elapsed / 3600:.2f} h, encoder in {out / 'ssl_encoder.pt'}", flush=True)


if __name__ == "__main__":
    main()
