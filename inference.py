"""Grand Challenge entry point for Light My Cells 2, Task 1 (Best Focus).

Reads every OME-TIFF in /input/images/organelles-transmitted-light-ome-tiff, predicts the four organelles with the tiled
U-Net and writes one OME-TIFF per organelle to /output/images/<organelle>-fluorescence-ome-tiff/<same file name>,
following the organizers' template (input OME-XML copied to the output, tiled TIFF written with tifffile).

Preprocessing is the training rule: the input is scaled by its own 1st and 99.5th percentile and clipped to [0, 1].
The network output p is clipped to [0, 1] and written in one of two modes (environment variable LMC2_OUTPUT_MODE):
  uint8    round(255 p), the v0.1.0 behaviour.
  float32  lo + p (hi - lo), with lo and hi per organelle from output_scale.json (typical raw training target p2 and
           p99.8). SSIM, PCC and cosine distance do not depend on this mapping (the evaluator normalizes each image),
           but the portal MAE and Euclidean distance appear to be computed on the raw scale (compare_output_scale.py).
The pixel type in the copied OME-XML is set to the written type. Note: panimg, which reads every output on Grand
Challenge, builds a DZI preview with libvips. A local test with libvips 8.15 aborted on uint16 and float32 outputs and
passed on uint8, so the float32 mode must be checked on Grand Challenge itself before it is submitted.

Weights: every *.pt file in /opt/ml/model (the model tarball uploaded to Grand Challenge), else in ./resources. A file
may hold a full training checkpoint ({"model", "cfg", ...}) or a bare state_dict. Several files are an ensemble: the
predictions are averaged in [0, 1] before the output mapping.

Optional files next to the weights:
  members.json      {"Nucleus": ["a.pt", "b.pt"], ...}: the weight files averaged for each organelle (#333 item 3).
                    Organelles not listed use every weight file. Names are paths relative to the weights folder.
  postprocess.json  per-organelle post-processing applied in [0, 1] before the output mapping (#333 item 9), same
                    format as postprocess.py writes (postprocess_selected.json).
For each input the log also records the z position fields of the OME-XML and a sharpness value (variance of the
Laplacian of the normalized input), for the focal-plane check of #333 item 1.

Tiling and test-time augmentation (environment variables):
  LMC2_WINDOW   tile blending weight, hann (default) or linear (the v0.1.0 behaviour), see predict.py.
  LMC2_OVERLAP  tile overlap in px, 128 by default (v0.1.0: 64).
  LMC2_TTA      1 (default, none), 4 (flips) or 8 (flips and 90-degree rotations), averaged per model.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import xmltodict

from config import ORGANELLES, ChallengeConfig
from dataset import normalize
from ome_utils import OUTPUT_TYPES, ome_with_type
from postprocess import apply_all, check_plan
from predict import TTA_TRANSFORMS, WINDOWS, predict_tta
from unet import load_any_checkpoint

INPUT_DIR = Path(os.environ.get("LMC2_INPUT", "/input")) / "images" / "organelles-transmitted-light-ome-tiff"
OUTPUT_ROOT = Path(os.environ.get("LMC2_OUTPUT", "/output")) / "images"
OUTPUT_MODE = os.environ.get("LMC2_OUTPUT_MODE", "uint8")
WINDOW = os.environ.get("LMC2_WINDOW", "hann")
OVERLAP = int(os.environ.get("LMC2_OVERLAP", "128"))
N_TTA = int(os.environ.get("LMC2_TTA", "1"))
SCALE_FILE = Path(__file__).resolve().parent / "output_scale.json"
MODEL_DIRS = [Path(os.environ.get("LMC2_MODEL_DIR", "/opt/ml/model")), Path(__file__).resolve().parent / "resources"]
# Network output order (config.ORGANELLES): Nucleus, Mitochondria, Tubulin, Actin
OUTPUT_ORDER = ("Nucleus", "Mitochondria", "Tubulin", "Actin")
assert tuple(ORGANELLES) == OUTPUT_ORDER, "network channel order changed"


def find_weights():
    for d in MODEL_DIRS:
        if d.is_dir():
            found = sorted(d.rglob("*.pt"))
            if found:
                return d, found
    raise FileNotFoundError(f"no *.pt weights in {[str(d) for d in MODEL_DIRS]}")


def load_plan(model_dir, weights):
    """Members per organelle (indices into weights) and post-processing settings, from the optional JSON files."""
    names = [str(p.relative_to(model_dir)) for p in weights]
    members = {o: list(range(len(weights))) for o in OUTPUT_ORDER}
    f = model_dir / "members.json"
    if f.exists():
        for organelle, files in json.load(open(f)).items():
            if organelle not in OUTPUT_ORDER:
                raise ValueError(f"members.json: unknown organelle {organelle!r}")
            missing = [n for n in files if n not in names]
            if missing or not files:
                raise ValueError(f"members.json: {organelle} lists {files}, weight files are {names}")
            members[organelle] = [names.index(n) for n in files]
    f = model_dir / "postprocess.json"
    plan = check_plan(json.load(open(f))) if f.exists() else {}
    return names, members, plan


def focus_info(ome, x):
    """z fields of the first OME Plane and Pixels, and the variance of the Laplacian of the normalized input."""
    info = {}
    try:
        pixels = xmltodict.parse(ome)["OME"]["Image"]["Pixels"]
        info["SizeZ"] = pixels.get("@SizeZ")
        plane = pixels.get("Plane")
        plane = plane[0] if isinstance(plane, list) else plane
        if isinstance(plane, dict):
            info.update({k[1:]: v for k, v in plane.items() if k in ("@TheZ", "@PositionZ", "@PositionZUnit")})
    except Exception as e:  # the log entry must never stop the prediction
        info["ome_error"] = type(e).__name__
    lap = x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1]
    info["laplacian_var"] = round(float(lap.var()), 6)
    return info


def load_model(path, device):
    return load_any_checkpoint(path, device)


def predict_ensemble(models, x, cfg, members=None):
    """Mean prediction in [0, 1]. members: {organelle: [model indices]}, default every model for every organelle."""
    used = sorted({i for idx in (members or {}).values() for i in idx}) if members else list(range(len(models)))
    preds = {i: predict_tta(models[i], x, cfg.tile_size, OVERLAP, window=WINDOW, n_tta=N_TTA) for i in used}
    if not members:
        return np.mean([preds[i] for i in used], axis=0)
    out = np.empty_like(preds[used[0]])
    for k, organelle in enumerate(OUTPUT_ORDER):
        out[k] = np.mean([preds[i][k] for i in members[organelle]], axis=0)
    return out


def read_image(path):
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        ome = tif.ome_metadata
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"{path.name}: expected one 2D plane, got shape {image.shape}")
    return image, ome


def to_uint8(pred):
    return np.round(np.clip(pred, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_scale():
    if OUTPUT_MODE == "uint8":
        return None
    scale = json.load(open(SCALE_FILE))
    for organelle in OUTPUT_ORDER:
        lo, hi = float(scale[organelle]["lo"]), float(scale[organelle]["hi"])
        assert 0 <= lo < hi, f"bad output scale for {organelle}: {lo}, {hi}"
    return scale


def to_output(pred, organelle, scale):
    if OUTPUT_MODE == "uint8":
        return to_uint8(pred)
    lo, hi = float(scale[organelle]["lo"]), float(scale[organelle]["hi"])
    return (lo + np.clip(pred, 0.0, 1.0) * (hi - lo)).astype(np.float32)


def save_image(path, array, ome):
    # Same call as the organizers' template inference.py, except that the OME-XML is passed as UTF-8 bytes:
    # the challenge files contain non-ASCII units (the micro sign in "µm"), and tifffile rejects a str description
    # that is not 7-bit ASCII.
    pixels = xmltodict.parse(ome)["OME"]["Image"]["Pixels"]
    tifffile.imwrite(path, array, description=ome.encode("utf-8"),
                     resolution=(float(pixels["@PhysicalSizeX"]), float(pixels["@PhysicalSizeY"])),
                     metadata=pixels, tile=(128, 128))


def run():
    if OUTPUT_MODE not in OUTPUT_TYPES:
        raise ValueError(f"LMC2_OUTPUT_MODE must be one of {sorted(OUTPUT_TYPES)}, got {OUTPUT_MODE!r}")
    if WINDOW not in WINDOWS:
        raise ValueError(f"LMC2_WINDOW must be one of {WINDOWS}, got {WINDOW!r}")
    if N_TTA not in TTA_TRANSFORMS:
        raise ValueError(f"LMC2_TTA must be one of {sorted(TTA_TRANSFORMS)}, got {N_TTA}")
    scale = load_scale()
    cfg = ChallengeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, weights = find_weights()
    models = [load_model(p, device) for p in weights]
    names, members, plan = load_plan(model_dir, weights)
    print(f"output mode {OUTPUT_MODE} | models {len(models)} | window {WINDOW} | overlap {OVERLAP} | tta {N_TTA}", flush=True)
    print("members per organelle:", json.dumps({o: [names[i] for i in m] for o, m in members.items()}), flush=True)
    print("post-processing:", json.dumps(plan) if plan else "none", flush=True)
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        raise FileNotFoundError(f"no input images in {INPUT_DIR}")
    for organelle in OUTPUT_ORDER:
        (OUTPUT_ROOT / f"{organelle.lower()}-fluorescence-ome-tiff").mkdir(parents=True, exist_ok=True)
    for path in files:
        t0 = time.time()
        raw, ome = read_image(path)
        xn = normalize(raw.astype(np.float32), cfg.p_low, cfg.p_high)
        print(f"{path.name} focus {json.dumps(focus_info(ome, xn))}", flush=True)
        x = torch.from_numpy(xn)[None, None].to(device)
        pred = predict_ensemble(models, x, cfg, members)
        if plan:
            pred = apply_all(pred, plan)
        assert pred.shape == (4,) + raw.shape, f"prediction shape {pred.shape} for input {raw.shape}"
        out_ome = ome_with_type(ome, *OUTPUT_TYPES[OUTPUT_MODE][1:])
        for k, organelle in enumerate(OUTPUT_ORDER):
            out = OUTPUT_ROOT / f"{organelle.lower()}-fluorescence-ome-tiff" / path.name
            save_image(out, to_output(pred[k], organelle, scale), out_ome)
        print(f"{path.name} {raw.shape} {raw.dtype} -> 4 outputs in {time.time() - t0:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
