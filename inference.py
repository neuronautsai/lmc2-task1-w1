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

Weights: the first *.pt file in /opt/ml/model (the model tarball uploaded to Grand Challenge), else in ./resources.
A file may hold a full training checkpoint ({"model", "cfg", ...}) or a bare state_dict.
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
from predict import predict_full
from unet import UNet

INPUT_DIR = Path(os.environ.get("LMC2_INPUT", "/input")) / "images" / "organelles-transmitted-light-ome-tiff"
OUTPUT_ROOT = Path(os.environ.get("LMC2_OUTPUT", "/output")) / "images"
OUTPUT_MODE = os.environ.get("LMC2_OUTPUT_MODE", "uint8")
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
                return found[0]
    raise FileNotFoundError(f"no *.pt weights in {[str(d) for d in MODEL_DIRS]}")


def load_model(device):
    path = find_weights()
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = ck.get("cfg", {}) if isinstance(ck, dict) else {}
    if cfg.get("modality_mode", "none") != "none":
        raise ValueError("this entry point serves the unconditioned U-Net only")
    model = UNet(base=int(cfg.get("base_channels", 32)), n_mod=0).to(device)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    print(f"weights {path} | epoch {ck.get('epoch') if isinstance(ck, dict) else 'n/a'} | device {device}", flush=True)
    return model


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
    scale = load_scale()
    print(f"output mode {OUTPUT_MODE}", flush=True)
    cfg = ChallengeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        raise FileNotFoundError(f"no input images in {INPUT_DIR}")
    for organelle in OUTPUT_ORDER:
        (OUTPUT_ROOT / f"{organelle.lower()}-fluorescence-ome-tiff").mkdir(parents=True, exist_ok=True)
    for path in files:
        t0 = time.time()
        raw, ome = read_image(path)
        x = torch.from_numpy(normalize(raw.astype(np.float32), cfg.p_low, cfg.p_high))[None, None].to(device)
        pred = predict_full(model, x, cfg.tile_size, cfg.tile_overlap)
        assert pred.shape == (4,) + raw.shape, f"prediction shape {pred.shape} for input {raw.shape}"
        out_ome = ome_with_type(ome, *OUTPUT_TYPES[OUTPUT_MODE][1:])
        for k, organelle in enumerate(OUTPUT_ORDER):
            out = OUTPUT_ROOT / f"{organelle.lower()}-fluorescence-ome-tiff" / path.name
            save_image(out, to_output(pred[k], organelle, scale), out_ome)
        print(f"{path.name} {raw.shape} {raw.dtype} -> 4 outputs in {time.time() - t0:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
