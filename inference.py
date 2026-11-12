"""Grand Challenge entry point for Light My Cells 2, Task 1 (Best Focus).

Reads every OME-TIFF in /input/images/organelles-transmitted-light-ome-tiff, predicts the four organelles with the tiled
U-Net and writes one OME-TIFF per organelle to /output/images/<organelle>-fluorescence-ome-tiff/<same file name>,
following the organizers' template (input OME-XML copied to the output, tiled TIFF written with tifffile).

Preprocessing is the training rule: the input is scaled by its own 1st and 99.5th percentile and clipped to [0, 1].
The network output is in [0, 1] and is written as uint8 (0 to 255), with the pixel type in the copied OME-XML set to
uint8. The evaluator normalizes every prediction by its own percentiles, so the scaling does not change the scores, and
the 8-bit rounding changes them by a negligible amount (checked separately). uint8 is used because panimg, which reads
every output on Grand Challenge, builds a DZI preview with libvips, and a local test with libvips 8.15 aborted on 16-bit
tiled TIFFs while uint8 passed.

Weights: the first *.pt file in /opt/ml/model (the model tarball uploaded to Grand Challenge), else in ./resources.
A file may hold a full training checkpoint ({"model", "cfg", ...}) or a bare state_dict.
"""
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
from ome_utils import ome_as_uint8
from predict import predict_full
from unet import UNet

INPUT_DIR = Path(os.environ.get("LMC2_INPUT", "/input")) / "images" / "organelles-transmitted-light-ome-tiff"
OUTPUT_ROOT = Path(os.environ.get("LMC2_OUTPUT", "/output")) / "images"
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


def save_image(path, array, ome):
    # Same call as the organizers' template inference.py
    pixels = xmltodict.parse(ome)["OME"]["Image"]["Pixels"]
    tifffile.imwrite(path, array, description=ome,
                     resolution=(float(pixels["@PhysicalSizeX"]), float(pixels["@PhysicalSizeY"])),
                     metadata=pixels, tile=(128, 128))


def run():
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
        out_ome = ome_as_uint8(ome)
        for k, organelle in enumerate(OUTPUT_ORDER):
            out = OUTPUT_ROOT / f"{organelle.lower()}-fluorescence-ome-tiff" / path.name
            save_image(out, to_uint8(pred[k]), out_ome)
        print(f"{path.name} {raw.shape} {raw.dtype} -> 4 outputs in {time.time() - t0:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
