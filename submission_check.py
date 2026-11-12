"""Checks for the Grand Challenge container (inference.py), used locally and in CI.

  python submission_check.py make-inputs INPUT_ROOT MODEL_DIR   synthetic OME-TIFF inputs and random-weight checkpoint
  python submission_check.py check-outputs INPUT_ROOT OUTPUT_ROOT  verify the four outputs per input, then panimg

The synthetic inputs test the file contract only (names, folders, shape, dtype, OME-XML, panimg readability).
They say nothing about prediction quality.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

from ome_utils import ome_as_uint8

ORGANELLES = ("nucleus", "mitochondria", "tubulin", "actin")
INPUT_SUB = Path("images") / "organelles-transmitted-light-ome-tiff"
CASES = [("case_bf_uint16.ome.tiff", "BF", np.uint16, (700, 950)),
         ("case_dic_uint8.ome.tiff", "DIC", np.uint8, (512, 512)),
         ("case_pc_uint16_small.ome.tiff", "PC", np.uint16, (300, 401))]


def make_inputs(input_root, model_dir):
    d = Path(input_root) / INPUT_SUB
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, modality, dtype, shape in CASES:
        img = rng.integers(0, np.iinfo(dtype).max, size=shape).astype(dtype)
        tifffile.imwrite(d / name, img, ome=True, metadata={
            "axes": "YX", "PhysicalSizeX": 0.1625, "PhysicalSizeY": 0.1625, "Channel": {"Name": modality}})
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unet import UNet
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save({"model": UNet(base=32, n_mod=0).state_dict(), "epoch": 0, "cfg": {"base_channels": 32, "modality_mode": "none"}},
               Path(model_dir) / "random_weights_for_ci.pt")
    print(f"wrote {len(CASES)} inputs to {d} and random weights to {model_dir}")


def check_outputs(input_root, output_root):
    inputs = sorted((Path(input_root) / INPUT_SUB).iterdir())
    errors = []
    for inp in inputs:
        with tifffile.TiffFile(inp) as t:
            src, src_ome = np.squeeze(t.asarray()), t.ome_metadata
        for org in ORGANELLES:
            out = Path(output_root) / "images" / f"{org}-fluorescence-ome-tiff" / inp.name
            if not out.exists():
                errors.append(f"missing {out}")
                continue
            with tifffile.TiffFile(out) as t:
                arr, ome = t.asarray(), t.ome_metadata
            if arr.shape != src.shape:
                errors.append(f"{out.name} {org}: shape {arr.shape} != input {src.shape}")
            if arr.dtype != np.uint8:
                errors.append(f"{out.name} {org}: dtype {arr.dtype}, expected uint8")
            if 'Type="uint8"' not in ome or ome_as_uint8(src_ome) != ome:
                errors.append(f"{out.name} {org}: OME-XML is not the input OME-XML with Type uint8")
            if arr.min() == arr.max():
                errors.append(f"{out.name} {org}: constant image")
    n_out = sum(1 for _ in (Path(output_root) / "images").rglob("*.tiff"))
    if n_out != 4 * len(inputs):
        errors.append(f"expected {4 * len(inputs)} output files, found {n_out}")
    # panimg with its default post-processors (DZI preview), in a subprocess so that a libvips abort is caught
    code = ("import sys, tempfile; from pathlib import Path; from panimg import convert\n"
            "with tempfile.TemporaryDirectory() as t:\n"
            "    r = convert(input_directory=Path(sys.argv[1]), output_directory=Path(t))\n"
            "    print(len(r.new_images), sum(f.image_type.value == 'DZI' for f in r.new_image_files))")
    p = subprocess.run([sys.executable, "-c", code, str(Path(output_root) / "images")], capture_output=True, text=True)
    if p.returncode != 0:
        errors.append(f"panimg process failed (exit {p.returncode}): {p.stderr.strip().splitlines()[-1:]}")
    else:
        n_img, n_dzi = map(int, p.stdout.split()[-2:])
        if n_img != n_out:
            errors.append(f"panimg read {n_img} of {n_out} outputs")
        print(f"panimg: {n_img} images, {n_dzi} DZI previews")
    if errors:
        print("FAILED\n" + "\n".join(errors))
        return 1
    print(f"OK: {len(inputs)} inputs, {n_out} outputs, shape, uint8 dtype and OME-XML checked, panimg read all outputs")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make-inputs":
        make_inputs(sys.argv[2], sys.argv[3])
    elif cmd == "check-outputs":
        sys.exit(check_outputs(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
