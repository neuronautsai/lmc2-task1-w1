"""Checks for the Grand Challenge container (inference.py), used locally and in CI.

  python submission_check.py make-inputs INPUT_ROOT MODEL_DIR   synthetic OME-TIFF inputs and random-weight checkpoint
  python submission_check.py check-outputs INPUT_ROOT OUTPUT_ROOT  verify the four outputs per input, then panimg

The synthetic inputs test the file contract only (names, folders, shape, dtype, OME-XML, panimg readability).
Accepted output types: uint8 and float32 (inference.py LMC2_OUTPUT_MODE), the same type for every output.
They say nothing about prediction quality.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

from ome_utils import OUTPUT_TYPES, ome_with_type

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
            "axes": "YX", "PhysicalSizeX": 0.1625, "PhysicalSizeXUnit": "\u00b5m", "PhysicalSizeY": 0.1625,
            "PhysicalSizeYUnit": "\u00b5m", "Channel": {"Name": modality}})  # non-ASCII unit as in the challenge files
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unet import UNet
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save({"model": UNet(base=32, n_mod=0).state_dict(), "epoch": 0, "cfg": {"base_channels": 32, "modality_mode": "none"}},
               Path(model_dir) / "random_weights_for_ci.pt")
    print(f"wrote {len(CASES)} inputs to {d} and random weights to {model_dir}")


def run_panimg(output_root, previews):
    code = ("import sys, tempfile; from pathlib import Path; from panimg import convert\n"
            "with tempfile.TemporaryDirectory() as t:\n"
            "    kw = {} if sys.argv[2] == '1' else {'post_processors': []}\n"
            "    r = convert(input_directory=Path(sys.argv[1]), output_directory=Path(t), **kw)\n"
            "    print(len(r.new_images), sum(f.image_type.value == 'DZI' for f in r.new_image_files))")
    return subprocess.run([sys.executable, "-c", code, str(Path(output_root) / "images"), "1" if previews else "0"],
                          capture_output=True, text=True)


def check_outputs(input_root, output_root):
    inputs = sorted((Path(input_root) / INPUT_SUB).iterdir())
    errors, warnings, modes = [], [], set()
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
            mode = next((m for m, (dt, _, _) in OUTPUT_TYPES.items() if arr.dtype == np.dtype(dt)), None)
            if mode is None:
                errors.append(f"{out.name} {org}: dtype {arr.dtype}, expected one of {sorted(OUTPUT_TYPES)}")
                continue
            modes.add(mode)
            if ome_with_type(src_ome, *OUTPUT_TYPES[mode][1:]) != ome:
                errors.append(f"{out.name} {org}: OME-XML is not the input OME-XML with the pixel type of {mode}")
            if not np.isfinite(arr).all():
                errors.append(f"{out.name} {org}: non-finite values")
            if arr.min() == arr.max():
                errors.append(f"{out.name} {org}: constant image")
    n_out = sum(1 for _ in (Path(output_root) / "images").rglob("*.tiff"))
    if n_out != 4 * len(inputs):
        errors.append(f"expected {4 * len(inputs)} output files, found {n_out}")
    if len(modes) > 1:
        errors.append(f"mixed output types {sorted(modes)}")
    # panimg with its default post-processors (DZI preview), in a subprocess so that a libvips abort is caught.
    # For non-uint8 outputs a local libvips may abort in the preview step: then the read is repeated without
    # post-processors and the preview failure is reported as a warning, to be checked on Grand Challenge itself.
    p = run_panimg(output_root, previews=True)
    if p.returncode != 0 and modes and modes != {"uint8"}:
        warnings.append(f"panimg DZI preview failed locally for {sorted(modes)} outputs (exit {p.returncode}): "
                        f"{p.stderr.strip().splitlines()[-1:]}. Check on Grand Challenge before submitting.")
        p = run_panimg(output_root, previews=False)
    if p.returncode != 0:
        errors.append(f"panimg process failed (exit {p.returncode}): {p.stderr.strip().splitlines()[-1:]}")
    else:
        n_img, n_dzi = map(int, p.stdout.split()[-2:])
        if n_img != n_out:
            errors.append(f"panimg read {n_img} of {n_out} outputs")
        print(f"panimg: {n_img} images, {n_dzi} DZI previews")
    for w in warnings:
        print("WARNING: " + w)
    if errors:
        print("FAILED\n" + "\n".join(errors))
        return 1
    print(f"OK: {len(inputs)} inputs, {n_out} outputs, output type {sorted(modes)}, shape, dtype and OME-XML checked, "
          f"panimg read all outputs" + (" (without previews, see warning)" if warnings else ""))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make-inputs":
        make_inputs(sys.argv[2], sys.argv[3])
    elif cmd == "check-outputs":
        sys.exit(check_outputs(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
