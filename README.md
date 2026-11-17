# Light My Cells 2, Task 1 (Best Focus)

Code for Task 1 of the [Light My Cells 2](https://lightmycells2.grand-challenge.org/) challenge (France-BioImaging, Grand Challenge): predict the best-focus fluorescence images of four structures (Nucleus, Mitochondria, Tubulin, Actin) from one label-free transmitted-light image (bright-field, phase contrast or DIC). This is also called in-silico labeling. Task 2 of the challenge (maximum-intensity projection from a stack) is not covered here.

The work is reported in the task issue (itu-biai/dlct-docs #323). A copy of every result comment, with tables, commits and commands, is in [`results/issue323/`](results/issue323/README.md).

## Current state

- Method: 2D U-Net (32 base channels), one transmitted-light plane in, four fluorescence channels out, observed-channel L1 loss, AdamW, 50 epochs, inference on 512 px tiles.
- Weeks 1 to 3: data inventory, frozen study-grouped split, baseline and two tested changes (cosine learning-rate decay, modality conditioning). Neither change beat the baseline over 3 seeds.
- Review items after Week 3:
  1. Duplicated targets: Study 3 Actin excluded (`check_duplicates.py`).
  2. Output intensity scale: float32 output tested on the Mock phase, scores went down, the container is back on uint8 (`v0.1.0`).
  3. Tile blending: a Hann window reduces tile seams (`compare_blending.py`).
  4. Training targets p2 to p99.8 without clipping: no gain on the official-style scorer.
  5. SSIM term in the loss: in progress, the first seed scores lower on the official-style scorer.
- Scores: `evaluate.py` is a local proxy and `official_metrics.py` an official-style reconstruction. Neither is an official challenge score. Decisions are taken on the official-style scorer, because the proxy was misleading in items 4 and 5.

## Pipeline

Run from the repository folder (Google Colab with Drive in our setup). Paths are examples.

```
# data (the download URL is on the challenge Database page, not in this repository)
python inspect_data.py manifest --info DIR_WITH_TSVS --out OUT_DIR
python download_data.py --base-url URL_TEMPLATE --root /content/lmc2 --out /content/drive/MyDrive/LMC2/task1 --manifest manifest.csv
python inspect_data.py scan --manifest manifest.csv --out OUT_DIR
python make_splits.py --manifest data_manifest_full.csv --out split_manifest.csv
python check_duplicates.py --out OUT/dup_check --exclude 3:Actin
python prepare_dataset.py build --out /content/prepared --shards-dir SHARDS

# training and scoring
python baseline_constant.py --out OUT/constant
python train.py --out OUT/base_s42 --seed 42 --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python eval_last.py --root OUT --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED
python score_official.py --checkpoint OUT/base_s42/best.pt --out OUT/base_s42/official --pairs-csv PAIRS_EXCLUDED
python make_figures.py --run OUT/base_s42

# single image
python infer.py --checkpoint best.pt --input IMAGE.tif --out PRED.ome.tiff
```

`train.py --config-from RUN/best.pt` copies every setting of an earlier run, so a controlled run changes only the options passed on the command line. `run_info.json` records the settings and the package and CUDA versions of each run.

## Files

| File | Purpose |
|---|---|
| `config.py` | Configuration dataclass and defaults |
| `download_data.py` | Download the data and keep only the files needed for Task 1 |
| `inspect_data.py` | Data manifest from the organizer TSVs, OME-TIFF metadata checks |
| `make_splits.py` | Frozen, study-grouped 80/20 split (seed 42) |
| `check_duplicates.py` | Find duplicated target channels, write the pairs file without them |
| `dataset.py` | Dataset: one transmitted-light plane to four fluorescence channels |
| `prepare_dataset.py` | Build and restore the prepared (normalized, float16) dataset |
| `unet.py` | U-Net model |
| `train.py` | Training, checkpoint selection, final evaluation |
| `predict.py` | Tiled full-image inference and local scoring |
| `evaluate.py` | Local proxy metrics (SSIM, PCC) |
| `official_metrics.py`, `score_official.py` | Official-style metrics and rescoring of a checkpoint |
| `baseline_constant.py` | Constant control (training-set mean per organelle) |
| `eval_last.py`, `aggregate_w3.py`, `w3_inspect.py` | Week 3 analysis: last epoch, aggregation over seeds, scores by modality and study |
| `compare_output_scale.py`, `compare_blending.py`, `proxy_ceiling.py` | Review items 2, 3 and 4 |
| `make_figures.py` | Training curves and example grids |
| `infer.py` | End-to-end inference on one image |
| `inference.py`, `ome_utils.py`, `Dockerfile`, `requirements-submission.txt` | Grand Challenge container |
| `submission_check.py`, `.github/workflows/submission.yml` | Container checks, run locally and in CI on synthetic inputs |
| `output_scale.json` | Per-organelle intensity range for the float32 output mode |
| `results/` | Data figures, split manifest, Week 2 figures, results record of the issue |

## Container

`docker build -t lmc2-task1 .` builds the Grand Challenge image. The model weights are uploaded to the Grand Challenge algorithm as a model file and are not part of the image. Tags: `v0.1.0` (uint8 output, active on Grand Challenge) and `v0.2.0` (float32 output mode).
