# Light My Cells 2, Task 1 (Best Focus)

Code for Task 1 of the [Light My Cells 2](https://lightmycells2.grand-challenge.org/) challenge (France-BioImaging, Grand Challenge): predict the best-focus fluorescence images of four structures (Nucleus, Mitochondria, Tubulin, Actin) from one label-free transmitted-light image (bright-field, phase contrast or DIC). This is also called in-silico labeling. Task 2 of the challenge (maximum-intensity projection from a stack) is not covered here.

The work is reported in the task issue (itu-biai/dlct-docs #323). A copy of every result comment, with tables, commits and commands, is in [`results/issue323/`](results/issue323/README.md). Results from the follow-up task (itu-biai/dlct-docs #333) are reported there and copied to [`results/issue333/`](results/issue333/README.md).

## Current state

- Method: 2D U-Net (32 base channels), one transmitted-light plane in, four fluorescence channels out, observed-channel L1 loss, AdamW, 50 epochs, inference on 512 px tiles.
- Weeks 1 to 3: data inventory, frozen study-grouped split, baseline and two tested changes (cosine learning-rate decay, modality conditioning). Neither change beat the baseline over 3 seeds.
- Review items after Week 3:
  1. Duplicated targets: Study 3 Actin excluded (`check_duplicates.py`).
  2. Output intensity scale: float32 output tested on the Mock phase, scores went down, the container is back on uint8 (`v0.1.0`).
  3. Tile blending: a Hann window reduces tile seams (`compare_blending.py`).
  4. Training targets p2 to p99.8 without clipping: no gain on the official-style scorer.
  5. SSIM term in the loss: lower PCC and higher errors on the official-style scorer over seeds 42 and 123, not adopted.
- Follow-up task #333 (code in place, runs pending unless a result is linked):
  1. Focal plane of the inputs: `defocus_eval.py` scores the validation images with the transmitted-light slice at best focus + d. The container logs the OME-XML z fields and a sharpness value of every input.
  2. Pretrained encoder: `train.py --arch smp --encoder resnet34` (ImageNet weights, encoder learning rate x `--encoder-lr-mult`). Run B: `ssl_pretrain.py` (masked-image pretraining on the training-split transmitted-light slices), then `train.py --ssl-checkpoint`.
  3. Per-organelle choice of checkpoints and ensemble members: `score_official.py --groups` (`--split heldout` scores held-out training studies), deployed with `members.json` next to the weights.
  4. Loss aligned with the metric: `--pcc-weight`, `--ssim-norm`, `--loss-term-organelles`.
  5. Checkpoint selection on held-out training studies with the official-style scorer: `--select-on heldout`.
  6. Weight averaging: `--ema-decay`.
  7. Ensemble and test-time augmentation: three base seeds with 8-fold TTA, official-style macro SSIM 0.4453 and PCC 0.6116 ([record](results/issue333/01_item7_ensemble_tta.md)). Run time on the Grand Challenge hardware is measured with `time_inference.py` before it goes into a submission.
  8. Modality: `modality_report.py` gives official-style scores per modality, `--modality film` conditions the U-Net.
  9. Post-processing per organelle (TV, background offset, gamma): `postprocess.py` sweep, deployed with `postprocess.json` next to the weights.
  10. Patch size: `train.py --patch-size 128`.
  11. BM3D-denoised input as a second channel: `bm3d_input.py time` and `build`, then `train.py --extra-input bm3d`.
  12. Literature review: [`results/literature_review.md`](results/literature_review.md).
  - Fixed example set of #323 for any runs: `example_grid.py`.
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

# follow-up task #333
python train.py --out OUT/smp_r34_s42 --seed 42 --arch smp --encoder resnet34 --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python score_official.py --checkpoint OUT/base_s42/best.pt OUT/base_s123/best.pt OUT/base_s456/best.pt --window hann --overlap 128 --tta --groups base3=0,1,2 --out OUT/ens --pairs-csv PAIRS_EXCLUDED
python defocus_eval.py --checkpoint OUT/base_s42/best.pt --manifest data_manifest_full.csv --pairs-csv PAIRS_EXCLUDED --out OUT/defocus
python time_inference.py --checkpoint OUT/base_s42/best.pt OUT/base_s123/best.pt OUT/base_s456/best.pt --tta 1 4 8
python modality_report.py --score-dir OUT/ens/base3
python postprocess.py --checkpoint OUT/base_s42/best.pt OUT/base_s123/best.pt OUT/base_s456/best.pt --tta --out OUT/postprocess --pairs-csv PAIRS_EXCLUDED
python ssl_pretrain.py --manifest data_manifest_full.csv --out OUT/ssl_r34 --encoder resnet34 --steps 10000 --resume
python train.py --out OUT/ssl_r34_s42 --seed 42 --arch smp --encoder resnet34 --ssl-checkpoint OUT/ssl_r34/ssl_encoder.pt --select-on heldout --prepared-dir /content/prepared --pairs-csv PAIRS_EXCLUDED --resume
python bm3d_input.py time --size 2048
python example_grid.py --ref OUT/base_s42/per_image_scores.csv --run base=OUT/base_s42/best.pt smp=OUT/smp_r34_s42/best.pt --out OUT/grids

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
| `unet.py` | U-Net model and U-Net on a pretrained encoder (segmentation_models_pytorch) |
| `train.py` | Training (loss terms, EMA, encoder learning rate), checkpoint selection (validation subset or held-out training studies), final evaluation |
| `predict.py` | Tiled full-image inference with test-time augmentation, local scoring |
| `evaluate.py` | Local proxy metrics (SSIM, PCC) |
| `official_metrics.py`, `score_official.py` | Official-style metrics, rescoring of checkpoints, ensembles and member groups, with TTA |
| `defocus_eval.py` | Sensitivity of a model to the focal plane of the input |
| `postprocess.py` | Per-organelle post-processing (TV, background offset, gamma), its sweep and selection |
| `ssl_pretrain.py` | Masked-image pretraining of the encoder on transmitted-light slices of the training split |
| `bm3d_input.py` | BM3D-denoised input channel: timing and build for the prepared folder |
| `modality_report.py` | Official-style scores per transmitted-light modality |
| `example_grid.py` | Fixed example set (median per modality, lowest per organelle) for any runs |
| `time_inference.py` | Run time and memory of the container prediction for an ensemble and TTA setting |
| `baseline_constant.py` | Constant control (training-set mean per organelle) |
| `eval_last.py`, `aggregate_w3.py`, `w3_inspect.py` | Week 3 analysis: last epoch, aggregation over seeds, scores by modality and study |
| `compare_output_scale.py`, `compare_blending.py`, `proxy_ceiling.py` | Review items 2, 3 and 4 |
| `leakage_check.py` | Leakage check of the local split: split integrity, duplicates across splits, effect of the checkpoint-selection subset |
| `make_figures.py` | Training curves and example grids |
| `infer.py` | End-to-end inference on one image |
| `inference.py`, `ome_utils.py`, `Dockerfile`, `requirements-submission.txt` | Grand Challenge container |
| `submission_check.py`, `.github/workflows/submission.yml` | Container checks, run locally and in CI on synthetic inputs |
| `output_scale.json` | Per-organelle intensity range for the float32 output mode |
| `results/` | Data figures, split manifest, Week 2 figures, results record of the issue, literature review (`literature_review.md`) |

## Container

`docker build -t lmc2-task1 .` builds the Grand Challenge image. The model weights are uploaded to the Grand Challenge algorithm as a model file and are not part of the image. Tags: `v0.1.0` (uint8 output, active on Grand Challenge) and `v0.2.0` (float32 output mode).
