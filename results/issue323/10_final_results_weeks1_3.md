## Final results: Light My Cells 2, Task 1 (Best Focus), Weeks 1 to 3

This comment completes the Week 3 tables of the ticket and collects the final state of the task. Details are in the earlier comments: Week 1 results, Week 2 results and follow-up, the official-style rescoring, and the three Week 3 comments (results, inspection, validation submission). The presentation follows separately.

### Method comparison (local proxy, not official scores)
Full local validation split (532 usable images), `best.pt` checkpoint rule, mean ± std over seeds 42, 123 and 456.

| Method | Seeds completed | Local macro SSIM ↑ | Local macro PCC ↑ | Inference seconds/image | Peak GPU memory |
|---|---|---|---|---|---|
| Constant control (training-set mean per organelle) | N/A | 0.310 | undefined (constant prediction) | N/A (no network) | N/A |
| U-Net + L1 (`base`) | 3 | 0.4947 ± 0.0086 | 0.5946 ± 0.0178 | 0.336 ± 0.003 | 3.70 GB |
| Tested: cosine LR decay (`cos`) | 3 | 0.4861 ± 0.0033 | 0.5804 ± 0.0053 | 0.341 ± 0.007 | 3.70 GB |
| Tested: cosine LR + FiLM modality conditioning (`cosfilm`) | 3 | 0.4873 ± 0.0021 | 0.5673 ± 0.0256 | 0.366 ± 0.000 | 4.13 GB |

- **Selected improvement: none.** Neither tested change beat the baseline, with `best.pt` or with the last epoch (`last.pt`: base 0.4837 / 0.5983, cos 0.4724 / 0.5973, cosfilm 0.4772 / 0.5806). The frozen method is the U-Net baseline.
- The constant control is the Week 2 result on the same split and proxy.
- Inference time and peak memory were measured on an NVIDIA L4. Peak memory is the maximum over seeds, because runs resumed after a Colab disconnect under-report it.

### Validation submission

| Submission ID | Task | Commit/model | Portal scores and metric names | Runtime | Remaining quota |
|---|---|---|---|---|---|
| `67c619bd-bee7-44c2-af3b-85c3476ce62b` | Task 1, Best Focus, Mock phase | tag `v0.1.0` (commit `c53165b`), `base` seed 42 `best.pt` | SSIM: Nucleus 0.4062, Mitochondria 0.3468, Actin 0.4513, Tubulin 0.1988. PCC: Nucleus 0.4353, Mitochondria 0.2448, Actin 0.6192, Tubulin 0.1454. MAE, Euclidean and cosine distance are in the submission comment. | 2 cases, both algorithm jobs succeeded, about 7 min from submission to result | 4 of 5 in the 70-day window (phase closes 1 December 2026) |

The Mock phase scores 2 cases, so the portal numbers are not comparable with the local means and were not used to choose the method.

**Key result:** Over three seeds, the Week 2 U-Net (constant learning rate, no conditioning) remains the best configuration, with macro SSIM 0.4947 ± 0.0086 and PCC 0.5946 ± 0.0178 against SSIM 0.310 for the constant control. Cosine LR decay and FiLM modality conditioning did not improve on it under either checkpoint rule. The baseline runs end to end on Grand Challenge (Best Focus Mock phase, 2 of 2 cases succeeded).

**Takeaway:** The differences between configurations are of the same size as the seed variation, so further tuning of this setup is unlikely to help much. The clearest problems found are in the data and the output, not in the training schedule: the Study 3 Actin target is a copy of the Nucleus target, near-empty images show background bands that may come from the tile blending, and the portal MAE and Euclidean distance seem to depend on the output intensity scale. These should be fixed before a method change is tested.

### Next steps
1. Check the Study 3 images in the training split and exclude Study 3 Actin from training and scoring until the files are corrected.
2. Write predictions in the target intensity range instead of 0 to 255 and test whether the portal MAE and Euclidean distance change (open question, inferred from the scale of the numbers).
3. Check the tile blending of `predict_full` on near-empty images.
4. Then test a method change (for example an SSIM term in the loss or balanced sampling across modalities), as suggested in the Week 1 and 2 review.

### Reproducibility (Week 3)
- Code: https://github.com/neuronautsai/lmc2-task1-w1. Training options `a4bc300`, analysis `d866508` (`aggregate_w3.py`, `eval_last.py`) and `5441b54` (`w3_inspect.py`), container tag `v0.1.0` (`c53165b`).
- Data and split: same download and frozen split as Weeks 1 and 2 (`results/split_manifest.csv`, 532 usable validation images). Training reads a prepared copy built once with `prepare_dataset.py` (best-z planes, p1 to p99.5 per image and channel, clipped to [0, 1], float16).
- Model: U-Net with 32 base channels (7,765,508 parameters, 7,862,788 with FiLM), 256 x 256 patches, batch 4, 4 crops per image, observed-channel L1 loss, AdamW 1e-4, bfloat16 autocast, 50 epochs. `cos`: per-epoch cosine decay to 1 % of the base rate. `cosfilm`: FiLM on the modality label (BF, DIC, PC, unknown), label hidden with probability 0.3 during training. Inference on 512 x 512 tiles with 64 px overlap.
- Checkpoint rules: `best.pt` is the maximum macro SSIM on a fixed 100-image subset of the validation split, `last.pt` is epoch 50.
- Metric: local proxy, SSIM (data_range 1) and PCC per image and observed channel on p1 to p99.5 normalized images, then mean per organelle and macro mean over the four organelles. Official-style scoring is in `official_metrics.py` and `score_official.py`.
- Seeds: 42, 123, 456. Environment: Google Colab, NVIDIA L4, 12 CPUs. Package versions were not recorded in the Week 3 run files, which is a gap. The Week 2 versions are in the Week 2 comment.
- Command pattern (from the repo folder, other options at their `config.py` defaults):
  ```
  python prepare_dataset.py restore --shards-dir SHARDS_DIR --out /content/prepared
  python train.py --out OUT/w3/base_s42 --seed 42 --prepared-dir /content/prepared --resume
  python train.py --out OUT/w3/cos_s42 --seed 42 --prepared-dir /content/prepared --lr-schedule cosine --resume
  python train.py --out OUT/w3/cosfilm_s42 --seed 42 --prepared-dir /content/prepared --lr-schedule cosine --modality film --resume
  python eval_last.py --root OUT/w3 --prepared-dir /content/prepared
  python aggregate_w3.py --root OUT/w3
  python w3_inspect.py --root OUT/w3 --out OUT/w3/analysis --seed 42
  ```
  The same for seeds 123 and 456.
- Container check: `docker build -t lmc2-task1 .`, then `submission_check.py make-inputs`, the container run with `--network none --memory 16g`, and `submission_check.py check-outputs` (the same steps run in GitHub Actions). `check_real_inputs.py` runs `inference.py` on real validation images.

### Deliverables
- Code: `download_data.py`, `inspect_data.py`, `make_splits.py`, `dataset.py`, `train.py`, `evaluate.py`, `predict.py`, `config.py` (dataclass configuration), plus `baseline_constant.py`, `unet.py`, `prepare_dataset.py` and the Week 3 scripts above.
- Artifacts: Week 3 weights, per-image CSVs, last-epoch scores, tables and figures in (Drive link in issue #323) Week 1 and 2 artifacts are linked in their comments.
- Submission: `Dockerfile`, `inference.py`, `requirements-submission.txt`, CI workflow, model weights uploaded to the Grand Challenge algorithm as a model file (not in the image), submission record above.
- Organizer baseline: I did not find a trained baseline from the organizers for this edition. Their container template shows the input and output handling only, and it was used for `inference.py`.
- Presentation: to follow.
