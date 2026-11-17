## Reply to the Week 1 and 2 review

Thanks @mozanunal. Status of the five points. Points 2 and 5 are still open and are marked as such.

### 1. Team
The team was created on the challenge portal: https://lightmycells2.grand-challenge.org/teams/5834/
I entered the name "ITU BIAI Lab", the portal displays it as "Itu Biai Lab". The only member so far is my account (owner). The rules allow one team per research group and one team per participant, so other lab members should join this team instead of creating a new one.

### 2. Output intensity scale (open)
Not settled yet. What I checked:
- The Light My Cells 2 evaluation and rules pages do not specify normalization, SSIM data_range or output scale.
- In the first-edition forum an organizer stated that a percentile normalization is computed over each image before the metrics, and that a multiplication by the data type maximum was removed. That statement is about the first edition, I could not confirm it for this edition and it does not say whether predictions are normalized too.
- The evaluation code linked from the first-edition page could not be opened from my environment.

Next step: ask on the challenge forum (question drafted, not yet posted) and check the example container once it is available to me. No portal submission before this is answered.

### 3. Week 3 direction
Agreed that a method change may be needed. Planned experiments: learning rate decay first as a separate change, then one method change. My current candidate is modality-aware conditioning (option B in the issue), because of the BF-heavy training set against the DIC and PC heavy validation set. This is a proposal, no experiment has been run yet.

### 4. Data loader
Implemented the prepared dataset as suggested. Commit efdd78e in https://github.com/neuronautsai/lmc2-task1-w1 (prepare_dataset.py, dataset.py, train.py with --prepared-dir).
- One float16 .npy file per image, shape (5, H, W): channel 0 is the transmitted-light input at its best-z plane, channels 1 to 4 are Nucleus, Mitochondria, Tubulin, Actin at best-z (zeros where the target is missing). All channels are mapped to [0, 1] with the declared p1 to p99.5 rule (the same normalize function as before).
- The observed-channel mask is not stored inside each file. It is read from pairs.csv (the per-organelle observed-channel columns) at load time. prepared_index.csv is only written as an inventory and is not read by the loader. This deviates from the suggestion, I can move the mask into the files if preferred. (Corrected on 8 November, the first version of this sentence named the wrong file.)
- Uncompressed .npy so training reads memory-mapped random crops. Compressed .npz was not tested.
- 2,725 images (2,193 train and 532 validation, the images with at least one target).
- Prepared dataset size: 63.32 GB in total, mean 23.2 MB per image, max 89.3 MB. Stored on Drive as 8 tar shards.
- Build time 1,323 s (one time, from the raw files on Drive). Copy from Drive to local disk and extract at session start: 1,131 s (2,725 files).
- Seconds per epoch: about 300 s on the prepared data, measured on an NVIDIA L4 with 12 CPUs (Week 3 base run, seed 42, 14,934 s for 50 epochs). The earlier figure of 113 s against 315 s is withdrawn: the GPU and CPU count of that run were not recorded, the A100 assumption was not supported, and the number has not been reproduced. No same-GPU comparison of the raw and the prepared pipeline exists, so no speedup is claimed. The 315 s of Week 2 was measured on an A100-SXM4-40GB. (Corrected on 9 November.)
- A first timing run on a T4 runtime with 2 CPUs took 791 s per epoch. This is not comparable and is reported only so that the number is not mistaken for the A100 result.
- Effect of float16 storage, one run only: the Week 3 base run (seed 42, prepared data) reached full-validation macro SSIM 0.502 and PCC 0.608, against 0.504 and 0.605 for the Week 2 raw pipeline. A single seed cannot separate this from seed noise. (A synthetic test had shown a maximum difference of 2.4e-4 on [0, 1] images.)

Command for the timing run: `python train.py --out OUT --prepared-dir /content/prepared --epochs 1 --workers 12 --val-subset 8`

### 5. Artifacts (open)
Weights and the per-image score CSVs are not shared yet. I will link a shared folder here once it is set up.
