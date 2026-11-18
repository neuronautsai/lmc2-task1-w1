# Literature review: transmitted light to fluorescence prediction

Scope: methods for predicting fluorescence images of subcellular structures from label-free transmitted-light images, with the points raised in the issue #323 review (patch size, state-of-the-art architectures and methods, normalization and preprocessing, ensembles, a denoised copy of the input as an extra channel). Every statement below is taken from the linked source. Scores are the values reported by each source and are not comparable with the scores of this repository.

## 1. Light My Cells 2024 (first edition, ISBI 2024)

The challenge page lists the top three teams with one line each:

| Rank | Method (as listed on the results page) |
|---|---|
| 1 | Modified VQGAN |
| 2 | Bag of experts (UNet and UNETR with ViT encoder) |
| 3 | Modified RUNet |

Sources: [results page](https://lightmycells.grand-challenge.org/results/).

### 1.1 First place: simplified VQGAN

- One simplified VQGAN model per organelle (Nucleus, Mitochondria, Actin, Tubulin), each about 72.8M parameters, built on a fork of the Stable Diffusion codebase. The loss configuration lists a VQ-LPIPS loss with a discriminator.
- Latent diffusion performed worse than the per-organelle VQGAN models on this dataset.
- The repository states that the models had not converged at the submission deadline.
- Sources: [code repository](https://github.com/trangle1302/lmc_GenCellPaint), [ICML 2024 paper page](https://icml.cc/virtual/2024/35913).

### 1.2 Patch-based encoder-decoder (RUNet)

- RUNet encoder-decoder, one model per organelle. RUNet performed better than UNETR, SwinUNETR and Attention UNet in the authors' tests.
- Patch size: 512 x 512 patches performed better than 256 x 256 and 128 x 128. Resampling whole images to 1024 x 1024 performed worse than the patch-based approach.
- Normalization: inputs and training targets scaled to [0, 1].
- Loss: MSE + 0.2 (negative SSIM) + 0.1 (negative Pearson correlation) + 0.1 (cosine distance).
- Augmentation: random flips along both axes, plus random elastic transforms for Actin (small dataset).
- Sampling: the authors report that random sampling overfits to the largest studies and argue for study-based sampling.
- Inference: patches merged with Hann-window averaging (TorchIO grid sampling).
- Rank: 3rd on the public leaderboard at the time of writing.
- Source: [arXiv 2406.01187](https://arxiv.org/html/2406.01187v1).

### 1.3 pix2pix with an adaptive loss

- ResNet generator with 9 blocks, PatchGAN discriminator, organelle-specific decoders and discriminators, tanh output.
- Loss: pix2pix loss on the observed organelles only, 100 x masked L1 + 1 x cGAN. The mask uses the 2nd and 99.8th percentiles of the ground truth: pixels inside that range get weight 1, the others 0.1.
- Normalization: 16-bit images rescaled from [0, 65535] to [-1, 1].
- Training on 512 x 512 random crops, organelles sampled evenly within each batch.
- Inference: sliding window of 512 with 0.8 overlap, test-time augmentation with random 90-degree rotations.
- Final submission: ensemble of three separately trained pix2pix models, plus a UNet++ for Actin on DIC images, which had no Actin labels in training.
- Reported phase 2 scores, SSIM and PCC: Mitochondria 0.656 and 0.484, Nucleus 0.720 and 0.610, Tubulin 0.670 and 0.538, Actin 0.555 and 0.562.
- Source: [arXiv 2406.15716](https://arxiv.org/html/2406.15716v1).

### 1.4 Self-supervised Vision Transformers (BiaPy)

- Four UNETR-like models, one per organelle. A base model is pretrained with masked autoencoders (MAE) and each organelle model is fine-tuned from it with heavy data augmentation.
- A custom data loader uses more than one out-of-focus image per sample. The validation set is 10 percent of the training samples, one in every ten, so that every study is covered.
- Source: [BiaPy tutorial page](https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html).

## 2. A public Light My Cells 2 solution

- Five U-Nets (4 levels, base width 32, GroupNorm, SiLU, about 7.77M parameters each), trained in a five-fold scheme with specimen groups kept within one fold. Predictions of the five models are averaged.
- Input: the transmitted-light image plus three one-hot modality channels (BF, PC, DIC). The modality is read from the OME channel name and missing metadata raises an error.
- Preprocessing: images resampled to 0.2 µm per pixel, intensities normalized with the 2nd and 99.8th percentiles without clipping.
- Loss: L1 masked to the observed organelles.
- Reported local cross-validation on the first-edition data (not official scores), SSIM and PCC: Nucleus 0.564 and 0.538, Mitochondria 0.435 and 0.446, Tubulin 0.371 and 0.369, Actin 0.397 and 0.378.
- Source: [code repository](https://github.com/Agnuxo1/LightMyCells2-ModalityAware-UNet).

## 3. Earlier work on label-free prediction

- **Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy** (Nature Methods, 2018): U-Net based models, one per structure, evaluated with the Pearson correlation. Inputs and targets are z-scored per image. Loss MSE, Adam (learning rate 0.001). Training patches of 64 x 64 x 32 px for 3D and 64 x 64 px for the 2D DNA model. At inference the input is cropped minimally to a multiple of 16. The 3D model gave more detailed predictions than the 2D variant. Sources: [Nature Methods](https://www.nature.com/articles/s41592-018-0111-2), [bioRxiv full text](https://www.biorxiv.org/content/10.1101/289504v4.full-text).
- **In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images** (Cell, 2018): predicts several fluorescent labels across cell types and laboratories, and the learned features transfer to new problems with small training sets. Source: [publication page](https://research.google/pubs/in-silico-labeling-predicting-fluorescent-labels-in-unlabeled-images/).

## 4. Denoising and an extra input channel

- **Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)**: 5 convolutional and 5 deconvolutional layers with residual shortcuts, no pooling, trained with MSE on 55 x 55 overlapping patches of low-dose CT images. The network denoises the image itself. A denoised image is not used as an extra input channel. Source: [arXiv 1702.00288](https://arxiv.org/abs/1702.00288).
- **DenoiSeg: Joint Denoising and Segmentation** (ECCV 2020 workshops): extends the self-supervised denoiser Noise2Void to predict a segmentation in the same network, trained end to end on raw microscopy images with few annotations. Segmentation benefits from denoising, especially when both are learned jointly. On very low-noise data, the best results came after adding moderate synthetic noise. Source: [publication record](https://edoc.mdc-berlin.de/id/eprint/22411/).

## 5. Comparison with the current setup of this repository

| Setting | This repository | Sources above |
|---|---|---|
| Models | one U-Net with four outputs | one model per organelle (1.1, 1.2, 1.4, Section 3), organelle-specific decoders (1.3), one model with four outputs (2) |
| Architecture | U-Net, 32 base channels, 7.77M parameters | VQGAN (1.1), RUNet (1.2), pix2pix ResNet generator (1.3), UNETR with MAE pretraining (1.4), U-Net (2, Section 3) |
| Training patch | 256 x 256 | 512 x 512 (1.2, 1.3), 512 better than 256 and 128 (1.2), 64 x 64 (Section 3, 2D) |
| Normalization | per image, 1st to 99.5th percentile, clipped | [0, 1] scaling (1.2), [-1, 1] from 16 bit (1.3), 2nd to 99.8th percentile without clipping (2), per-image z-score (Section 3) |
| Pixel size | not resampled | resampled to 0.2 µm per pixel (2) |
| Loss | L1 on observed channels | MSE + SSIM + PCC + cosine terms (1.2), masked L1 + cGAN (1.3), masked L1 (2), MSE (Section 3) |
| Sampling | random over images | study-based (1.2), organelles balanced per batch (1.3) |
| Tile blending | linear ramp (Hann tested in review item 3) | Hann window (1.2), 0.8 overlap (1.3) |
| Ensemble and TTA | none in the container | 3 models (1.3), 5 fold models (2), bag of experts (rank 2), 90-degree rotations (1.3) |
| Extra input channels | none | modality one-hot (2) |
