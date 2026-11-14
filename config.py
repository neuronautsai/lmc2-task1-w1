"""Configuration for the LMC2 Task 1 pipeline."""
from dataclasses import dataclass
from pathlib import Path

ORGANELLES = ("Nucleus", "Mitochondria", "Tubulin", "Actin")
MODALITIES = ("BF", "DIC", "PC")
UNKNOWN_MODALITY = len(MODALITIES)  # index used when the modality label is withheld or not recognised


def modality_id(name):
    return MODALITIES.index(name) if name in MODALITIES else UNKNOWN_MODALITY


@dataclass(frozen=True)
class ChallengeConfig:
    data_root: Path = Path("/content/drive/MyDrive/LMC2/task1")
    pairs_csv: Path = Path("/content/drive/MyDrive/LMC2/task1/pairs.csv")
    output_dir: Path = Path("/content/drive/MyDrive/LMC2/outputs")
    prepared_dir: Path | None = None  # local folder with prepare_dataset.py output, None reads raw TIFFs
    patch_size: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 50
    seed: int = 42
    num_workers: int = 4
    crops_per_image: int = 4
    base_channels: int = 32
    tile_size: int = 512
    tile_overlap: int = 64
    val_subset: int = 100
    # Week 3 options, defaults reproduce the Week 2 run exactly
    lr_schedule: str = "const"  # "const" or "cosine" (per-epoch cosine decay to lr_min_factor times the base rate)
    lr_min_factor: float = 0.01
    modality_mode: str = "none"  # "none" or "film" (modality-aware conditioning, see unet.py)
    modality_dropout: float = 0.3  # chance per image that the modality label is hidden during training
    # Per-image percentile normalization, applied to inputs and targets
    p_low: float = 1.0
    p_high: float = 99.5
    # Training target rule (review of Week 3, item 3). The defaults keep the declared rule above. Only the training
    # targets change: inputs and the validation targets used for checkpoint selection and the local proxy keep the
    # declared rule, so the proxy numbers stay comparable with earlier runs.
    train_target_p_low: float | None = None  # None uses p_low
    train_target_p_high: float | None = None  # None uses p_high
    train_target_clip: bool = True
    train_prepared_dir: Path | None = None  # prepared folder built with the training target rule (None uses prepared_dir)
    output_activation: str = "sigmoid"  # "sigmoid" (output in (0, 1)) or "linear" (unbounded output)


def declared_rule(cfg):
    return (float(cfg.p_low), float(cfg.p_high), True)


def train_target_rule(cfg):
    lo = cfg.p_low if cfg.train_target_p_low is None else cfg.train_target_p_low
    hi = cfg.p_high if cfg.train_target_p_high is None else cfg.train_target_p_high
    return (float(lo), float(hi), bool(cfg.train_target_clip))
