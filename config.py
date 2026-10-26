"""Configuration for the LMC2 Task 1 pipeline."""
from dataclasses import dataclass
from pathlib import Path

ORGANELLES = ("Nucleus", "Mitochondria", "Tubulin", "Actin")


@dataclass(frozen=True)
class ChallengeConfig:
    data_root: Path = Path("/content/drive/MyDrive/LMC2/task1")
    pairs_csv: Path = Path("/content/drive/MyDrive/LMC2/task1/pairs.csv")
    output_dir: Path = Path("/content/drive/MyDrive/LMC2/outputs")
    patch_size: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 50
    seed: int = 42
    num_workers: int = 4
    # Per-image percentile normalization, applied to inputs and targets
    p_low: float = 1.0
    p_high: float = 99.5
