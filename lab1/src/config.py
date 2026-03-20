from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "food-101"
IMAGES_DIR = DATASET_ROOT / "images"
META_DIR = DATASET_ROOT / "meta"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_RANDOM_SEED = 42
DEFAULT_VAL_SIZE = 0.2
DEFAULT_GRADCAM_CORRECT_EXAMPLES = 5
DEFAULT_GRADCAM_INCORRECT_EXAMPLES = 5


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = DEFAULT_IMAGE_SIZE
    baseline_epochs: int = 3
    improved_epochs: int = 5
    baseline_learning_rate: float = 1e-3
    improved_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    val_size: float = DEFAULT_VAL_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    debug_samples: int | None = None
    unfreeze_blocks: int = 3


@dataclass(slots=True)
class GradCAMConfig:
    batch_size: int = 8
    image_size: int = DEFAULT_IMAGE_SIZE
    correct_examples: int = DEFAULT_GRADCAM_CORRECT_EXAMPLES
    incorrect_examples: int = DEFAULT_GRADCAM_INCORRECT_EXAMPLES
    debug_samples: int | None = None


def ensure_directories() -> None:
    for directory in (CHECKPOINTS_DIR, ONNX_DIR, GRADCAM_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cpu: bool = False) -> torch.device:
    if prefer_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers() -> int:
    return 0
