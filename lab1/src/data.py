from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    DATASET_ROOT,
    DEFAULT_MEAN,
    DEFAULT_STD,
    IMAGES_DIR,
    META_DIR,
    TrainingConfig,
)


@dataclass(slots=True)
class FoodSample:
    image_path: Path
    label_index: int
    label_name: str


class Food101Dataset(Dataset):
    def __init__(self, samples: list[FoodSample], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label_index,
            "label_name": sample.label_name,
            "path": str(sample.image_path),
        }


def _read_class_names() -> list[str]:
    classes_path = META_DIR / "classes.txt"
    return [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]


def _read_split_file(split_name: str) -> list[str]:
    split_path = META_DIR / f"{split_name}.txt"
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def _build_samples(image_ids: list[str], class_names: list[str]) -> list[FoodSample]:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    samples: list[FoodSample] = []
    for image_id in image_ids:
        label_name = image_id.split("/")[0]
        samples.append(
            FoodSample(
                image_path=IMAGES_DIR / f"{image_id}.jpg",
                label_index=label_to_index[label_name],
                label_name=label_name,
            )
        )
    return samples


def load_food101_splits(
    val_size: float,
    random_seed: int,
    debug_samples: int | None = None,
) -> tuple[list[FoodSample], list[FoodSample], list[FoodSample], list[str]]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Food-101 directory was not found: {DATASET_ROOT}")

    class_names = _read_class_names()
    train_samples = _build_samples(_read_split_file("train"), class_names)
    test_samples = _build_samples(_read_split_file("test"), class_names)

    train_indices = list(range(len(train_samples)))
    train_labels = [sample.label_index for sample in train_samples]
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_labels,
    )

    train_split = [train_samples[index] for index in train_idx]
    val_split = [train_samples[index] for index in val_idx]

    if debug_samples is not None:
        train_split = train_split[:debug_samples]
        val_split = val_split[: max(1, debug_samples // 4)]
        test_samples = test_samples[:debug_samples]

    return train_split, val_split, test_samples, class_names


def build_transforms(image_size: int, use_augmentations: bool) -> tuple[transforms.Compose, transforms.Compose]:
    if use_augmentations:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    config: TrainingConfig,
    use_augmentations: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_samples, val_samples, test_samples, class_names = load_food101_splits(
        val_size=config.val_size,
        random_seed=config.random_seed,
        debug_samples=config.debug_samples,
    )
    train_transform, eval_transform = build_transforms(config.image_size, use_augmentations)

    train_dataset = Food101Dataset(train_samples, transform=train_transform)
    val_dataset = Food101Dataset(val_samples, transform=eval_transform)
    test_dataset = Food101Dataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader, class_names


def create_eval_loader(
    batch_size: int,
    image_size: int,
    num_workers: int,
    debug_samples: int | None = None,
) -> tuple[DataLoader, list[str]]:
    config = TrainingConfig(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        debug_samples=debug_samples,
    )
    _, _, test_loader, class_names = create_dataloaders(config=config, use_augmentations=False)
    return test_loader, class_names

