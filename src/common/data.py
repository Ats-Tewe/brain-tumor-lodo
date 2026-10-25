from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from common.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    VAL_SPLIT_RATIO,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def build_eval_transform(img_size: int = IMG_SIZE) -> Callable:
    """Deterministic evaluation transform. The ONLY definition in the project."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_train_transform(img_size: int = IMG_SIZE) -> Callable:
    """Training augmentation pipeline (paper Section IV-I / Table II)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])

class BrainTumorDataset(Dataset):
    """
    Class-directory image dataset using the project's authoritative class
    ordering (common.config.CLASS_NAMES), NOT ImageFolder's alphabetical order.

    Difference from the original module3_training implementation: an unreadable
    image is no longer silently replaced by a black frame carrying its original
    label. Corrupt files are recorded in `self.corrupt_files` and, by default,
    raise on access. Silent black-frame substitution meant a corrupted file was
    trained on and evaluated as a valid sample, quietly perturbing every metric
    with no trace in any log.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        class_names: Sequence[str] = CLASS_NAMES,
        on_corrupt: str = "raise",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = list(class_names)
        self.on_corrupt = on_corrupt
        self.corrupt_files: list[str] = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                continue
            idx = CLASS_TO_IDX[cls]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((cls_dir / fname, idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self.class_names}
        for _, idx in self.samples:
            counts[CLASS_NAMES[idx]] += 1
        return counts

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            self.corrupt_files.append(str(path))
            if self.on_corrupt == "raise":
                raise RuntimeError(
                    f"Unreadable image: {path}. Fix or remove the file, or pass "
                    f"on_corrupt='skip'. Do not train on silently substituted data."
                ) from exc
            if self.on_corrupt == "black":
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            else:
                return self.__getitem__((i + 1) % len(self.samples))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def present_classes(data_dir: str | Path, class_names: Sequence[str] = CLASS_NAMES) -> list[str]:
    """
    Return the classes that actually have at least one image under `data_dir`.

    Replaces three near-identical copies (explain.py, bootstrap_ci.py, and a
    variant in ablation_vit_standalone.py). Needed because LODO Round 2 tests on
    Figshare, which contains no no-tumor images.
    """
    data_dir = Path(data_dir)
    found = []
    for cls in class_names:
        cls_dir = data_dir / cls
        if cls_dir.is_dir() and any(
            f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(cls_dir)
        ):
            found.append(cls)
    return found

def imagefolder_label_remap() -> dict[int, int]:
    """
    Map torchvision ImageFolder's alphabetical class indices onto the project's
    authoritative CLASS_NAMES ordering.

    ImageFolder sorts alphabetically: glioma(0), meningioma(1), notumor(2),
    pituitary(3). The project uses glioma(0), meningioma(1), pituitary(2),
    notumor(3). Any script that uses ImageFolder must apply this remap, or its
    per-class metrics will silently swap pituitary and no-tumor.
    """
    alphabetical = sorted(CLASS_NAMES)
    return {i: CLASS_TO_IDX[name] for i, name in enumerate(alphabetical)}

def stratified_train_val_split(
    dataset: BrainTumorDataset,
    val_ratio: float = VAL_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[Subset, Subset]:
    """
    Per-class stratified train/val split.

    Uses a local `random.Random(seed)` instance rather than reseeding the global
    RNG. The original implementation called `random.seed(seed)` as a side effect
    inside the split function, which meant the global RNG state at the start of
    training depended on the order in which functions happened to be called.
    """
    rng = random.Random(seed)

    by_class: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, indices in by_class.items():
        if not indices:
            continue
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))

def seed_worker(worker_id: int) -> None:
    """
    DataLoader `worker_init_fn` that seeds python and numpy inside each worker.

    Without this, `num_workers>0` leaves `random` and `numpy` unseeded in worker
    processes, so any augmentation routed through them is not reproducible even
    when torch itself is seeded. Pass alongside an explicit `generator=`:

        g = torch.Generator(); g.manual_seed(RANDOM_SEED)
        DataLoader(..., worker_init_fn=seed_worker, generator=g)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int = RANDOM_SEED) -> torch.Generator:
    """Seeded generator for DataLoader shuffling and WeightedRandomSampler."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.nn.Module, dict]:
    """
    Load a trained DualHeadHybridCNNViT from a checkpoint.

    Replaces five copies of the same load-and-eval block. Returns the model in
    eval() mode plus the raw checkpoint dict (so callers can read img_size,
    epoch, metrics, and the run metadata stamp).

    The img_size is read FROM the checkpoint rather than defaulting to 224 --
    a default of 224 silently builds a model with the wrong number of position
    embeddings for a 320px run.
    """
    from module2_model.model import build_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    img_size = ckpt.get("img_size")
    if img_size is None:
        raise KeyError(
            f"{checkpoint_path} does not record 'img_size'. Refusing to guess: "
            f"building at the wrong resolution silently changes the ViT token "
            f"count and the position-embedding shape."
        )

    model = build_model(num_classes=num_classes, pretrained=False, img_size=img_size)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, ckpt
