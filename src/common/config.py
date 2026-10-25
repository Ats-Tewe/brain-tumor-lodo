from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_V1_HARMONIZED = PROJECT_ROOT / "outputs_v1_224" / "harmonized_dataset"
_DEFAULT_HARMONIZED = PROJECT_ROOT / "outputs" / "harmonized_dataset"

HARMONIZED_DIR = Path(
    os.environ.get(
        "BTP_HARMONIZED_DIR",
        _V1_HARMONIZED if _V1_HARMONIZED.is_dir() else _DEFAULT_HARMONIZED,
    )
)

OUTPUTS_DIR = Path(os.environ.get("BTP_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
UNCERTAINTY_DIR = OUTPUTS_DIR / "uncertainty"
FIGURES_DIR = OUTPUTS_DIR / "figures"

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42
IMG_SIZE = 320
BATCH_SIZE = 24
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
VAL_SPLIT_RATIO = 0.10
MC_DROPOUT_PASSES = 50
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
ECE_N_BINS = 10
UNCERTAINTY_THRESHOLD = 0.10

PROTOCOLS = {
    "merged": {
        "description": "Merged Baseline (80/10/10 stratified split)",
        "train_dir": HARMONIZED_DIR / "merged_baseline" / "train",
        "val_dir": HARMONIZED_DIR / "merged_baseline" / "val",
        "test_dir": HARMONIZED_DIR / "merged_baseline" / "test",
        "held_out_dataset": None,
    },
    "lodo1": {
        "description": "LODO Round 1 (Train: Figshare+Mendeley, Test: Kaggle)",
        "train_dir": HARMONIZED_DIR / "lodo_round1" / "train",
        "val_dir": None,
        "test_dir": HARMONIZED_DIR / "lodo_round1" / "test",
        "held_out_dataset": "Kaggle",
    },
    "lodo2": {
        "description": "LODO Round 2 (Train: Kaggle+Mendeley, Test: Figshare)",
        "train_dir": HARMONIZED_DIR / "lodo_round2" / "train",
        "val_dir": None,
        "test_dir": HARMONIZED_DIR / "lodo_round2" / "test",
        "held_out_dataset": "Figshare",
        "absent_classes": ["notumor"],
    },
    "lodo3": {
        "description": "LODO Round 3 (Train: Kaggle+Figshare, Test: Mendeley)",
        "train_dir": HARMONIZED_DIR / "lodo_round3" / "train",
        "val_dir": None,
        "test_dir": HARMONIZED_DIR / "lodo_round3" / "test",
        "held_out_dataset": "Mendeley",
    },
}

ALL_PROTOCOLS = list(PROTOCOLS.keys())

DEDUP_TOTALS = {"Kaggle": 7200, "Figshare": 1068, "Mendeley": 4068}

def protocol_config(name: str) -> dict:
    """Look up one protocol config, with a helpful error on typos."""
    if name not in PROTOCOLS:
        raise KeyError(
            f"Unknown protocol {name!r}. Valid options: {ALL_PROTOCOLS}"
        )
    return PROTOCOLS[name]

def resolve_protocols(arg: str) -> list[str]:
    """Map a --protocol CLI value ('all' or one name) to a list of names."""
    return ALL_PROTOCOLS if arg == "all" else [arg]
