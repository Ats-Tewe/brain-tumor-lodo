
import os
import csv
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEDUP_ROOT         = os.path.join(PROJECT_ROOT, "outputs", "deduplicated")
KAGGLE_TRAIN_DIR   = os.path.join(DEDUP_ROOT, "kaggle",   "train")
KAGGLE_TEST_DIR    = os.path.join(DEDUP_ROOT, "kaggle",   "test")
FIGSHARE_DIR       = os.path.join(DEDUP_ROOT, "figshare", "all")
MENDELEY_TRAIN_DIR = os.path.join(DEDUP_ROOT, "mendeley", "train")
MENDELEY_TEST_DIR  = os.path.join(DEDUP_ROOT, "mendeley", "test")

HARMONIZED_ROOT  = os.path.join(PROJECT_ROOT, "outputs", "harmonized_dataset")
MERGED_ROOT      = os.path.join(HARMONIZED_ROOT, "merged_baseline")
MERGED_LOGS_DIR  = os.path.join(MERGED_ROOT, "logs")
MERGED_PLOTS_DIR = os.path.join(MERGED_ROOT, "plots")
REPORT_CSV       = os.path.join(MERGED_LOGS_DIR, "merged_baseline_report.csv")
SUMMARY_TXT      = os.path.join(MERGED_LOGS_DIR, "merged_baseline_summary.txt")

TARGET_SIZE  = (224, 224)
JPEG_QUALITY = 95

STANDARD_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

FIGSHARE_MENING_AUG_OPS = [
    'hflip',
    'vflip',
    'rot15',
    'rot345',
    'bright_up',
    'contrast',
]

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def collect_images(base_dir, cls):
    """Return sorted list of all image paths under base_dir/cls/."""
    folder = os.path.join(base_dir, cls)
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in VALID_EXT
    ])

def load_and_resize(path):
    """Open image, convert to RGB, resize to TARGET_SIZE. Returns PIL Image."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return img

def apply_augmentation(img, aug_type):
    """Apply a single augmentation operation to a PIL Image."""
    ops = {
        'hflip':     lambda i: ImageOps.mirror(i),
        'vflip':     lambda i: ImageOps.flip(i),
        'rot15':     lambda i: i.rotate(15,  resample=Image.BICUBIC, expand=False),
        'rot345':    lambda i: i.rotate(345, resample=Image.BICUBIC, expand=False),
        'bright_up': lambda i: ImageEnhance.Brightness(i).enhance(1.25),
        'contrast':  lambda i: ImageEnhance.Contrast(i).enhance(1.20),
    }
    return ops.get(aug_type, lambda i: i)(img)

def save_image(img, out_path):
    """Save a PIL Image as JPEG to out_path."""
    img.save(out_path, format='JPEG', quality=JPEG_QUALITY)

def save_split(image_paths, out_base_dir, cls, prefix):
    """
    Resize and save all image_paths to out_base_dir/cls/.
    Filename format: {prefix}_{cls}_{NNNNN}.jpg
    Returns number of images successfully saved.
    """
    out_dir = os.path.join(out_base_dir, cls)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for i, src_path in enumerate(image_paths, start=1):
        out_name = f"{prefix}_{cls}_{i:05d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        try:
            if "_aug_" in os.path.basename(src_path):
                shutil.copy2(src_path, out_path)
            else:
                img = load_and_resize(src_path)
                save_image(img, out_path)
            saved += 1
        except Exception as e:
            tqdm.write(f"  [SAVE ERROR] {os.path.basename(src_path)}: {e}")

    return saved

def collect_all_sources():
    """
    Gathers all deduplicated images from all three datasets into one pool
    per class, then applies a stratified 80/10/10 split.

    Figshare has no train/test split (single 'all' folder).
    Kaggle and Mendeley have separate train/test folders — both are pooled
    together for the merged baseline (the original splits are irrelevant here).

    Returns:
        pool: dict {class: [all image paths]}
    """
    print("\n[1/5] Collecting images from all three deduplicated datasets...")
    pool = {cls: [] for cls in STANDARD_CLASSES}

    for cls in STANDARD_CLASSES:
        kaggle_paths   = (collect_images(KAGGLE_TRAIN_DIR, cls) +
                          collect_images(KAGGLE_TEST_DIR,  cls))
        figshare_paths = collect_images(FIGSHARE_DIR, cls)
        mendeley_paths = (collect_images(MENDELEY_TRAIN_DIR, cls) +
                          collect_images(MENDELEY_TEST_DIR,  cls))

        combined = kaggle_paths + figshare_paths + mendeley_paths
        pool[cls] = combined

        print(f"  [{cls:<12}]  Kaggle={len(kaggle_paths):>4}  "
              f"Figshare={len(figshare_paths):>4}  "
              f"Mendeley={len(mendeley_paths):>4}  "
              f"Total={len(combined):>5}")

    total_all = sum(len(v) for v in pool.values())
    print(f"\n  Grand total (all classes): {total_all:,} images")
    return pool

def stratified_split(pool):
    """
    Applies a reproducible stratified 80/10/10 split per class.

    The split is deterministic (seed=42). Within each class, the full list
    is shuffled once, then cut at the 80% and 90% boundaries.

    IMPORTANT — augmentation rule:
        Figshare meningioma images that fall into the TRAIN split will
        receive offline augmentation in Stage 3.
        Images that fall into VAL or TEST are NEVER augmented.
        This is enforced by tracking original paths before augmentation.

    Returns:
        splits: dict {class: {'train': [...], 'val': [...], 'test': [...]}}
    """
    random.seed(RANDOM_SEED)
    splits = {}

    print("\n[2/5] Applying stratified 80/10/10 split per class...")
    for cls in STANDARD_CLASSES:
        paths = list(pool[cls])
        random.shuffle(paths)

        n        = len(paths)
        n_train  = int(n * TRAIN_RATIO)
        n_val    = int(n * VAL_RATIO)
        n_test   = n - n_train - n_val

        splits[cls] = {
            'train': paths[:n_train],
            'val':   paths[n_train : n_train + n_val],
            'test':  paths[n_train + n_val:],
        }

        print(f"  [{cls:<12}]  total={n:>5}  "
              f"train={len(splits[cls]['train']):>4}  "
              f"val={len(splits[cls]['val']):>4}  "
              f"test={len(splits[cls]['test']):>4}")

    return splits

def augment_figshare_meningioma_in_split(train_paths, aug_temp_dir):
    """
    Identifies Figshare meningioma images in the training split and
    generates 6 augmented copies of each.

    Only images from the FIGSHARE_DIR are augmented — Kaggle and Mendeley
    meningioma images in the training split are NOT augmented offline.
    (All classes receive online augmentation during training via DataLoader.)

    Figshare meningioma gets special offline augmentation for the same
    reason as in Step 4: its post-deduplication count (58) is too small
    for even a weighted sampler to provide sufficient diversity.

    Returns updated train_paths list (originals + aug copies).
    """
    figshare_prefix = os.path.normpath(FIGSHARE_DIR)
    fs_mening_in_train = [
        p for p in train_paths
        if os.path.normpath(p).startswith(figshare_prefix)
    ]

    if not fs_mening_in_train:
        print("  [INFO] No Figshare meningioma images in training split — skipping aug.")
        return train_paths

    print(f"\n  Augmenting {len(fs_mening_in_train)} Figshare meningioma "
          f"training images × {len(FIGSHARE_MENING_AUG_OPS)} ops = "
          f"{len(fs_mening_in_train) * len(FIGSHARE_MENING_AUG_OPS)} copies...")

    os.makedirs(aug_temp_dir, exist_ok=True)
    aug_paths = []

    for src_path in tqdm(fs_mening_in_train, desc="    Augmenting", unit="img"):
        base = os.path.splitext(os.path.basename(src_path))[0]
        try:
            img = load_and_resize(src_path)
        except Exception as e:
            tqdm.write(f"    [SKIP] {os.path.basename(src_path)}: {e}")
            continue

        for aug_type in FIGSHARE_MENING_AUG_OPS:
            try:
                aug_img  = apply_augmentation(img, aug_type)
                out_name = f"{base}_aug_{aug_type}.jpg"
                out_path = os.path.join(aug_temp_dir, out_name)
                save_image(aug_img, out_path)
                aug_paths.append(out_path)
            except Exception as e:
                tqdm.write(f"    [AUG ERROR] {aug_type} on {base}: {e}")

    print(f"  Augmented copies created: {len(aug_paths)}")
    return train_paths + aug_paths

def write_merged_splits(splits):
    """
    Writes train/val/test directories to outputs/harmonized_dataset/merged_baseline/.
    Applies Figshare meningioma augmentation to the training split.

    Returns counts: {'train': {cls: n}, 'val': {cls: n}, 'test': {cls: n}}
    """
    aug_temp = os.path.join(MERGED_ROOT, "_aug_temp", "figshare_meningioma")
    counts   = {s: {} for s in ['train', 'val', 'test']}

    print("\n[3/5] Writing training split (with Figshare meningioma augmentation)...")
    for cls in tqdm(STANDARD_CLASSES, desc="  train", unit="class"):
        train_paths = splits[cls]['train']

        if cls == "meningioma":
            train_paths = augment_figshare_meningioma_in_split(
                train_paths,
                os.path.join(aug_temp, cls)
            )

        n = save_split(
            train_paths,
            os.path.join(MERGED_ROOT, "train"),
            cls,
            prefix="mb_train"
        )
        counts['train'][cls] = n

    print("\n[4/5] Writing validation split...")
    for cls in tqdm(STANDARD_CLASSES, desc="  val  ", unit="class"):
        n = save_split(
            splits[cls]['val'],
            os.path.join(MERGED_ROOT, "val"),
            cls,
            prefix="mb_val"
        )
        counts['val'][cls] = n

    print("\n       Writing test split...")
    for cls in tqdm(STANDARD_CLASSES, desc="  test ", unit="class"):
        n = save_split(
            splits[cls]['test'],
            os.path.join(MERGED_ROOT, "test"),
            cls,
            prefix="mb_test"
        )
        counts['test'][cls] = n

    temp_root = os.path.join(MERGED_ROOT, "_aug_temp")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    print("\n  Temp augmentation folder cleaned up.")

    return counts

def save_csv_report(counts):
    os.makedirs(MERGED_LOGS_DIR, exist_ok=True)
    with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Split", "Class", "Count"])
        for split in ['train', 'val', 'test']:
            for cls in STANDARD_CLASSES:
                w.writerow([split, cls, counts[split].get(cls, 0)])
    print(f"  [SAVED] CSV     -> {REPORT_CSV}")

def save_summary(counts, splits):
    os.makedirs(MERGED_LOGS_DIR, exist_ok=True)

    n_fs_orig = len(collect_images(FIGSHARE_DIR, "meningioma"))
    figshare_prefix = os.path.normpath(FIGSHARE_DIR)
    fs_in_train = len([
        p for p in splits["meningioma"]["train"]
        if os.path.normpath(p).startswith(figshare_prefix)
    ])
    n_aug = fs_in_train * len(FIGSHARE_MENING_AUG_OPS)

    lines = [
        "=" * 68,
        "  MODULE 1 — STEP 5: Merged Baseline Dataset",
        "=" * 68,
        "",
        "  CONFIGURATION",
        f"  Split ratios        : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} train/val/test",
        f"  Image size          : {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px",
        f"  JPEG quality        : {JPEG_QUALITY}",
        f"  Random seed         : {RANDOM_SEED}",
        "",
        "  SOURCE DATASETS (all pooled before splitting)",
        "  - Kaggle Brain Tumor MRI Dataset (Nickparvar) [deduplicated]",
        "  - Figshare Jun Cheng Dataset [deduplicated]",
        "  - Mendeley Brain Tumor MRI Dataset (Hamada) [deduplicated]",
        "",
        "  AUGMENTATION (Figshare meningioma in train split only)",
        f"  Figshare meningioma in train : {fs_in_train} original images",
        f"  Augmented copies created     : {n_aug}",
        f"  Operations applied           : {', '.join(FIGSHARE_MENING_AUG_OPS)}",
        "  Val/Test splits              : NO augmentation (originals only)",
        "",
        "  SPLIT COUNTS",
        f"  {'Class':<14} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}",
        "  " + "-" * 48,
    ]

    for cls in STANDARD_CLASSES:
        tr  = counts['train'].get(cls, 0)
        va  = counts['val'].get(cls, 0)
        te  = counts['test'].get(cls, 0)
        tot = tr + va + te
        lines.append(f"  {cls:<14} {tr:>8} {va:>8} {te:>8} {tot:>8}")

    tr_tot  = sum(counts['train'].values())
    va_tot  = sum(counts['val'].values())
    te_tot  = sum(counts['test'].values())
    all_tot = tr_tot + va_tot + te_tot
    lines += [
        "  " + "-" * 48,
        f"  {'TOTAL':<14} {tr_tot:>8} {va_tot:>8} {te_tot:>8} {all_tot:>8}",
        "",
        "  IMBALANCE ANALYSIS (training split)",
    ]

    train_counts = [counts['train'].get(c, 0) for c in STANDARD_CLASSES]
    non_zero = [c for c in train_counts if c > 0]
    if non_zero:
        ratio = max(non_zero) / min(non_zero)
        max_cls = STANDARD_CLASSES[train_counts.index(max(non_zero))]
        min_cls = STANDARD_CLASSES[train_counts.index(min(non_zero))]
        lines += [
            f"  Imbalance ratio: {ratio:.2f}x  ({max_cls} vs {min_cls})",
            "  Handled in Module 3 via:",
            "    - WeightedRandomSampler (balanced sampling per epoch)",
            "    - Focal Loss gamma=2 with inverse-frequency alpha weights",
            "    - Online augmentation in DataLoader",
            "",
            "  USAGE IN THESIS",
            "  The merged baseline provides the standard evaluation protocol",
            "  used in most published brain tumor classification papers.",
            "  Comparing merged vs LODO results demonstrates whether the",
            "  model generalises beyond its training data distribution.",
        ]

    lines += [
        "",
        "  Output: outputs/harmonized_dataset/merged_baseline/",
        "    train/  val/  test/  logs/  plots/",
        "",
        "  Next: Module 3 — Training Loop",
        "=" * 68,
    ]

    report = '\n'.join(lines)
    print('\n' + report)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  [SAVED] Summary -> {SUMMARY_TXT}")

def plot_merged_counts(counts):
    """Bar chart comparing train/val/test counts per class."""
    os.makedirs(MERGED_PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x      = np.arange(len(STANDARD_CLASSES))
    width  = 0.25
    labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    colors = {'train': '#2196F3', 'val': '#4CAF50', 'test': '#FF9800'}

    for i, split in enumerate(['train', 'val', 'test']):
        vals = [counts[split].get(cls, 0) for cls in STANDARD_CLASSES]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=split.capitalize(),
                      color=colors[split], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 15, str(val),
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold', rotation=45)

    ax.set_title(
        "Merged Baseline Dataset — 80/10/10 Stratified Split\n"
        "All three deduplicated datasets combined",
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = os.path.join(MERGED_PLOTS_DIR, "merged_baseline_counts.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Chart   -> {out}")

def plot_sample_grid(n_per_class=4):
    """Sample image grid from merged training split."""
    save_path = os.path.join(MERGED_PLOTS_DIR, "merged_baseline_sample_grid.png")
    cls_labels = {"glioma": "Glioma", "meningioma": "Meningioma",
                  "pituitary": "Pituitary", "notumor": "No Tumor"}

    fig, axes = plt.subplots(len(STANDARD_CLASSES), n_per_class,
                             figsize=(n_per_class * 2.5, len(STANDARD_CLASSES) * 2.5))
    fig.suptitle("Sample Training Images — Merged Baseline",
                 fontsize=12, fontweight='bold')

    for row, cls in enumerate(STANDARD_CLASSES):
        cls_dir   = os.path.join(MERGED_ROOT, "train", cls)
        all_files = []
        if os.path.isdir(cls_dir):
            all_files = [f for f in os.listdir(cls_dir)
                         if os.path.splitext(f.lower())[1] in VALID_EXT]
        random.seed(RANDOM_SEED + row)
        sample_files = random.sample(all_files, min(n_per_class, len(all_files)))

        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(sample_files):
                try:
                    with Image.open(os.path.join(cls_dir, sample_files[col])) as img:
                        ax.imshow(img)
                    if col == 0:
                        ax.set_ylabel(cls_labels[cls], fontsize=10, fontweight='bold')
                except Exception:
                    ax.axis('off')
                    continue
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Grid    -> {save_path}")

def main():
    print("=" * 68)
    print("  MODULE 1 — STEP 5: Merged Baseline Dataset Creation")
    print("=" * 68)

    print("\n[CHECK] Verifying deduplicated input directories...")
    all_ok = True
    for path, label in [
        (KAGGLE_TRAIN_DIR,   "Kaggle   train"),
        (KAGGLE_TEST_DIR,    "Kaggle   test"),
        (FIGSHARE_DIR,       "Figshare all"),
        (MENDELEY_TRAIN_DIR, "Mendeley train"),
        (MENDELEY_TEST_DIR,  "Mendeley test"),
    ]:
        ok = os.path.isdir(path)
        print(f"  {'[OK]   ' if ok else '[ERROR]'} {label} -> {path}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n  ERROR: One or more source directories not found.")
        print("  Please ensure Step 3 (step3_deduplicate.py) has been run.")
        return

    print("\n[0/5] Removing previous merged_baseline output (if any)...")
    if os.path.exists(MERGED_ROOT):
        shutil.rmtree(MERGED_ROOT)
        print("  Removed previous output.")
    os.makedirs(MERGED_ROOT,      exist_ok=True)
    os.makedirs(MERGED_LOGS_DIR,  exist_ok=True)
    os.makedirs(MERGED_PLOTS_DIR, exist_ok=True)

    pool = collect_all_sources()

    splits = stratified_split(pool)

    counts = write_merged_splits(splits)

    print("\n[5/5] Saving reports and charts...")
    save_csv_report(counts)
    plot_merged_counts(counts)
    plot_sample_grid()
    save_summary(counts, splits)

    print("\n" + "=" * 68)
    print("  STEP 5 COMPLETE — Merged Baseline Ready")
    print("=" * 68)
    print(f"\n  Output: {MERGED_ROOT}\n")

    for split in ['train', 'val', 'test']:
        total = sum(counts[split].values())
        cls_str = "  ".join(
            f"{cls[:4]}={counts[split].get(cls, 0)}"
            for cls in STANDARD_CLASSES
        )
        print(f"    {split:<6}  {total:>5} images  |  {cls_str}")

    print()
    tr_tot = sum(counts['train'].values())
    va_tot = sum(counts['val'].values())
    te_tot = sum(counts['test'].values())
    print(f"  Actual split ratios:")
    grand  = tr_tot + va_tot + te_tot
    print(f"    Train : {tr_tot:>5} ({tr_tot/grand*100:.1f}%)")
    print(f"    Val   : {va_tot:>5} ({va_tot/grand*100:.1f}%)")
    print(f"    Test  : {te_tot:>5} ({te_tot/grand*100:.1f}%)")
    print(f"    Total : {grand:>5}")

    print(f"\n  Imbalance handling: WeightedRandomSampler + Focal Loss")
    print(f"  (same as LODO folds — implemented in Module 3)")
    print()
    print(f"  Module 1 is now fully complete.")
    print(f"  All 5 steps done:")
    print(f"    Step 1: Figshare .mat -> JPEG conversion         [DONE]")
    print(f"    Step 2: Dataset inspection & class count report  [DONE]")
    print(f"    Step 3: Cross-dataset deduplication              [DONE]")
    print(f"    Step 4: Harmonization & LODO split creation      [DONE]")
    print(f"    Step 5: Merged baseline dataset creation         [DONE]")
    print()
    print(f"  Ready for Module 3: Training Loop.")
    print("=" * 68)
