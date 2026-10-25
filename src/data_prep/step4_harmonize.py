
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

HARMONIZED_ROOT = os.path.join(PROJECT_ROOT, "outputs", "harmonized_dataset")
LOGS_DIR        = os.path.join(HARMONIZED_ROOT, "logs")
PLOTS_DIR       = os.path.join(HARMONIZED_ROOT, "plots")
REPORT_CSV      = os.path.join(LOGS_DIR, "harmonization_report.csv")
SUMMARY_TXT     = os.path.join(LOGS_DIR, "harmonization_summary.txt")

TARGET_SIZE  = (224, 224)
JPEG_QUALITY = 95

STANDARD_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

MENDELEY_TEST_RATIO = 0.20
RANDOM_SEED         = 42

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

def resplit_mendeley():
    """
    Pool all Mendeley images from both original splits and perform a fresh
    stratified 80/20 split per class (seed=42).

    Returns:
        mendeley_train : dict {class: [paths]}
        mendeley_test  : dict {class: [paths]}
    """
    random.seed(RANDOM_SEED)
    mendeley_train = {}
    mendeley_test  = {}

    print("  Performing stratified 80/20 resplit of Mendeley dataset...")
    for cls in STANDARD_CLASSES:
        all_paths = (collect_images(MENDELEY_TRAIN_DIR, cls) +
                     collect_images(MENDELEY_TEST_DIR,  cls))
        random.shuffle(all_paths)
        split_idx             = int(len(all_paths) * (1 - MENDELEY_TEST_RATIO))
        mendeley_train[cls]   = all_paths[:split_idx]
        mendeley_test[cls]    = all_paths[split_idx:]
        print(f"    [{cls:<12}]  total={len(all_paths):>4}  "
              f"train={len(mendeley_train[cls]):>4}  "
              f"test={len(mendeley_test[cls]):>4}")

    return mendeley_train, mendeley_test

def augment_figshare_meningioma(figshare_paths, aug_temp_dir):
    """
    Apply FIGSHARE_MENING_AUG_OPS to all 58 Figshare meningioma images.
    Saves augmented copies to aug_temp_dir.

    Returns list of augmented image paths (originals NOT included — they are
    added separately when assembling the fold).
    """
    os.makedirs(aug_temp_dir, exist_ok=True)
    aug_paths = []

    print(f"  Augmenting Figshare meningioma: "
          f"{len(figshare_paths)} originals × "
          f"{len(FIGSHARE_MENING_AUG_OPS)} ops = "
          f"{len(figshare_paths) * len(FIGSHARE_MENING_AUG_OPS)} augmented copies")

    for src_path in tqdm(figshare_paths, desc="    Augmenting", unit="img"):
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

    print(f"  Augmented copies saved: {len(aug_paths)}")
    return aug_paths

def save_split(image_paths, out_base_dir, cls, prefix, desc=""):
    """
    Resize and save all image_paths to out_base_dir/cls/.
    Filenames: {prefix}_{cls}_{NNNNN}.jpg

    Augmented images (already 224x224) are copied directly.
    Original images are resized before saving.

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

def assemble_lodo_folds(mendeley_train, mendeley_test):
    """
    Build Round 1, 2, and 3 fold directories.
    Returns counts dict: {round: {train/test: {class: n}}}
    """
    counts      = {}
    aug_temp    = os.path.join(HARMONIZED_ROOT, "_aug_temp", "figshare_meningioma")

    kaggle_train = {cls: collect_images(KAGGLE_TRAIN_DIR, cls)
                    for cls in STANDARD_CLASSES}
    kaggle_test  = {cls: collect_images(KAGGLE_TEST_DIR,  cls)
                    for cls in STANDARD_CLASSES}
    figshare     = {cls: collect_images(FIGSHARE_DIR,     cls)
                    for cls in STANDARD_CLASSES}

    print("\n[AUGMENTATION] Figshare meningioma offline augmentation...")
    fs_mening_orig = figshare["meningioma"]
    fs_mening_aug  = augment_figshare_meningioma(fs_mening_orig, aug_temp)
    fs_mening_train = fs_mening_orig + fs_mening_aug

    figshare_for_train          = dict(figshare)
    figshare_for_train["meningioma"] = fs_mening_train

    print("\n" + "─"*60)
    print("  ROUND 1 — Train: Figshare + Mendeley  |  Test: Kaggle")
    print("─"*60)
    r1_dir = os.path.join(HARMONIZED_ROOT, "lodo_round1")
    counts["round1"] = {"train": {}, "test": {}}

    print("  Saving Round 1 training split...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R1 train", unit="class"):
        paths = figshare_for_train[cls] + mendeley_train[cls]
        n = save_split(paths, os.path.join(r1_dir, "train"), cls, "r1_train")
        counts["round1"]["train"][cls] = n

    print("  Saving Round 1 test split (Kaggle)...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R1 test ", unit="class"):
        n = save_split(kaggle_test[cls], os.path.join(r1_dir, "test"), cls, "r1_test")
        counts["round1"]["test"][cls] = n

    print("\n" + "─"*60)
    print("  ROUND 2 — Train: Kaggle + Mendeley  |  Test: Figshare")
    print("─"*60)
    r2_dir = os.path.join(HARMONIZED_ROOT, "lodo_round2")
    counts["round2"] = {"train": {}, "test": {}}

    print("  Saving Round 2 training split...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R2 train", unit="class"):
        paths = kaggle_train[cls] + mendeley_train[cls]
        n = save_split(paths, os.path.join(r2_dir, "train"), cls, "r2_train")
        counts["round2"]["train"][cls] = n

    print("  Saving Round 2 test split (Figshare originals only)...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R2 test ", unit="class"):
        n = save_split(figshare[cls], os.path.join(r2_dir, "test"), cls, "r2_test")
        counts["round2"]["test"][cls] = n

    print("\n" + "─"*60)
    print("  ROUND 3 — Train: Kaggle + Figshare  |  Test: Mendeley")
    print("─"*60)
    r3_dir = os.path.join(HARMONIZED_ROOT, "lodo_round3")
    counts["round3"] = {"train": {}, "test": {}}

    print("  Saving Round 3 training split...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R3 train", unit="class"):
        paths = kaggle_train[cls] + figshare_for_train[cls]
        n = save_split(paths, os.path.join(r3_dir, "train"), cls, "r3_train")
        counts["round3"]["train"][cls] = n

    print("  Saving Round 3 test split (Mendeley)...")
    for cls in tqdm(STANDARD_CLASSES, desc="    R3 test ", unit="class"):
        n = save_split(mendeley_test[cls], os.path.join(r3_dir, "test"), cls, "r3_test")
        counts["round3"]["test"][cls] = n

    temp_root = os.path.join(HARMONIZED_ROOT, "_aug_temp")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    print("\n  Temp augmentation folder cleaned up.")

    return counts

def save_csv_report(counts):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Round", "Split", "Class", "Count"])
        for rnd, splits in counts.items():
            for split, classes in splits.items():
                for cls, n in classes.items():
                    w.writerow([rnd, split, cls, n])
    print(f"  [SAVED] CSV  -> {REPORT_CSV}")

def save_summary(counts, mendeley_train, mendeley_test):
    os.makedirs(LOGS_DIR, exist_ok=True)

    n_fs_orig = len(collect_images(FIGSHARE_DIR, "meningioma"))
    n_fs_aug  = n_fs_orig * len(FIGSHARE_MENING_AUG_OPS)
    n_fs_tot  = n_fs_orig + n_fs_aug

    lines = [
        "=" * 68,
        "  MODULE 1 — STEP 4: Harmonization & LODO Splits  (v3 FINAL)",
        "=" * 68,
        "",
        "  CONFIGURATION",
        f"  Image size          : {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px",
        f"  JPEG quality        : {JPEG_QUALITY}",
        f"  Random seed         : {RANDOM_SEED}",
        f"  Mendeley resplit    : stratified 80/20",
        "",
        "  AUGMENTATION (Figshare meningioma only)",
        f"  Original images     : {n_fs_orig}",
        f"  Operations applied  : {', '.join(FIGSHARE_MENING_AUG_OPS)}",
        f"  Augmented copies    : {n_fs_aug}",
        f"  Total for training  : {n_fs_tot}",
        f"  Augmentation ratio  : {n_fs_tot/n_fs_orig:.1f}x",
        "",
        "  All other class imbalance handled in Module 3:",
        "    - WeightedRandomSampler (oversamples minority each epoch)",
        "    - Focal Loss gamma=2 (focuses on hard/rare examples)",
        "    - Online augmentation in DataLoader (unlimited diversity)",
        "",
        "  MENDELEY RESPLIT",
        f"  {'Class':<14} {'Total':>6}  {'Train':>6}  {'Test':>6}",
        "  " + "-" * 36,
    ]

    for cls in STANDARD_CLASSES:
        tr  = len(mendeley_train[cls])
        te  = len(mendeley_test[cls])
        tot = tr + te
        lines.append(f"  {cls:<14} {tot:>6}  {tr:>6}  {te:>6}")

    lines.append("")

    for rnd_key, rnd_label, train_src, test_src in [
        ("round1", "ROUND 1", "Figshare + Mendeley", "Kaggle"),
        ("round2", "ROUND 2", "Kaggle + Mendeley",   "Figshare"),
        ("round3", "ROUND 3", "Kaggle + Figshare",   "Mendeley"),
    ]:
        lines += [
            "  " + "─" * 66,
            f"  {rnd_label}  Train: {train_src}  |  Test: {test_src}",
            "  " + "─" * 66,
            f"  {'Split':<8} {'Glioma':>8} {'Mening':>8} "
            f"{'Pituit':>8} {'NoTumor':>8} {'Total':>8}",
            "  " + "-" * 52,
        ]
        for split in ["train", "test"]:
            c = counts[rnd_key][split]
            g = c.get("glioma",     0)
            m = c.get("meningioma", 0)
            p = c.get("pituitary",  0)
            n = c.get("notumor",    0)
            t = g + m + p + n
            lines.append(
                f"  {split:<8} {g:>8} {m:>8} {p:>8} {n:>8} {t:>8}"
            )

        train_vals = [v for v in counts[rnd_key]["train"].values() if v > 0]
        if train_vals:
            ratio = max(train_vals) / min(train_vals)
            lines.append(f"\n  Training imbalance ratio: {ratio:.2f}x "
                         f"→ handled by WeightedRandomSampler + Focal Loss in Module 3")
        lines.append("")

    lines += [
        "  KNOWN LIMITATIONS",
        "  - Round 2 Figshare meningioma test = 58 images.",
        "    Report 95% bootstrap confidence intervals for Round 2 meningioma.",
        "  - Round 2 Figshare NoTumor test = 0 (class absent in Figshare).",
        "    Do not report NoTumor metrics for Round 2.",
        "",
        "  Next: Module 2 — Hybrid CNN-ViT Model Architecture.",
        "=" * 68,
    ]

    report = '\n'.join(lines)
    print('\n' + report)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  [SAVED] Summary -> {SUMMARY_TXT}")

def plot_fold_counts(counts):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "LODO Fold Image Counts — After Harmonisation\n"
        "Round 1: Test=Kaggle  ·  Round 2: Test=Figshare  ·  Round 3: Test=Mendeley",
        fontsize=12, fontweight='bold'
    )

    cls_labels  = ["Glioma", "Mening.", "Pituit.", "NoTumor"]
    train_color = '#2196F3'
    test_color  = '#FF9800'

    for ax, (rnd_key, rnd_title) in zip(axes, [
        ("round1", "Round 1 (Test: Kaggle)"),
        ("round2", "Round 2 (Test: Figshare)"),
        ("round3", "Round 3 (Test: Mendeley)"),
    ]):
        x     = np.arange(len(STANDARD_CLASSES))
        width = 0.35
        train_counts = [counts[rnd_key]["train"].get(c, 0) for c in STANDARD_CLASSES]
        test_counts  = [counts[rnd_key]["test"].get(c, 0)  for c in STANDARD_CLASSES]

        b1 = ax.bar(x - width/2, train_counts, width, label='Train',
                    color=train_color, alpha=0.85, edgecolor='white')
        b2 = ax.bar(x + width/2, test_counts,  width, label='Test',
                    color=test_color,  alpha=0.85, edgecolor='white')

        for bar, count in zip(list(b1) + list(b2), train_counts + test_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 10, str(count),
                        ha='center', va='bottom', fontsize=7,
                        fontweight='bold', rotation=45)

        ax.set_title(rnd_title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cls_labels, fontsize=9)
        ax.set_ylabel("Images", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "lodo_fold_counts.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Chart  -> {out}")

def plot_sample_grid(round_num, n_per_class=4):
    rnd_dir   = os.path.join(HARMONIZED_ROOT, f"lodo_round{round_num}", "train")
    save_path = os.path.join(PLOTS_DIR, f"sample_grid_round{round_num}.png")

    cls_labels = {"glioma": "Glioma", "meningioma": "Meningioma",
                  "pituitary": "Pituitary", "notumor": "No Tumor"}
    fig, axes = plt.subplots(len(STANDARD_CLASSES), n_per_class,
                             figsize=(n_per_class * 2.5, len(STANDARD_CLASSES) * 2.5))
    fig.suptitle(f"Sample Training Images — LODO Round {round_num}",
                 fontsize=12, fontweight='bold')

    for row, cls in enumerate(STANDARD_CLASSES):
        cls_dir    = os.path.join(rnd_dir, cls)
        all_files  = []
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
    print(f"  [SAVED] Grid   -> {save_path}")

def main():
    print("=" * 68)
    print("  MODULE 1 — STEP 4: Harmonization & LODO Split Creation")
    print("  Version 3 — FINAL  (Figshare meningioma aug only)")
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
        print("\n  One or more input directories not found.")
        print("  Please run Step 3 (step3_deduplicate.py) first.")
        return

    print("\n[0/6] Removing previous harmonized output (if any)...")
    if os.path.exists(HARMONIZED_ROOT):
        shutil.rmtree(HARMONIZED_ROOT)
        print("  Removed previous output.")
    os.makedirs(HARMONIZED_ROOT, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n[1/6] Mendeley stratified 80/20 resplit (seed=42)...")
    mendeley_train, mendeley_test = resplit_mendeley()

    print("\n[2/6] Assembling LODO folds...")
    counts = assemble_lodo_folds(mendeley_train, mendeley_test)

    print("\n[3/6] Generating fold count chart...")
    plot_fold_counts(counts)

    print("\n[4/6] Generating sample image grids...")
    for rnd in [1, 2, 3]:
        plot_sample_grid(rnd)

    print("\n[5/6] Saving CSV report...")
    save_csv_report(counts)

    print("\n[6/6] Saving text summary...")
    save_summary(counts, mendeley_train, mendeley_test)

    print("\n" + "=" * 68)
    print("  STEP 4 COMPLETE")
    print("=" * 68)
    print(f"\n  Output root: {HARMONIZED_ROOT}\n")
    for rnd in [1, 2, 3]:
        for split in ["train", "test"]:
            d     = os.path.join(HARMONIZED_ROOT, f"lodo_round{rnd}", split)
            total = 0
            if os.path.isdir(d):
                for cls in STANDARD_CLASSES:
                    cls_dir = os.path.join(d, cls)
                    if os.path.isdir(cls_dir):
                        total += len([
                            f for f in os.listdir(cls_dir)
                            if os.path.splitext(f.lower())[1] in VALID_EXT
                        ])
            print(f"    lodo_round{rnd}\\{split}\\  —  {total:,} images")

    print(f"\n  Class imbalance will be handled in Module 3 via:")
    print(f"    WeightedRandomSampler + Focal Loss (gamma=2) + Online Augmentation")
    print(f"\n  Ready for Module 2: Hybrid CNN-ViT Architecture.")
    print("=" * 68)
