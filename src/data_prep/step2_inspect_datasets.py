
import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KAGGLE_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "kaggle", "Training")
KAGGLE_TEST  = os.path.join(PROJECT_ROOT, "datasets", "kaggle", "Testing")

FIGSHARE_CONVERTED = os.path.join(PROJECT_ROOT, "datasets", "figshare_converted")

MENDELEY_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "mendeley", "Train")
MENDELEY_TEST  = os.path.join(PROJECT_ROOT, "datasets", "mendeley", "Test")

OUTPUTS_PLOTS = os.path.join(PROJECT_ROOT, "outputs", "plots")
os.makedirs(OUTPUTS_PLOTS, exist_ok=True)

REPORT_CSV = os.path.join(PROJECT_ROOT, "outputs", "logs",
                          "dataset_inspection_report.csv")
os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)

STANDARD_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

KAGGLE_CLASS_MAP = {
    "glioma":     "glioma",
    "meningioma": "meningioma",
    "pituitary":  "pituitary",
    "notumor":    "notumor",
}

FIGSHARE_CLASS_MAP = {
    "glioma":     "glioma",
    "meningioma": "meningioma",
    "pituitary":  "pituitary",
    "notumor":    None,
}

MENDELEY_CLASS_MAP = {
    "glioma":     "glioma",
    "meningioma": "meningioma",
    "pituitary":  "pituitary",
    "notumor":    "notumor",
}

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

SIZE_SAMPLE_LIMIT = 50

def count_images_in_folder(folder_path):
    """
    Count all valid image files in a folder (non-recursive).
    Returns count and list of filenames.
    """
    if not os.path.exists(folder_path):
        return 0, []
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS
    ]
    return len(files), files

def scan_dataset(name, split_paths, class_map):
    """
    Scan one dataset and return a structured result dictionary.

    Args:
        name       : human-readable dataset name (e.g. "Kaggle")
        split_paths: dict of {"train": path, "test": path} or {"all": path}
        class_map  : dict mapping standard class names to actual folder names

    Returns:
        dict with keys: name, classes, splits, corrupted, sizes
    """
    result = {
        "name":      name,
        "classes":   {cls: 0 for cls in STANDARD_CLASSES},
        "splits":    {},
        "corrupted": [],
        "sizes":     {}
    }

    for split_name, base_path in split_paths.items():
        result["splits"][split_name] = {cls: 0 for cls in STANDARD_CLASSES}

        if not os.path.exists(base_path):
            print(f"  [WARNING] Path not found: {base_path}")
            continue

        for std_class in STANDARD_CLASSES:
            actual_folder = class_map.get(std_class)

            if actual_folder is None:
                continue

            class_path = os.path.join(base_path, actual_folder)
            if not os.path.exists(class_path):
                print(f"  [WARNING] Class folder not found: {class_path}")
                continue

            count, files = count_images_in_folder(class_path)
            result["classes"][std_class]            += count
            result["splits"][split_name][std_class]  = count

            if std_class not in result["sizes"]:
                result["sizes"][std_class] = []

            sample_files = files[:SIZE_SAMPLE_LIMIT]
            for fname in sample_files:
                fpath = os.path.join(class_path, fname)
                try:
                    with Image.open(fpath) as img:
                        result["sizes"][std_class].append(img.size)
                except Exception:
                    result["corrupted"].append(fpath)

    return result

def analyse_sizes(sizes_dict):
    """
    Summarise image dimensions for a dataset.
    Returns dict: class -> {min_w, max_w, avg_w, min_h, max_h, avg_h, uniform}
    """
    summary = {}
    for cls, size_list in sizes_dict.items():
        if not size_list:
            continue
        widths  = [s[0] for s in size_list]
        heights = [s[1] for s in size_list]
        summary[cls] = {
            "min_w":   min(widths),
            "max_w":   max(widths),
            "avg_w":   int(np.mean(widths)),
            "min_h":   min(heights),
            "max_h":   max(heights),
            "avg_h":   int(np.mean(heights)),
            "uniform": (min(widths) == max(widths) and
                        min(heights) == max(heights))
        }
    return summary

def plot_class_distribution(dataset_results):
    """
    Grouped bar chart: class distribution across all three datasets.
    Saved to outputs/plots/class_distribution.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Class Distribution Across All Three Datasets\n"
        "Kaggle (A)  ·  Figshare (B)  ·  Mendeley (C)",
        fontsize=16, fontweight='bold', y=1.02
    )

    colours = ['#2196F3', '#4CAF50', '#FF9800']

    ax = axes[0]
    x     = np.arange(len(STANDARD_CLASSES))
    width = 0.22

    for i, (res, colour) in enumerate(zip(dataset_results, colours)):
        counts = [res["classes"][cls] for cls in STANDARD_CLASSES]
        bars   = ax.bar(x + i * width, counts, width,
                        label=res["name"], color=colour, alpha=0.85,
                        edgecolor='white')
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 20,
                        str(count), ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax.set_xlabel("Tumor Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Absolute Image Count per Class", fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Glioma", "Meningioma", "Pituitary", "No Tumor"],
                       fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    for i, (res, colour) in enumerate(zip(dataset_results, colours)):
        total = sum(res["classes"].values())
        if total == 0:
            continue
        percs = [res["classes"][cls] / total * 100 for cls in STANDARD_CLASSES]
        bars  = ax2.bar(x + i * width, percs, width,
                        label=res["name"], color=colour, alpha=0.85,
                        edgecolor='white')
        for bar, pct in zip(bars, percs):
            if pct > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.3,
                         f"{pct:.1f}%", ha='center', va='bottom',
                         fontsize=7.5, fontweight='bold')

    ax2.set_xlabel("Tumor Class", fontsize=12)
    ax2.set_ylabel("Percentage of Dataset (%)", fontsize=12)
    ax2.set_title("Percentage Distribution per Class", fontsize=13,
                  fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(["Glioma", "Meningioma", "Pituitary", "No Tumor"],
                        fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_PLOTS, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Class distribution chart -> {save_path}")

def plot_dataset_totals(dataset_results):
    """
    Pie chart: total image contribution per dataset.
    Saved to outputs/plots/dataset_totals.png
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    labels  = [r["name"] for r in dataset_results]
    sizes   = [sum(r["classes"].values()) for r in dataset_results]
    colours = ['#2196F3', '#4CAF50', '#FF9800']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colours, explode=explode,
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')

    ax.set_title(
        f"Total Images per Dataset\n(Grand Total: {sum(sizes):,} images)\n"
        f"Note: Mendeley count includes augmented variants\n"
        f"and Kaggle-origin images. Step 3 deduplication\n"
        f"will reduce Mendeley to independent images only.",
        fontsize=11, fontweight='bold'
    )

    legend_labels = [f"{l}: {s:,} images" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.12), fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_PLOTS, "dataset_totals.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Dataset totals pie chart -> {save_path}")

def plot_imbalance_heatmap(dataset_results):
    """
    Heatmap: class imbalance across all datasets.
    Colour intensity = relative proportion within each dataset.
    Saved to outputs/plots/imbalance_heatmap.png
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ds_names   = [r["name"] for r in dataset_results]
    cls_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    data = np.array([
        [r["classes"][cls] for cls in STANDARD_CLASSES]
        for r in dataset_results
    ], dtype=float)

    row_maxes = data.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1
    data_norm = data / row_maxes

    im = ax.imshow(data_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(cls_labels)))
    ax.set_yticks(range(len(ds_names)))
    ax.set_xticklabels(cls_labels, fontsize=12)
    ax.set_yticklabels(ds_names, fontsize=12)
    ax.set_title(
        "Class Imbalance Heatmap\n"
        "(Colour = relative proportion within each dataset; numbers = actual count)\n"
        "Figshare 'No Tumor' = N/A (class does not exist in Figshare)",
        fontsize=11, fontweight='bold'
    )

    for i in range(len(ds_names)):
        for j in range(len(cls_labels)):
            count = int(data[i, j])
            if ds_names[i] == "Figshare" and cls_labels[j] == "No Tumor":
                text = "N/A"
            else:
                text = str(count) if count > 0 else "0"
            colour = 'white' if data_norm[i, j] > 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=13, fontweight='bold', color=colour)

    plt.colorbar(im, ax=ax, label='Relative proportion within dataset')
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_PLOTS, "imbalance_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Imbalance heatmap -> {save_path}")

def save_csv_report(dataset_results, size_summaries):
    """Save full structured CSV report with counts and size statistics."""
    with open(REPORT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Split", "Class", "Count",
            "Avg Width", "Avg Height",
            "Min Width", "Max Width",
            "Min Height", "Max Height",
            "Uniform Size"
        ])
        for res, size_sum in zip(dataset_results, size_summaries):
            for split_name, split_data in res["splits"].items():
                for cls in STANDARD_CLASSES:
                    count = split_data.get(cls, 0)
                    sz    = size_sum.get(cls, {})
                    writer.writerow([
                        res["name"], split_name, cls, count,
                        sz.get("avg_w", "N/A"), sz.get("avg_h", "N/A"),
                        sz.get("min_w", "N/A"), sz.get("max_w", "N/A"),
                        sz.get("min_h", "N/A"), sz.get("max_h", "N/A"),
                        sz.get("uniform", "N/A")
                    ])
    print(f"  [SAVED] CSV report -> {REPORT_CSV}")

def inspect_datasets():

    print("=" * 70)
    print("  MODULE 1 — STEP 2: Dataset Inspection & Class Count Report")
    print("  Brain Tumor Classification Research")
    print("  Datasets: Kaggle (A) | Figshare (B) | Mendeley (C)")
    print("=" * 70)

    print("\n[SCANNING] Scanning all three datasets...\n")

    print("  [A] Scanning Kaggle (Nickparvar) dataset...")
    kaggle_result = scan_dataset(
        name="Kaggle",
        split_paths={"train": KAGGLE_TRAIN, "test": KAGGLE_TEST},
        class_map=KAGGLE_CLASS_MAP
    )

    print("  [B] Scanning Figshare (Jun Cheng, converted) dataset...")
    figshare_result = scan_dataset(
        name="Figshare",
        split_paths={"all": FIGSHARE_CONVERTED},
        class_map=FIGSHARE_CLASS_MAP
    )

    print("  [C] Scanning Mendeley 12k dataset...")
    mendeley_result = scan_dataset(
        name="Mendeley",
        split_paths={"train": MENDELEY_TRAIN, "test": MENDELEY_TEST},
        class_map=MENDELEY_CLASS_MAP
    )

    dataset_results = [kaggle_result, figshare_result, mendeley_result]

    print("\n[ANALYSING] Computing image size statistics (sample of 50 per class)...")
    size_summaries = [analyse_sizes(r["sizes"]) for r in dataset_results]

    print("\n" + "=" * 70)
    print("  DATASET INSPECTION REPORT")
    print("=" * 70)

    grand_total = 0
    for res, size_sum in zip(dataset_results, size_summaries):
        ds_total    = sum(res["classes"].values())
        grand_total += ds_total

        print(f"\n{'─'*70}")
        print(f"  DATASET: {res['name']}")
        print(f"{'─'*70}")
        print(f"  Total images: {ds_total:,}")

        for split_name, split_data in res["splits"].items():
            split_total = sum(split_data.values())
            print(f"\n  [{split_name.upper()} SPLIT]  —  {split_total:,} images")
            print(f"  {'Class':<15} {'Count':>8}  {'% of split':>12}  {'Avg Size':>12}  {'Uniform?':>10}")
            print(f"  {'─'*15} {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")

            for cls in STANDARD_CLASSES:
                count = split_data.get(cls, 0)
                pct   = (count / split_total * 100) if split_total > 0 else 0
                sz    = size_sum.get(cls, {})
                size_str    = f"{sz['avg_w']}x{sz['avg_h']}" if sz else "N/A"
                uniform_str = ("Yes" if sz.get("uniform") else "No") if sz else "N/A"

                if count == 0 and cls == "notumor" and res["name"] == "Figshare":
                    count_str = "N/A"
                    pct_str   = "— (no class)"
                else:
                    count_str = f"{count:,}"
                    pct_str   = f"{pct:.1f}%"

                print(f"  {cls:<15} {count_str:>8}  {pct_str:>12}  {size_str:>12}  {uniform_str:>10}")

        if res["corrupted"]:
            print(f"\n  [WARNING] {len(res['corrupted'])} corrupted/unreadable files found:")
            for fpath in res["corrupted"][:5]:
                print(f"    {fpath}")
            if len(res["corrupted"]) > 5:
                print(f"    ... and {len(res['corrupted']) - 5} more")
        else:
            print(f"\n  [OK] No corrupted files detected.")

    print(f"\n{'=' * 70}")
    print(f"  GRAND TOTAL ACROSS ALL DATASETS: {grand_total:,} images")
    print(f"  (Before deduplication)")
    print(f"  NOTE: Mendeley total includes augmented variants and Kaggle-origin")
    print(f"        images. Step 3 deduplication retains only the clinically")
    print(f"        independent Epic/CSCR hospital images from Mendeley.")
    print(f"{'=' * 70}")

    print("\n[ANALYSIS] Class Imbalance Assessment:")
    combined = {cls: 0 for cls in STANDARD_CLASSES}
    for res in dataset_results:
        for cls in STANDARD_CLASSES:
            combined[cls] += res["classes"][cls]

    combined_total  = sum(combined.values())
    max_count       = max(combined.values())
    min_count       = min(v for v in combined.values() if v > 0)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\n  Combined class counts (all datasets merged):")
    for cls in STANDARD_CLASSES:
        count = combined[cls]
        pct   = (count / combined_total * 100) if combined_total > 0 else 0
        bar   = '█' * int(pct / 2)
        print(f"  {cls:<15} {count:>6}  ({pct:.1f}%)  {bar}")

    print(f"\n  Imbalance ratio (most/least common): {imbalance_ratio:.2f}x")
    if imbalance_ratio > 3:
        print("  [WARNING] Significant imbalance detected.")
        print("            Module 3 training loop will apply class weighting.")
    else:
        print("  [OK] Class imbalance is within acceptable range.")

    print("\n[NOTE] Mendeley Dataset:")
    print("  - Contains augmented variants (BR/DA/HF/RO/SP/VF suffixes).")
    print("  - Contains Kaggle-origin images (Te-gl_, Tr-gl_ naming).")
    print("  - Step 3 will remove Kaggle-origin images via pHash deduplication.")
    print("  - Remaining images are verified independent clinical scans (Epic/CSCR).")

    print("\n[ANALYSIS] Image Size Assessment:")
    print("  All images will be resized to 224x224 in Step 4 (harmonization).")
    print("  Current native sizes (sampled):")
    for res, size_sum in zip(dataset_results, size_summaries):
        sizes_seen = set()
        for cls, sz in size_sum.items():
            sizes_seen.add((sz.get('avg_w', '?'), sz.get('avg_h', '?')))
        if sizes_seen:
            size_desc = ", ".join([f"{w}x{h}" for w, h in sorted(sizes_seen)])
            print(f"    {res['name']:<12}: {size_desc}")

    print("\n[GENERATING] Creating visualisation charts...")
    plot_class_distribution(dataset_results)
    plot_dataset_totals(dataset_results)
    plot_imbalance_heatmap(dataset_results)

    print("\n[SAVING] Writing CSV report...")
    save_csv_report(dataset_results, size_summaries)

    print("\n" + "=" * 70)
    print("  STEP 2 COMPLETE")
    print("=" * 70)
    print(f"\n  Charts saved to : {OUTPUTS_PLOTS}")
    print(f"  CSV report      : {REPORT_CSV}")
    print(f"\n  Review these files before proceeding to Step 3:")
    print(f"    outputs\\plots\\class_distribution.png")
    print(f"    outputs\\plots\\dataset_totals.png")
    print(f"    outputs\\plots\\imbalance_heatmap.png")
    print(f"    outputs\\logs\\dataset_inspection_report.csv")
    print(f"\n  When ready, run: python module1_data_prep\\step3_deduplicate.py")
    print("=" * 70)
