
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.io
import warnings
warnings.filterwarnings('ignore')

try:
    import mat73
    MAT73_AVAILABLE = True
except ImportError:
    MAT73_AVAILABLE = False
    print("[WARNING] mat73 not found. Will use scipy only.")
    print("          If you see errors, run: pip install mat73")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGSHARE_INPUT = os.path.join(PROJECT_ROOT, "datasets", "figshare")

FIGSHARE_OUTPUT = os.path.join(PROJECT_ROOT, "datasets", "figshare_converted")

FIGSHARE_SUBFOLDERS = [
    "brainTumorDataPublic_1-766",
    "brainTumorDataPublic_767-1532",
    "brainTumorDataPublic_1533-2298",
    "brainTumorDataPublic_2299-3064"
]

LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

TARGET_SIZE = (224, 224)

JPEG_QUALITY = 95

def load_mat_file(mat_path):
    """
    Load a .mat file and return its contents as a dictionary.
    Tries scipy first (for older .mat format), then mat73 (for v7.3 HDF5 format).

    Args:
        mat_path: full path to the .mat file

    Returns:
        Dictionary containing the .mat file contents, or None if loading fails
    """
    try:
        data = scipy.io.loadmat(mat_path)
        return data, 'scipy'
    except Exception as scipy_error:
        pass

    if MAT73_AVAILABLE:
        try:
            data = mat73.loadmat(mat_path)
            return data, 'mat73'
        except Exception as mat73_error:
            pass

    return None, None

def extract_image_and_label(data, loader_type, mat_path):
    """
    Extract the MRI image array and class label from the loaded .mat dictionary.

    The Figshare .mat files contain these keys:
        'cjdata' — a struct containing:
            'image'  — the MRI image as a 2D or 3D numpy array
            'label'  — the class label (1, 2, or 3)
            'tumorMask' — binary mask (we ignore this)
            'PID'    — patient ID (we ignore this)

    Args:
        data: dictionary loaded from the .mat file
        loader_type: 'scipy' or 'mat73' (affects how to access nested structs)
        mat_path: path to the original file (for error messages)

    Returns:
        tuple: (image_array as numpy array, label as integer)
               or (None, None) if extraction fails
    """
    try:
        if loader_type == 'scipy':
            cjdata = data['cjdata']

            if hasattr(cjdata, 'dtype') and cjdata.dtype.names:
                image = cjdata['image'][0, 0]
                label = int(cjdata['label'][0, 0].flat[0])
            else:
                image = cjdata[0, 0]['image']
                label = int(cjdata[0, 0]['label'].flat[0])

        elif loader_type == 'mat73':
            cjdata = data['cjdata']
            image = np.array(cjdata['image'])
            label = int(cjdata['label'])

        else:
            return None, None

        image = np.array(image, dtype=np.float64)

        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)

        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image = np.zeros_like(image)

        image = image.astype(np.uint8)

        return image, label

    except Exception as e:
        print(f"\n  [ERROR] Failed to extract from {os.path.basename(mat_path)}: {e}")
        return None, None

def convert_to_rgb_pil(image_array):
    """
    Convert a numpy image array to a PIL RGB image.

    MRI images in Figshare are grayscale (2D arrays).
    We convert to RGB because our model (EfficientNet-B0) expects 3-channel input.
    Converting grayscale to RGB simply copies the single channel 3 times.
    This is the standard approach in medical imaging deep learning.

    Args:
        image_array: 2D or 3D numpy uint8 array

    Returns:
        PIL Image in RGB mode, resized to TARGET_SIZE
    """
    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array, mode='L').convert('RGB')
    elif image_array.ndim == 3:
        if image_array.shape[2] == 1:
            pil_img = Image.fromarray(image_array[:, :, 0], mode='L').convert('RGB')
        elif image_array.shape[2] == 3:
            pil_img = Image.fromarray(image_array, mode='RGB')
        else:
            pil_img = Image.fromarray(image_array[:, :, 0], mode='L').convert('RGB')
    else:
        raise ValueError(f"Unexpected image dimensions: {image_array.shape}")

    pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)

    return pil_img

def create_output_folders():
    """
    Create the output folder structure:
        datasets\figshare_converted\
            glioma\
            meningioma\
            pituitary\
    Note: No 'notumor' folder because Figshare has no No Tumor class.
    """
    for class_name in LABEL_MAP.values():
        folder = os.path.join(FIGSHARE_OUTPUT, class_name)
        os.makedirs(folder, exist_ok=True)
        print(f"  [CREATED] {folder}")

def verify_input_folders():
    """
    Check that all expected input folders exist before starting conversion.
    Prints a clear error message if anything is missing.

    Returns:
        True if all folders exist, False otherwise
    """
    print("\n[CHECKING] Verifying input folder structure...")

    all_ok = True

    if not os.path.exists(FIGSHARE_INPUT):
        print(f"  [ERROR] Figshare input folder not found: {FIGSHARE_INPUT}")
        print(f"          Make sure you placed the extracted Figshare files in datasets\\figshare\\")
        all_ok = False
    else:
        print(f"  [OK] Main Figshare folder: {FIGSHARE_INPUT}")

    for subfolder in FIGSHARE_SUBFOLDERS:
        full_path = os.path.join(FIGSHARE_INPUT, subfolder)
        if not os.path.exists(full_path):
            print(f"  [MISSING] {subfolder}")
            print(f"            Extract the ZIP file for this subfolder and try again.")
            all_ok = False
        else:
            mat_files = [f for f in os.listdir(full_path) if f.endswith('.mat')]
            print(f"  [OK] {subfolder}: {len(mat_files)} .mat files found")

    return all_ok

def convert_figshare():
    """
    Main conversion pipeline.
    Scans all .mat files, converts each one to JPEG, and saves to output folders.
    """

    print("=" * 70)
    print("  MODULE 1 — STEP 1: Figshare .mat to JPEG Conversion")
    print("  Brain Tumor Classification Research")
    print("=" * 70)

    if not verify_input_folders():
        print("\n[ABORTED] Fix the missing folders above and run again.")
        return

    print(f"\n[CREATING] Output folders in: {FIGSHARE_OUTPUT}")
    create_output_folders()

    print("\n[SCANNING] Collecting all .mat files...")
    all_mat_files = []

    for subfolder in FIGSHARE_SUBFOLDERS:
        subfolder_path = os.path.join(FIGSHARE_INPUT, subfolder)
        mat_files = sorted([
            os.path.join(subfolder_path, f)
            for f in os.listdir(subfolder_path)
            if f.endswith('.mat') and f != 'cvind.mat'
        ])
        all_mat_files.extend(mat_files)
        print(f"  Found {len(mat_files)} .mat files in {subfolder}")

    total_files = len(all_mat_files)
    print(f"\n  Total .mat files to convert: {total_files}")
    print(f"  Expected: ~3,064 files\n")

    counters = {name: 0 for name in LABEL_MAP.values()}
    skipped    = 0
    errors     = 0
    unknown_labels = []

    print("[CONVERTING] Processing .mat files...")
    print("  (This may take 5-15 minutes depending on your computer speed)\n")

    for mat_path in tqdm(all_mat_files, desc="Converting", unit="file",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):

        data, loader_type = load_mat_file(mat_path)

        if data is None:
            errors += 1
            tqdm.write(f"  [ERROR] Could not load: {os.path.basename(mat_path)}")
            continue

        image_array, label = extract_image_and_label(data, loader_type, mat_path)

        if image_array is None or label is None:
            skipped += 1
            continue

        if label not in LABEL_MAP:
            unknown_labels.append((os.path.basename(mat_path), label))
            skipped += 1
            continue

        class_name = LABEL_MAP[label]

        try:
            pil_img = convert_to_rgb_pil(image_array)
        except Exception as e:
            tqdm.write(f"  [ERROR] Image conversion failed for {os.path.basename(mat_path)}: {e}")
            errors += 1
            continue

        mat_basename = os.path.splitext(os.path.basename(mat_path))[0]
        output_filename = f"figshare_{mat_basename.zfill(4)}.jpg"
        output_path = os.path.join(FIGSHARE_OUTPUT, class_name, output_filename)

        try:
            pil_img.save(output_path, format='JPEG', quality=JPEG_QUALITY)
            counters[class_name] += 1
        except Exception as e:
            tqdm.write(f"  [ERROR] Could not save {output_filename}: {e}")
            errors += 1
            continue

    total_converted = sum(counters.values())

    print("\n" + "=" * 70)
    print("  CONVERSION COMPLETE — REPORT")
    print("=" * 70)
    print(f"\n  Total .mat files found:     {total_files}")
    print(f"  Successfully converted:     {total_converted}")
    print(f"  Skipped (bad data):         {skipped}")
    print(f"  Errors:                     {errors}")

    print(f"\n  Images saved per class:")
    print(f"    Glioma:      {counters['glioma']:>5} images")
    print(f"    Meningioma:  {counters['meningioma']:>5} images")
    print(f"    Pituitary:   {counters['pituitary']:>5} images")
    print(f"    No Tumor:    {'N/A':>5} (Figshare has no No Tumor class)")
    print(f"    TOTAL:       {total_converted:>5} images")

    print(f"\n  Expected counts (from Figshare README):")
    print(f"    Glioma:      1,426 images")
    print(f"    Meningioma:    708 images")
    print(f"    Pituitary:     930 images")
    print(f"    TOTAL:       3,064 images")

    if unknown_labels:
        print(f"\n  [WARNING] Files with unexpected labels (not 1, 2, or 3):")
        for fname, lbl in unknown_labels[:10]:
            print(f"    {fname}: label={lbl}")

    print(f"\n  Output saved to: {FIGSHARE_OUTPUT}")

    print("\n[VERIFYING] Checking output folder contents...")
    all_ok = True
    for class_name in LABEL_MAP.values():
        out_folder = os.path.join(FIGSHARE_OUTPUT, class_name)
        actual_count = len([f for f in os.listdir(out_folder) if f.endswith('.jpg')])
        status = "[OK]" if actual_count > 0 else "[EMPTY!]"
        print(f"  {status} {class_name}: {actual_count} JPEG files in output folder")
        if actual_count == 0:
            all_ok = False

    if all_ok:
        print("\n  [SUCCESS] All class folders have images.")
        print("  You are ready to run Step 2: Dataset Inspection & Class Count Report.")
    else:
        print("\n  [WARNING] One or more class folders are empty.")
        print("  Check the errors above and re-run if needed.")

    print("\n" + "=" * 70)
