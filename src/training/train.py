
import os, sys, csv, time, argparse, random
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2_model.model import build_model, FocalLoss, compute_class_weights

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARMONIZED_DIR = os.path.join(PROJECT_ROOT, "outputs_v1_224", "harmonized_dataset")
if not os.path.isdir(HARMONIZED_DIR):
    HARMONIZED_DIR = os.path.join(PROJECT_ROOT, "outputs", "harmonized_dataset")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "models")

CLASS_NAMES  = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES  = 4
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_SIZE        = 320
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 24
NUM_EPOCHS      = 100
EARLY_STOP_PAT  = 15
AUX_LOSS_WEIGHT = 0.3
FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS   = 5
VAL_SPLIT_RATIO = 0.10
RANDOM_SEED     = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PROTOCOLS = {
    "merged": {
        "train_dir":   os.path.join(HARMONIZED_DIR, "merged_baseline", "train"),
        "val_dir":     os.path.join(HARMONIZED_DIR, "merged_baseline", "val"),
        "test_dir":    os.path.join(HARMONIZED_DIR, "merged_baseline", "test"),
        "has_val_dir": True,
        "description": "Merged Baseline (80/10/10 split)",
    },
    "lodo1": {
        "train_dir":   os.path.join(HARMONIZED_DIR, "lodo_round1", "train"),
        "val_dir":     None,
        "test_dir":    os.path.join(HARMONIZED_DIR, "lodo_round1", "test"),
        "has_val_dir": False,
        "description": "LODO Round 1 — Train: Figshare+Mendeley | Test: Kaggle",
    },
    "lodo2": {
        "train_dir":   os.path.join(HARMONIZED_DIR, "lodo_round2", "train"),
        "val_dir":     None,
        "test_dir":    os.path.join(HARMONIZED_DIR, "lodo_round2", "test"),
        "has_val_dir": False,
        "description": "LODO Round 2 — Train: Kaggle+Mendeley | Test: Figshare",
    },
    "lodo3": {
        "train_dir":   os.path.join(HARMONIZED_DIR, "lodo_round3", "train"),
        "val_dir":     None,
        "test_dir":    os.path.join(HARMONIZED_DIR, "lodo_round3", "test"),
        "has_val_dir": False,
        "description": "LODO Round 3 — Train: Kaggle+Figshare | Test: Mendeley",
    },
}

class BrainTumorDataset(Dataset):
    VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(self, base_dir, transform=None, classes=None):
        self.base_dir = base_dir
        self.transform = transform
        self.samples = []
        self.class_counts = {}
        for cls_name in (classes or CLASS_NAMES):
            cls_dir = os.path.join(base_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            imgs = sorted([
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if os.path.splitext(f.lower())[1] in self.VALID_EXT
            ])
            if len(imgs) == 0:
                continue
            self.samples.extend([(p, cls_idx) for p in imgs])
            self.class_counts[cls_name] = len(imgs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

def get_train_transform():
    """v2: RandAugment + Random Erasing for stronger generalization."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_weighted_sampler(dataset):
    counts_by_idx = {}
    for _, label in dataset.samples:
        counts_by_idx[label] = counts_by_idx.get(label, 0) + 1
    weights = torch.tensor(
        [1.0 / counts_by_idx[label] for _, label in dataset.samples], dtype=torch.float,
    )
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def split_train_val(dataset, val_ratio=VAL_SPLIT_RATIO, seed=RANDOM_SEED):
    random.seed(seed)
    by_class = {}
    for s in dataset.samples:
        by_class.setdefault(s[1], []).append(s)
    train_s, val_s = [], []
    for lbl, samps in by_class.items():
        random.shuffle(samps)
        n_val = max(1, int(len(samps) * val_ratio))
        val_s.extend(samps[:n_val])
        train_s.extend(samps[n_val:])

    def _make(samples, transform):
        ds = BrainTumorDataset.__new__(BrainTumorDataset)
        ds.base_dir = dataset.base_dir
        ds.transform = transform
        ds.samples = samples
        ds.class_counts = {
            CLASS_NAMES[lbl]: sum(1 for s in samples if s[1] == lbl)
            for lbl in by_class
        }
        return ds

    return _make(train_s, get_train_transform()), _make(val_s, get_eval_transform())

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PAT, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False

class MetricsTracker:
    def __init__(self):
        self.history = []

    def add(self, epoch, phase, metrics):
        entry = {'epoch': epoch, 'phase': phase}
        entry.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
        self.history.append(entry)

    def save_csv(self, path):
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)

    def get_phase(self, phase):
        return [h for h in self.history if h['phase'] == phase]

def train_one_epoch(model, loader, optimizer, focal_loss, device):
    model.train()
    total_loss = total_vit = total_cnn = 0.0
    vit_correct = cnn_correct = n = 0
    pbar = tqdm(loader, desc='  train', leave=False, unit='batch')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(images, return_attention=False)
        l_vit = focal_loss(out['vit_logits'], labels)
        l_cnn = focal_loss(out['cnn_logits'], labels)
        loss = l_vit + AUX_LOSS_WEIGHT * l_cnn
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_vit += l_vit.item() * bs
        total_cnn += l_cnn.item() * bs
        n += bs
        vit_correct += (out['vit_logits'].argmax(1) == labels).sum().item()
        cnn_correct += (out['cnn_logits'].argmax(1) == labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'vit_acc': f'{vit_correct/n:.3f}'})
    return {
        'loss': total_loss/n, 'vit_loss': total_vit/n, 'cnn_loss': total_cnn/n,
        'vit_acc': vit_correct/n, 'cnn_acc': cnn_correct/n,
    }

@torch.no_grad()
def evaluate(model, loader, focal_loss, device, phase='val', present_classes=None):
    model.eval()
    track = present_classes if present_classes else CLASS_NAMES
    total_loss = total_vit = total_cnn = 0.0
    vit_correct = cnn_correct = n = 0
    per_class_correct = {c: 0 for c in track}
    per_class_total = {c: 0 for c in track}
    pbar = tqdm(loader, desc=f'  {phase}', leave=False, unit='batch')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images, return_attention=False)
        l_vit = focal_loss(out['vit_logits'], labels)
        l_cnn = focal_loss(out['cnn_logits'], labels)
        loss = l_vit + AUX_LOSS_WEIGHT * l_cnn
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_vit += l_vit.item() * bs
        total_cnn += l_cnn.item() * bs
        n += bs
        vit_preds = out['vit_logits'].argmax(1)
        cnn_preds = out['cnn_logits'].argmax(1)
        vit_correct += (vit_preds == labels).sum().item()
        cnn_correct += (cnn_preds == labels).sum().item()
        for pred, true in zip(vit_preds.cpu(), labels.cpu()):
            cls = CLASS_NAMES[true.item()]
            if cls in track:
                per_class_total[cls] += 1
                per_class_correct[cls] += int(pred == true)
    return {
        'loss': total_loss/n, 'vit_loss': total_vit/n, 'cnn_loss': total_cnn/n,
        'vit_acc': vit_correct/n, 'cnn_acc': cnn_correct/n,
        'per_class_correct': per_class_correct, 'per_class_total': per_class_total,
    }

def plot_training_curves(tracker, out_dir, protocol):
    train_h = tracker.get_phase('train')
    val_h = tracker.get_phase('val')
    if not train_h:
        return
    epochs = [h['epoch'] for h in train_h]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {protocol.upper()} (v2: {IMG_SIZE}px)', fontsize=13, fontweight='bold')
    ax1.plot(epochs, [h['loss'] for h in train_h], 'b-o', ms=2, label='Train', lw=1)
    if val_h:
        ax1.plot([h['epoch'] for h in val_h], [h['loss'] for h in val_h], 'r-o', ms=2, label='Val', lw=1)
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h['vit_acc'] for h in train_h], 'b-o', ms=2, label='Train', lw=1)
    if val_h:
        ax2.plot([h['epoch'] for h in val_h], [h['vit_acc'] for h in val_h], 'r-o', ms=2, label='Val', lw=1)
    ax2.set_title('ViT Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_training_summary(protocol, tracker, best_epoch, total_time_s, train_counts, val_counts, out_dir, test_present_classes=None):
    train_h = tracker.get_phase('train')
    val_h = tracker.get_phase('val')
    best_val_loss = min((h['loss'] for h in val_h), default=float('nan'))
    best_val_acc = max((h['vit_acc'] for h in val_h), default=float('nan'))
    lines = [
        '=' * 64, f'  TRAINING SUMMARY — {protocol.upper()} (v2: {IMG_SIZE}px)', '=' * 64, '',
        f'  Description:  {PROTOCOLS[protocol]["description"]}',
        f'  Resolution:   {IMG_SIZE}x{IMG_SIZE} (v2)',
        f'  Total time:   {total_time_s/60:.1f} minutes',
        f'  Epochs run:   {len(train_h)}  (max={NUM_EPOCHS})',
        f'  Best epoch:   {best_epoch}', '',
        '  v2 HYPERPARAMETERS',
        f'  Resolution={IMG_SIZE}  LR={LEARNING_RATE}  WD={WEIGHT_DECAY}  BS={BATCH_SIZE}',
        f'  Focal gamma={FOCAL_GAMMA}  Aux weight={AUX_LOSS_WEIGHT}  Label smoothing={LABEL_SMOOTHING}',
        f'  Early stop patience={EARLY_STOP_PAT}  Warmup epochs={WARMUP_EPOCHS}',
        f'  Augmentation: RandAugment(n=2,m=9) + RandomErasing(p=0.25)', '',
        f'  Best val loss: {best_val_loss:.4f}  Best val acc: {best_val_acc:.4f}',
        f'  Final train acc: {train_h[-1]["vit_acc"]:.4f}' if train_h else '',
        '', '=' * 64,
    ]
    report = '\n'.join(lines)
    path = os.path.join(out_dir, 'training_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

def train_protocol(protocol, device):
    set_seed(RANDOM_SEED)
    cfg = PROTOCOLS[protocol]
    out_dir = os.path.join(MODELS_DIR, protocol)
    os.makedirs(out_dir, exist_ok=True)

    print('\n' + '=' * 64)
    print(f'  TRAINING v2: {protocol.upper()} ({IMG_SIZE}px, {NUM_EPOCHS} epochs)')
    print(f'  {cfg["description"]}')
    print('=' * 64)

    print('\n  [1/6] Loading datasets...')
    full_train = BrainTumorDataset(cfg['train_dir'], transform=get_train_transform())
    if cfg['has_val_dir']:
        train_ds = full_train
        val_ds = BrainTumorDataset(cfg['val_dir'], transform=get_eval_transform())
    else:
        train_ds, val_ds = split_train_val(full_train)
    test_ds = BrainTumorDataset(cfg['test_dir'], transform=get_eval_transform())
    test_present = list(test_ds.class_counts.keys())

    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')
    for cls in CLASS_NAMES:
        print(f'    {cls:<14} Train:{train_ds.class_counts.get(cls,0):>6}  Val:{val_ds.class_counts.get(cls,0):>5}  Test:{test_ds.class_counts.get(cls,0):>5}')

    print('\n  [2/6] Building DataLoaders...')
    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print('\n  [3/6] Focal Loss (gamma={}, smoothing={})...'.format(FOCAL_GAMMA, LABEL_SMOOTHING))
    train_counts = {cls: train_ds.class_counts.get(cls, 1) for cls in CLASS_NAMES}
    class_weights = compute_class_weights(train_counts)
    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights, label_smoothing=LABEL_SMOOTHING)

    print(f'\n  [4/6] Building model (img_size={IMG_SIZE})...')
    model = build_model(num_classes=NUM_CLASSES, pretrained=True, img_size=IMG_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spatial = IMG_SIZE // 32
    print(f'  Parameters: {n_params:,}')
    print(f'  Backbone spatial: {spatial}x{spatial} = {spatial**2} tokens')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    print(f'\n  [5/6] Training (max {NUM_EPOCHS} epochs, patience={EARLY_STOP_PAT}, warmup={WARMUP_EPOCHS})...\n')
    stopper = EarlyStopping(patience=EARLY_STOP_PAT)
    tracker = MetricsTracker()
    t0 = time.time()
    best_ep = 1
    best_path = os.path.join(out_dir, 'best_model.pth')

    for epoch in range(1, NUM_EPOCHS + 1):
        lr = scheduler.get_last_lr()[0]
        phase_str = "WARMUP" if epoch <= WARMUP_EPOCHS else "COSINE"
        print(f'  Epoch {epoch:>3}/{NUM_EPOCHS}  LR={lr:.2e}  [{phase_str}]')

        t_m = train_one_epoch(model, train_loader, optimizer, focal_loss, device)
        v_m = evaluate(model, val_loader, focal_loss, device, phase='val')
        tracker.add(epoch, 'train', t_m)
        tracker.add(epoch, 'val', v_m)
        scheduler.step()

        print(f'    Train loss={t_m["loss"]:.4f} vit_acc={t_m["vit_acc"]:.4f}')
        print(f'    Val   loss={v_m["loss"]:.4f} vit_acc={v_m["vit_acc"]:.4f}')

        improved = stopper(v_m['loss'], epoch)
        if improved:
            best_ep = epoch
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': v_m['loss'], 'val_acc': v_m['vit_acc'],
                'protocol': protocol, 'img_size': IMG_SIZE,
                'class_weights': class_weights.tolist(),
                'test_classes': test_present,
            }, best_path)
            print(f'    >>> Best model saved (val_loss={v_m["loss"]:.4f})')

        if stopper.stop:
            print(f'\n  Early stopping at epoch {epoch} (best={best_ep}).')
            break
        print()

    total_time = time.time() - t0
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'protocol': protocol}, os.path.join(out_dir, 'final_model.pth'))

    print('\n  [6/6] Saving logs and plots...')
    tracker.save_csv(os.path.join(out_dir, 'training_log.csv'))
    plot_training_curves(tracker, out_dir, protocol)
    save_training_summary(protocol, tracker, best_ep, total_time, train_ds.class_counts, val_ds.class_counts, out_dir, test_present)
    print(f'\n  {protocol.upper()} complete. Best model: {best_path}')
    print(f'  Time: {total_time/60:.1f} min')
    return best_path

def main():
    parser = argparse.ArgumentParser(description='Module 3 v2 — Train DualHeadHybridCNNViT')
    parser.add_argument('--protocol', type=str, default='all', choices=['all', 'merged', 'lodo1', 'lodo2', 'lodo3'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    args = parser.parse_args()

    _mod = sys.modules[__name__]
    _mod.BATCH_SIZE = args.batch_size
    _mod.NUM_EPOCHS = args.epochs
    _mod.IMG_SIZE = args.img_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU:    {torch.cuda.get_device_name(0)}')
        print(f'  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  Resolution: {args.img_size}px | Epochs: {args.epochs} | BS: {args.batch_size}')

    protocols = ['merged', 'lodo1', 'lodo2', 'lodo3'] if args.protocol == 'all' else [args.protocol]

    print(f'\n  Verifying data directories...')
    for p in protocols:
        cfg = PROTOCOLS[p]
        for key in ['train_dir', 'test_dir']:
            if not os.path.isdir(cfg[key]):
                print(f'  [ERROR] Missing: {cfg[key]}')
                sys.exit(1)
    print('  All directories verified.')

    results = {}
    t_start = time.time()
    for protocol in protocols:
        t_proto = time.time()
        try:
            best = train_protocol(protocol, device)
            results[protocol] = {'status': 'complete', 'minutes': (time.time()-t_proto)/60}
        except Exception as e:
            import traceback; traceback.print_exc()
            results[protocol] = {'status': 'failed', 'error': str(e), 'minutes': (time.time()-t_proto)/60}

    total_min = (time.time() - t_start) / 60
    print('\n' + '=' * 64)
    print(f'  ALL TRAINING COMPLETE — v2 ({IMG_SIZE}px)')
    print('=' * 64)
    for p, r in results.items():
        sym = 'OK' if r['status'] == 'complete' else 'FAIL'
        print(f'  [{sym}] {p.upper():<12} {r["minutes"]:.1f} min')
    print(f'  TOTAL: {total_min:.1f} min ({total_min/60:.1f} hours)')
    print('=' * 64)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'training_timing_report.txt'), 'w') as f:
        f.write(f'v2 Training ({IMG_SIZE}px) — {total_min:.1f} min total\n')
        for p, r in results.items():
            f.write(f'  {p}: {r["status"]} ({r["minutes"]:.1f} min)\n')
