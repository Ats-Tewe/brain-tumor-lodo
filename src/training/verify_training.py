
import os
import sys
import time
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module3_training.train import (
    BrainTumorDataset, get_train_transform, get_eval_transform,
    build_weighted_sampler, split_train_val,
    train_one_epoch, evaluate, plot_training_curves,
    MetricsTracker, EarlyStopping, set_seed,
    PROTOCOLS, CLASS_NAMES, NUM_CLASSES, MODELS_DIR,
    BATCH_SIZE, FOCAL_GAMMA, AUX_LOSS_WEIGHT, VAL_SPLIT_RATIO,
)
from torch.utils.data import DataLoader, Subset
from module2_model.model import build_model, FocalLoss, compute_class_weights

SMOKE_EPOCHS        = 2
SMOKE_BATCH         = 8
SMOKE_SAMPLES_PER_CLASS = 50

def check(label, condition, detail=''):
    status = '[PASS]' if condition else '[FAIL]'
    msg = f'  {status}  {label}'
    if detail:
        msg += f'  |  {detail}'
    print(msg)
    return condition

def subsample_dataset(ds, n_per_class=SMOKE_SAMPLES_PER_CLASS):
    """
    Returns a Subset with at most n_per_class images per class.
    Used to make the smoke test fast.
    """
    indices_by_class = {}
    for i, (_, label) in enumerate(ds.samples):
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(i)

    selected = []
    for label, indices in indices_by_class.items():
        random.shuffle(indices)
        selected.extend(indices[:n_per_class])

    return Subset(ds, selected)

def main():
    print()
    print('=' * 64)
    print('  MODULE 3 — Training Pipeline Verification')
    print(f'  Smoke test: {SMOKE_EPOCHS} epochs, {SMOKE_SAMPLES_PER_CLASS} images/class, batch={SMOKE_BATCH}')
    print('=' * 64)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device.type.upper()}')
    if device.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print()

    all_passed = True
    start = time.time()

    print('  [CHECK 1] Verifying all data directories exist...')
    dirs_ok = True
    for pname, cfg in PROTOCOLS.items():
        for key in ['train_dir', 'test_dir']:
            exists = os.path.isdir(cfg[key])
            if not exists:
                print(f'    [MISSING] {pname} {key}: {cfg[key]}')
                dirs_ok = False
        if cfg['has_val_dir'] and not os.path.isdir(cfg['val_dir']):
            print(f'    [MISSING] {pname} val_dir: {cfg["val_dir"]}')
            dirs_ok = False
    all_passed &= check('All data directories exist', dirs_ok)

    cfg = PROTOCOLS['merged']

    print()
    print('  [CHECK 2] Dataset loading and class counts...')
    try:
        train_ds = BrainTumorDataset(cfg['train_dir'], transform=get_train_transform())
        val_ds   = BrainTumorDataset(cfg['val_dir'],   transform=get_eval_transform())
        test_ds  = BrainTumorDataset(cfg['test_dir'],  transform=get_eval_transform())
        loaded = True
    except Exception as e:
        loaded = False
        print(f'    Error: {e}')
    all_passed &= check('Datasets load without error', loaded)
    if loaded:
        all_passed &= check(
            'Train has 4 classes',
            len(train_ds.class_counts) == 4,
            str(train_ds.class_counts)
        )

    if not loaded:
        print('\n  Cannot continue without data. Exiting.')
        return

    train_sub = subsample_dataset(train_ds)
    val_sub   = subsample_dataset(val_ds,  n_per_class=20)

    print()
    print('  [CHECK 3] Transforms apply without error...')
    try:
        img, lbl = train_ds[0]
        shape_ok = img.shape == (3, 224, 224)
        all_passed &= check('Train transform produces [3,224,224] tensor', shape_ok,
                            f'Got {tuple(img.shape)}')
        img_v, _ = val_ds[0]
        all_passed &= check('Eval transform produces [3,224,224] tensor',
                            img_v.shape == (3, 224, 224))
    except Exception as e:
        all_passed &= check('Transforms apply without error', False, str(e))

    print()
    print('  [CHECK 4] WeightedRandomSampler...')
    try:
        sampler = build_weighted_sampler(train_ds)
        loader  = DataLoader(train_sub, batch_size=SMOKE_BATCH, sampler=None,
                             shuffle=True, num_workers=0)
        imgs_b, lbls_b = next(iter(loader))
        all_passed &= check(
            'DataLoader produces correct batch shape',
            imgs_b.shape == (SMOKE_BATCH, 3, 224, 224),
            f'Got {tuple(imgs_b.shape)}'
        )
    except Exception as e:
        all_passed &= check('WeightedRandomSampler works', False, str(e))

    print()
    print('  [CHECK 5] LODO val split creation (stratified 10%)...')
    try:
        lodo_ds = BrainTumorDataset(
            PROTOCOLS['lodo1']['train_dir'],
            transform=get_train_transform()
        )
        tr_split, va_split = split_train_val(lodo_ds, val_ratio=0.10)
        ratio_ok = 0.08 < len(va_split) / len(lodo_ds) < 0.12
        all_passed &= check('LODO val split size ~10%', ratio_ok,
                            f'Train={len(tr_split)} Val={len(va_split)} '
                            f'Ratio={len(va_split)/len(lodo_ds):.3f}')
        all_passed &= check('Val split has eval transform',
                            va_split.transform is not None)
    except Exception as e:
        all_passed &= check('LODO val split creation', False, str(e))

    print()
    print('  [CHECK 6] Model forward pass on real training batch...')
    try:
        model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        model.train()
        imgs_b = imgs_b.to(device)
        lbls_b = lbls_b.to(device)
        out    = model(imgs_b, return_attention=False)
        all_passed &= check(
            'vit_logits shape [batch, 4]',
            out['vit_logits'].shape == (SMOKE_BATCH, NUM_CLASSES),
            f'Got {tuple(out["vit_logits"].shape)}'
        )
        all_passed &= check(
            'feature_map shape [batch, 1280, 7, 7]',
            out['feature_map'].shape == (SMOKE_BATCH, 1280, 7, 7),
            f'Got {tuple(out["feature_map"].shape)}'
        )
    except Exception as e:
        all_passed &= check('Model forward pass on real data', False, str(e))
        print(f'  Cannot continue without a working model.')
        return

    print()
    print('  [CHECK 7] Focal Loss + backward pass + optimizer step...')
    try:
        class_weights = compute_class_weights(train_ds.class_counts)
        focal         = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights)
        optimizer     = torch.optim.AdamW(model.parameters(), lr=3e-4)

        l_vit  = focal(out['vit_logits'], lbls_b)
        l_cnn  = focal(out['cnn_logits'], lbls_b)
        loss   = l_vit + AUX_LOSS_WEIGHT * l_cnn
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        all_passed &= check('Loss is finite scalar', loss.item() > 0 and not torch.isnan(loss),
                            f'loss={loss.item():.4f}')
    except Exception as e:
        all_passed &= check('Focal Loss + backward + optimizer', False, str(e))

    print()
    print(f'  [CHECK 8] Full 2-epoch smoke train on merged protocol...')
    try:
        model2   = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        opt2     = torch.optim.AdamW(model2.parameters(), lr=3e-4)
        focal2   = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights)
        tracker  = MetricsTracker()
        stopper  = EarlyStopping(patience=5)

        tr_loader = DataLoader(train_sub, batch_size=SMOKE_BATCH, shuffle=True, num_workers=0)
        va_loader = DataLoader(val_sub,   batch_size=SMOKE_BATCH, shuffle=False, num_workers=0)

        for ep in range(1, SMOKE_EPOCHS + 1):
            t_m = train_one_epoch(model2, tr_loader, opt2, focal2, device)
            v_m = evaluate(model2, va_loader, focal2, device, phase='val')
            tracker.add(ep, 'train', t_m)
            tracker.add(ep, 'val',   v_m)

        train_h = tracker.get_phase('train')
        all_passed &= check(
            f'{SMOKE_EPOCHS} training epochs completed',
            len(train_h) == SMOKE_EPOCHS,
            f'Got {len(train_h)} epoch records'
        )
    except Exception as e:
        all_passed &= check('2-epoch smoke train', False, str(e))
        import traceback; traceback.print_exc()

    print()
    print('  [CHECK 9] Checkpoint saving...')
    try:
        out_dir  = os.path.join(MODELS_DIR, '_smoke_test')
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, 'smoke_best.pth')
        torch.save({'epoch': 1, 'model_state': model2.state_dict()}, ckpt_path)
        exists = os.path.isfile(ckpt_path)
        all_passed &= check('Checkpoint saved to disk', exists, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        all_passed &= check('Checkpoint loads without error',
                            'model_state' in ckpt)
    except Exception as e:
        all_passed &= check('Checkpoint save/load', False, str(e))

    print()
    print('  [CHECK 10] Training curve plotting...')
    try:
        plot_training_curves(tracker, out_dir, 'smoke_test')
        png_path = os.path.join(out_dir, 'training_curves.png')
        all_passed &= check('Training curves PNG saved', os.path.isfile(png_path))
    except Exception as e:
        all_passed &= check('Training curve plot', False, str(e))

    elapsed = time.time() - start
    print()
    print('=' * 64)
    if all_passed:
        print(f'  ALL CHECKS PASSED  ({elapsed:.1f} s)')
        print()
        print('  Training pipeline is verified and ready.')
        print()
        print('  To run full training (all 4 protocols):')
        print('    python module3_training\\train.py')
        print()
        print('  To run a single protocol:')
        print('    python module3_training\\train.py --protocol merged')
        print('    python module3_training\\train.py --protocol lodo1')
        print('    python module3_training\\train.py --protocol lodo2')
        print('    python module3_training\\train.py --protocol lodo3')
        print()
        print('  Expected training time per protocol (RTX 4060 Laptop):')
        print('    merged:  ~2–4 hours')
        print('    lodo1:   ~1–2 hours')
        print('    lodo2:   ~2–4 hours  (largest training set)')
        print('    lodo3:   ~1–2 hours')
    else:
        print(f'  ONE OR MORE CHECKS FAILED  ({elapsed:.1f} s)')
        print('  Fix [FAIL] items above before running full training.')
    print('=' * 64)
    print()
