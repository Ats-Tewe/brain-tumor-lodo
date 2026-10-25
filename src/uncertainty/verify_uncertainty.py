
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module6_uncertainty.uncertainty import (
    mc_dropout_inference, compute_uncertainty_metrics, compute_ece,
    plot_reliability_diagram, plot_uncertainty_distribution,
    PROTOCOLS, UNCERTAINTY_DIR, CLASS_NAMES, N_MC_PASSES,
)
from module3_training.train import BrainTumorDataset, get_eval_transform
from module2_model.model import build_model
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

def check(label, condition, detail=''):
    status = '[PASS]' if condition else '[FAIL]'
    msg = f'  {status}  {label}'
    if detail:
        msg += f'  |  {detail}'
    print(msg)
    return condition

def main():
    print()
    print('=' * 68)
    print('  MODULE 6 \u2014 Uncertainty Pipeline Verification')
    print('  (T=5 passes, 32-image subset for speed)')
    print('=' * 68)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device.type.upper()}')
    if device.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print()

    all_passed = True
    out_dir    = os.path.join(UNCERTAINTY_DIR, '_verify')
    os.makedirs(out_dir, exist_ok=True)

    print('  [CHECK 1] Checkpoint loading...')
    ckpt_path = PROTOCOLS['merged']['checkpoint']
    try:
        ckpt  = torch.load(ckpt_path, map_location=device)
        model = build_model(num_classes=4, pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state'])
        all_passed &= check('Checkpoint loads successfully',
                            True, f'epoch={ckpt["epoch"]}')
    except Exception as e:
        all_passed &= check('Checkpoint loads', False, str(e))
        print('\n  CRITICAL: Cannot continue without model.')
        return

    print()
    print('  [CHECK 2] MC Dropout mode activation...')
    model.enable_mc_dropout()
    mc_dropout_in_train = model.mc_dropout.training
    backbone_in_eval    = not model.backbone.training
    vit_in_eval         = not model.vit_blocks[0].training
    all_passed &= check('mc_dropout layer in TRAIN mode (stochastic)',
                        mc_dropout_in_train, str(mc_dropout_in_train))
    all_passed &= check('backbone in EVAL mode (frozen BN)',
                        backbone_in_eval, str(backbone_in_eval))
    all_passed &= check('ViT blocks in EVAL mode',
                        vit_in_eval, str(vit_in_eval))

    print()
    print('  [CHECK 3] Stochastic output (T passes differ)...')
    test_dir = PROTOCOLS['merged']['test_dir']
    test_ds  = BrainTumorDataset(test_dir, transform=get_eval_transform())
    sub_ds   = Subset(test_ds, list(range(min(4, len(test_ds)))))
    sub_loader = DataLoader(sub_ds, batch_size=4, shuffle=False, num_workers=0)

    probs_pass1, probs_pass2 = [], []
    model.enable_mc_dropout()
    with torch.no_grad():
        for images, _ in sub_loader:
            images = images.to(device)
            out1 = model(images, return_attention=False)
            out2 = model(images, return_attention=False)
            probs_pass1.append(F.softmax(out1['vit_logits'], dim=-1).cpu().numpy())
            probs_pass2.append(F.softmax(out2['vit_logits'], dim=-1).cpu().numpy())

    p1 = np.concatenate(probs_pass1)
    p2 = np.concatenate(probs_pass2)
    diff = np.abs(p1 - p2).max()
    all_passed &= check('Two MC passes give different outputs (stochastic)',
                        diff > 1e-6, f'Max diff: {diff:.6f}')

    print()
    print('  [CHECK 4] Deterministic in eval mode...')
    model.eval()
    with torch.no_grad():
        for images, _ in sub_loader:
            images = images.to(device)
            out1 = model(images, return_attention=False)
            out2 = model(images, return_attention=False)
            p1e  = F.softmax(out1['vit_logits'], dim=-1).cpu().numpy()
            p2e  = F.softmax(out2['vit_logits'], dim=-1).cpu().numpy()
            break
    diff_eval = np.abs(p1e - p2e).max()
    all_passed &= check('Two passes in eval mode are identical (deterministic)',
                        diff_eval < 1e-6, f'Max diff: {diff_eval:.2e}')

    print()
    print('  [CHECK 5-8] MC Dropout inference (T=5 passes, 32 images)...')
    full_sub = Subset(test_ds, list(range(min(32, len(test_ds)))))
    full_loader = DataLoader(full_sub, batch_size=8, shuffle=False, num_workers=0)

    T = 5
    try:
        mc_results = mc_dropout_inference(model, full_loader, device, n_passes=T)
        N = len(mc_results['true_labels'])
        all_passed &= check('MC inference completes without error', True, f'N={N}')

        mp = mc_results['mean_probs']
        sp = mc_results['std_probs']
        ap = mc_results['all_passes']

        all_passed &= check('mean_probs shape [N,4]',
                            mp.shape == (N, 4), f'Got {mp.shape}')
        all_passed &= check('mean_probs rows sum to ~1.0',
                            np.allclose(mp.sum(axis=1), 1.0, atol=1e-5),
                            f'Max deviation: {np.abs(mp.sum(axis=1)-1).max():.2e}')
        all_passed &= check('mean_probs in [0,1]',
                            mp.min() >= 0 and mp.max() <= 1,
                            f'Min={mp.min():.4f}  Max={mp.max():.4f}')
        all_passed &= check('std_probs shape [N,4]',
                            sp.shape == (N, 4), f'Got {sp.shape}')
        all_passed &= check('std_probs >= 0',
                            sp.min() >= 0, f'Min={sp.min():.6f}')
        all_passed &= check('all_passes shape [T,N,4]',
                            ap.shape == (T, N, 4), f'Got {ap.shape}')

    except Exception as e:
        all_passed &= check('MC inference', False, str(e))
        mc_results = None

    print()
    print('  [CHECK 9-10] Uncertainty metrics...')
    if mc_results is not None:
        try:
            present = ['glioma', 'meningioma', 'pituitary', 'notumor']
            metrics = compute_uncertainty_metrics(mc_results, present)
            all_passed &= check('compute_uncertainty_metrics runs',
                                True, f'n_records={len(metrics["records"])}')
            ece = metrics['ece']
            all_passed &= check('ECE in [0,1]',
                                0.0 <= ece <= 1.0, f'ECE={ece:.4f}')
        except Exception as e:
            all_passed &= check('Uncertainty metrics', False, str(e))
            metrics = None
    else:
        metrics = None

    print()
    print('  [CHECK 11] Correct predictions have lower uncertainty...')
    if metrics is not None:
        correct_std   = metrics.get('correct_mean_std')
        incorrect_std = metrics.get('incorrect_mean_std')
        if correct_std is not None and incorrect_std is not None:
            all_passed &= check('Correct std < Incorrect std',
                                correct_std < incorrect_std,
                                f'Correct={correct_std:.4f}  Incorrect={incorrect_std:.4f}')
        else:
            all_passed &= check('Correct/incorrect comparison (skipped: all correct)',
                                True, 'All predictions correct on tiny subset')

    print()
    print('  [CHECK 12] Reliability diagram PNG...')
    if mc_results is not None:
        rdiag_path = os.path.join(out_dir, 'verify_reliability.png')
        try:
            pred_classes = mc_results['mean_probs'].argmax(axis=1)
            plot_reliability_diagram(
                mc_results['mean_probs'],
                pred_classes,
                mc_results['true_labels'],
                rdiag_path,
                'Verify \u2014 Reliability Diagram',
                ece=0.05,
            )
            all_passed &= check('Reliability diagram saves',
                                os.path.isfile(rdiag_path), rdiag_path)
        except Exception as e:
            all_passed &= check('Reliability diagram', False, str(e))

    print()
    print('  [CHECK 13] Uncertainty distribution PNG...')
    if metrics is not None:
        udist_path = os.path.join(out_dir, 'verify_uncertainty.png')
        try:
            plot_uncertainty_distribution(
                metrics,
                ['glioma', 'meningioma', 'pituitary', 'notumor'],
                udist_path,
                'Verify \u2014 Uncertainty Distribution',
            )
            all_passed &= check('Uncertainty distribution saves',
                                os.path.isfile(udist_path), udist_path)
        except Exception as e:
            all_passed &= check('Uncertainty distribution', False, str(e))

    print()
    print('=' * 68)
    if all_passed:
        print('  ALL CHECKS PASSED')
        print()
        print('  Uncertainty pipeline is verified and ready.')
        print()
        print('  Expected runtime (T=50 passes, full test sets):')
        print('    merged:  ~15-20 minutes  (1,237 images x 50 passes)')
        print('    lodo1:   ~20-25 minutes  (1,600 images x 50 passes)')
        print('    lodo2:   ~12-15 minutes  (1,068 images x 50 passes)')
        print('    lodo3:   ~10-12 minutes  (816 images x 50 passes)')
        print('    Total:   ~60-75 minutes')
        print()
        print('  To run full uncertainty estimation (all 4 protocols):')
        print('    python module6_uncertainty\\uncertainty.py')
        print()
        print('  To run a single protocol:')
        print('    python module6_uncertainty\\uncertainty.py --protocol merged')
    else:
        print('  ONE OR MORE CHECKS FAILED')
        print('  Fix [FAIL] items before running full uncertainty estimation.')
    print('=' * 68)
    print()
