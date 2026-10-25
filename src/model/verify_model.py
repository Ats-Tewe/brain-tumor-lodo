
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module2_model.model import (
    build_model,
    FocalLoss,
    compute_class_weights,
)

BATCH_SIZE   = 2
NUM_CLASSES  = 4
IMG_SIZE     = 224

ROUND1_COUNTS = {
    'glioma':      2198,
    'meningioma':  1007,
    'pituitary':    899,
    'notumor':      564,
}

def check(label: str, condition: bool, detail: str = '') -> bool:
    status = '[PASS]' if condition else '[FAIL]'
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  |  {detail}"
    print(msg)
    return condition

def main():
    print()
    print('=' * 70)
    print('  MODULE 2 — Dual-Head Hybrid CNN-ViT  Verification')
    print('=' * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n  Device : {device.upper()}')
    if device == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print()

    all_passed = True

    print('  [CHECK 1] Building model...')
    try:
        model = build_model(num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(device)
        model.eval()
        all_passed &= check('Model builds successfully', True)
    except Exception as e:
        check('Model builds successfully', False, str(e))
        print('\n  CRITICAL: Cannot continue.')
        return

    print()
    print('  [CHECK 2] Parameter counts:')
    counts = model.count_parameters()
    for name, n in counts.items():
        print(f"    {name:<40} {n:>12,}")
    total = counts['total_trainable']
    ok = check('Total params in range 15M-40M', 15_000_000 <= total <= 40_000_000,
               f'Got {total:,}')
    all_passed &= ok

    print()
    print('  [CHECK 3] Forward pass — output shapes:')
    dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        out = model(dummy, return_attention=False)

    shapes = [
        ('vit_logits  [B, 4]',       out['vit_logits'].shape  == (BATCH_SIZE, NUM_CLASSES)),
        ('cnn_logits  [B, 4]',       out['cnn_logits'].shape  == (BATCH_SIZE, NUM_CLASSES)),
        ('feature_map [B, 1280,7,7]',out['feature_map'].shape == (BATCH_SIZE, 1280, 7, 7)),
    ]
    for label, cond in shapes:
        all_passed &= check(label, cond)

    print()
    print('  [CHECK 4] Attention weights (12 ViT layers):')
    with torch.no_grad():
        out_attn = model(dummy, return_attention=True)
    attn_list = out_attn.get('attention_list', [])

    all_passed &= check('12 attention tensors returned', len(attn_list) == 12,
                        f'Got {len(attn_list)}')
    if attn_list:
        exp = (BATCH_SIZE, 6, 50, 50)
        all_passed &= check('Shape per layer [B, 6, 50, 50]',
                            attn_list[0].shape == exp,
                            f'Got {tuple(attn_list[0].shape)}')
        sums = attn_list[0].sum(dim=-1)
        dev = (sums - 1.0).abs().max().item()
        all_passed &= check('Attention rows sum to 1.0 (softmax valid)',
                            dev < 1e-4, f'Max deviation: {dev:.2e}')

    print()
    print('  [CHECK 5] MC Dropout for uncertainty estimation:')

    model.eval()
    with torch.no_grad():
        eval_preds = [model(dummy)['vit_logits'] for _ in range(5)]
    identical = all(torch.allclose(eval_preds[0], eval_preds[i]) for i in range(1, 5))
    all_passed &= check('Eval mode: 5 passes identical (dropout OFF)', identical)

    model.enable_mc_dropout()
    with torch.no_grad():
        mc10 = [model(dummy)['vit_logits'] for _ in range(10)]
    varies = not all(torch.allclose(mc10[0], mc10[i]) for i in range(1, 10))
    all_passed &= check('MC mode: passes differ (dropout ON)', varies)

    model.enable_mc_dropout()
    with torch.no_grad():
        probs50 = torch.stack([
            model(dummy)['vit_logits'].softmax(-1) for _ in range(50)
        ])
    mean_p = probs50.mean(0)
    unc    = probs50.std(0)

    all_passed &= check('50-pass mean shape [B, 4]',
                        mean_p.shape == (BATCH_SIZE, NUM_CLASSES))
    all_passed &= check('50-pass uncertainty shape [B, 4]',
                        unc.shape == (BATCH_SIZE, NUM_CLASSES))
    dev = (mean_p.sum(-1) - 1.0).abs().max().item()
    all_passed &= check('Mean probs sum to ~1.0', dev < 1e-3,
                        f'Max deviation: {dev:.2e}')

    print()
    print('  [CHECK 6] Focal Loss:')
    model.eval()

    weights = compute_class_weights(ROUND1_COUNTS)
    all_passed &= check('4 class weights computed', len(weights) == 4,
                        str([f'{w:.3f}' for w in weights.tolist()]))
    all_passed &= check('NoTumor has highest weight',
                        weights[3] == weights.max())

    focal = FocalLoss(gamma=2.0, alpha=weights, reduction='mean')

    rand_logits  = torch.randn(BATCH_SIZE, NUM_CLASSES).to(device)
    rand_targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device)

    lv = focal(rand_logits, rand_targets)
    all_passed &= check('Focal Loss scalar, not NaN',
                        lv.item() > 0 and not torch.isnan(lv),
                        f'value={lv.item():.4f}')

    print()
    print('  [CHECK 7] Total loss = L_ViT + 0.3 * L_CNN:')

    with torch.no_grad():
        out7 = model(dummy, return_attention=False)

    targets7 = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device)

    l_vit  = focal(out7['vit_logits'], targets7)
    l_cnn  = focal(out7['cnn_logits'], targets7)
    l_tot  = l_vit + 0.3 * l_cnn

    all_passed &= check(
        'Total loss computes correctly',
        l_tot.item() > 0 and not torch.isnan(l_tot),
        f'L_ViT={l_vit.item():.4f}  L_CNN={l_cnn.item():.4f}  Total={l_tot.item():.4f}'
    )

    print()
    print('  [CHECK 8] Branch independence:')
    differ = not torch.allclose(out7['vit_logits'], out7['cnn_logits'], atol=1e-3)
    all_passed &= check('CNN and ViT logits differ', differ,
                        'Confirms independent reasoning pathways')

    print()
    print('  [CHECK 9] Grad-CAM purity (gradient stays in Branch A):')

    model.zero_grad()
    x_grad = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    out9   = model(x_grad, return_attention=False)

    out9['cnn_logits'][0, 0].backward(retain_graph=True)

    vit_grad = model.vit_primary_fc.weight.grad
    vit_clean = (vit_grad is None) or torch.all(vit_grad == 0).item()
    all_passed &= check(
        'ViT primary head: NO gradient from CNN aux backward',
        vit_clean,
        'Grad-CAM is architecturally pure — stays in Branch A'
    )

    cnn_grad = model.cnn_aux_fc.weight.grad
    all_passed &= check(
        'CNN auxiliary head: gradient IS present',
        cnn_grad is not None and not torch.all(cnn_grad == 0).item(),
        'Grad-CAM gradient path active in Branch A'
    )

    print()
    print('=' * 70)
    if all_passed:
        print('  ALL CHECKS PASSED')
        print()
        print('  Module 2 is verified and ready.')
        print('  Next: Module 3 — Training Loop')
        print('        (Focal Loss + WeightedRandomSampler + Online Augmentation)')
    else:
        print('  ONE OR MORE CHECKS FAILED')
        print('  Review [FAIL] lines above before proceeding.')
    print('=' * 70)
    print()
