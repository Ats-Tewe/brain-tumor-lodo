
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module5_explainability.explain import (
    GradCAM, AttentionRollout, cosine_similarity_maps,
    render_overlay, save_combined_figure, load_image_tensor,
    PROTOCOLS, EXPLANATIONS_DIR, CLASS_NAMES, CLASS_TO_IDX,
)
from module3_training.train import BrainTumorDataset, get_eval_transform
from module2_model.model import build_model
from PIL import Image
import cv2

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
    print('  MODULE 5 — Explainability Pipeline Verification')
    print('=' * 68)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device.type.upper()}')
    if device.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print()

    all_passed = True
    out_dir    = os.path.join(EXPLANATIONS_DIR, '_verify')
    os.makedirs(out_dir, exist_ok=True)

    print('  [CHECK 1] Checkpoint loading...')
    ckpt_path = PROTOCOLS['merged']['checkpoint']
    try:
        ckpt  = torch.load(ckpt_path, map_location=device)
        model = build_model(num_classes=4, pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        all_passed &= check('Checkpoint loads and model in eval mode', True,
                            f'epoch={ckpt["epoch"]}')
    except Exception as e:
        all_passed &= check('Checkpoint loads', False, str(e))
        print('\n  CRITICAL: Cannot continue without model.')
        return

    test_dir = PROTOCOLS['merged']['test_dir']
    test_images = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if files:
            test_images[cls] = os.path.join(cls_dir, files[0])

    if not test_images:
        print('  CRITICAL: No test images found.')
        return

    test_cls  = list(test_images.keys())[0]
    test_path = test_images[test_cls]
    tensor, original = load_image_tensor(test_path)

    print()
    print('  [CHECK 2] GradCAM hook registration...')
    try:
        gradcam = GradCAM(model)
        n_hooks = len(gradcam._hooks)
        all_passed &= check('GradCAM hooks registered', n_hooks == 2,
                            f'{n_hooks} hooks registered')
    except Exception as e:
        all_passed &= check('GradCAM hooks', False, str(e))
        return

    print()
    print('  [CHECK 3+4] Grad-CAM heatmap generation...')
    try:
        gc_map, gc_pred, gc_conf = gradcam.generate(
            tensor, device, target_class=CLASS_TO_IDX[test_cls]
        )
        all_passed &= check('Grad-CAM shape [224,224]',
                            gc_map.shape == (224, 224),
                            f'Got {gc_map.shape}')
        all_passed &= check('Grad-CAM values in [0,1]',
                            gc_map.min() >= 0.0 and gc_map.max() <= 1.0,
                            f'Min={gc_map.min():.4f}  Max={gc_map.max():.4f}')
    except Exception as e:
        all_passed &= check('Grad-CAM generation', False, str(e))
        gc_map = np.zeros((224, 224), dtype=np.float32)
        gc_pred, gc_conf = 0, 0.0

    print()
    print('  [CHECK 5] Grad-CAM purity — ViT head receives zero gradient...')
    try:
        tensor_grad = tensor.to(device)
        model.zero_grad()
        out = model(tensor_grad, return_attention=False)
        out['cnn_logits'][0, gc_pred].backward()

        vit_fc_grad = model.vit_primary_fc.weight.grad
        if vit_fc_grad is None:
            vit_grad_norm = 0.0
        else:
            vit_grad_norm = float(vit_fc_grad.abs().max().item())

        all_passed &= check('ViT primary FC gradient == 0 (Grad-CAM purity)',
                            vit_grad_norm < 1e-10,
                            f'Max ViT grad: {vit_grad_norm:.2e}')
        model.zero_grad()
    except Exception as e:
        all_passed &= check('Grad-CAM purity check', False, str(e))

    print()
    print('  [CHECK 6-8] Attention Rollout generation...')
    try:
        rollout = AttentionRollout(model)

        with torch.no_grad():
            out_attn = model(tensor.to(device), return_attention=True)
        n_attn = len(out_attn.get('attention_list', []))
        all_passed &= check('Attention Rollout collects 12 attention tensors',
                            n_attn == 12,
                            f'Got {n_attn} tensors')

        ar_map, ar_pred, ar_conf = rollout.generate(tensor, device)
        all_passed &= check('Attention Rollout shape [224,224]',
                            ar_map.shape == (224, 224),
                            f'Got {ar_map.shape}')
        all_passed &= check('Attention Rollout values in [0,1]',
                            ar_map.min() >= 0.0 and ar_map.max() <= 1.0,
                            f'Min={ar_map.min():.4f}  Max={ar_map.max():.4f}')
    except Exception as e:
        all_passed &= check('Attention Rollout generation', False, str(e))
        ar_map = np.zeros((224, 224), dtype=np.float32)
        ar_pred, ar_conf = 0, 0.0

    print()
    print('  [CHECK 9] Cosine similarity computation...')
    cos_sim = cosine_similarity_maps(gc_map, ar_map)
    all_passed &= check('Cosine similarity in [0,1]',
                        0.0 <= cos_sim <= 1.0,
                        f'{cos_sim:.4f}')

    print()
    print('  [CHECK 10] Overlay rendering...')
    try:
        overlay = render_overlay(original, gc_map, cv2.COLORMAP_JET, alpha=0.5)
        all_passed &= check('Overlay shape [224,224,3]',
                            overlay.shape == (224, 224, 3),
                            f'Got {overlay.shape}')
        all_passed &= check('Overlay dtype uint8',
                            overlay.dtype == np.uint8,
                            f'Got {overlay.dtype}')
    except Exception as e:
        all_passed &= check('Overlay rendering', False, str(e))

    print()
    print('  [CHECK 11] Combined figure save...')
    combined_path = os.path.join(out_dir, 'verify_combined.png')
    try:
        save_combined_figure(
            original   = original,
            gradcam_map= gc_map,
            attn_map   = ar_map,
            out_path   = combined_path,
            title      = 'Verify — Combined Figure',
            gc_pred    = CLASS_NAMES[gc_pred],
            gc_conf    = gc_conf,
            ar_pred    = CLASS_NAMES[ar_pred],
            ar_conf    = ar_conf,
            true_class = test_cls,
            cosine_sim = cos_sim,
        )
        all_passed &= check('Combined figure PNG saved',
                            os.path.isfile(combined_path),
                            combined_path)
    except Exception as e:
        all_passed &= check('Combined figure save', False, str(e))

    print()
    print('  [CHECK 12] Text report save...')
    report_path = os.path.join(out_dir, 'verify_report.txt')
    try:
        with open(report_path, 'w') as f:
            f.write(f'Verify: cosine_sim={cos_sim:.4f}\n')
        all_passed &= check('Text report saves',
                            os.path.isfile(report_path),
                            report_path)
    except Exception as e:
        all_passed &= check('Text report save', False, str(e))

    print()
    print('  [CHECK 13] GradCAM hook cleanup...')
    try:
        gradcam.remove_hooks()
        all_passed &= check('Hooks removed cleanly',
                            len(gradcam._hooks) == 0,
                            f'{len(gradcam._hooks)} hooks remaining')
    except Exception as e:
        all_passed &= check('Hook cleanup', False, str(e))

    print()
    print('=' * 68)
    if all_passed:
        print('  ALL CHECKS PASSED')
        print()
        print('  Explainability pipeline is verified and ready.')
        print()
        print('  To run full explainability (all 4 protocols):')
        print('    python module5_explainability\\explain.py')
        print()
        print('  To run a single protocol:')
        print('    python module5_explainability\\explain.py --protocol merged')
        print()
        print('  To generate more samples per class:')
        print('    python module5_explainability\\explain.py --n_samples 10')
    else:
        print('  ONE OR MORE CHECKS FAILED')
        print('  Fix [FAIL] items before running full explainability.')
    print('=' * 68)
    print()
