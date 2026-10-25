
import os, sys, json, argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2_model.model import build_model, DualHeadHybridCNNViT
from module3_training.train import CLASS_NAMES, CLASS_TO_IDX, HARMONIZED_DIR, MODELS_DIR

PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLANATIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "explanations")
N_SAMPLES_PER_CLASS = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PROTOCOLS = {
    "merged": {"checkpoint": os.path.join(MODELS_DIR, "merged", "best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR, "merged_baseline", "test"),
               "description": "Merged Baseline"},
    "lodo1":  {"checkpoint": os.path.join(MODELS_DIR, "lodo1", "best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR, "lodo_round1", "test"),
               "description": "LODO Round 1 (Test: Kaggle)"},
    "lodo2":  {"checkpoint": os.path.join(MODELS_DIR, "lodo2", "best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR, "lodo_round2", "test"),
               "description": "LODO Round 2 (Test: Figshare)"},
    "lodo3":  {"checkpoint": os.path.join(MODELS_DIR, "lodo3", "best_model.pth"),
               "test_dir": os.path.join(HARMONIZED_DIR, "lodo_round3", "test"),
               "description": "LODO Round 3 (Test: Mendeley)"},
}

def make_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image_tensor(image_path, img_size):
    tf = make_eval_transform(img_size)
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    original = np.array(img_rgb, dtype=np.uint8)
    tensor = tf(img_rgb).unsqueeze(0)
    return tensor, original

def create_brain_mask(original_rgb, img_size):
    """
    Creates a binary brain mask from the original MRI image.
    Uses Otsu thresholding + morphological operations to isolate the brain.
    Returns a float32 mask [img_size, img_size] in [0, 1].
    """
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:
        return np.ones((img_size, img_size), dtype=np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    mask = (labels == largest_label).astype(np.uint8) * 255

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    mask = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
    mask = mask / max(mask.max(), 1e-8)

    return mask.astype(np.float32)

def apply_brain_mask(heatmap, brain_mask):
    """Multiplies heatmap by brain mask to zero out activation outside skull."""
    masked = heatmap * brain_mask
    mmin, mmax = masked.min(), masked.max()
    if mmax > mmin:
        masked = (masked - mmin) / (mmax - mmin)
    return masked.astype(np.float32)

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out): self.activations = out.detach()
        def bwd(module, grad_in, grad_out): self.gradients = grad_out[0].detach()
        h1 = self.model.backbone.register_forward_hook(fwd)
        h2 = self.model.backbone.register_full_backward_hook(bwd)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def generate(self, image_tensor, device, img_size, target_class=None):
        self.model.eval()
        image_tensor = image_tensor.to(device)
        output = self.model(image_tensor, return_attention=False)
        cnn_logits = output['cnn_logits']
        cnn_probs = F.softmax(cnn_logits, dim=-1)
        pred_class = cnn_logits.argmax(dim=1).item()
        if target_class is None: target_class = pred_class
        confidence = cnn_probs[0, pred_class].item()
        self.model.zero_grad()
        cnn_logits[0, target_class].backward()
        alpha = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = F.relu((alpha * self.activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=(img_size, img_size), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin: heatmap = (heatmap - hmin) / (hmax - hmin)
        else: heatmap = np.zeros_like(heatmap)
        return heatmap.astype(np.float32), pred_class, confidence

class AttentionRollout:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate(self, image_tensor, device, img_size):
        self.model.eval()
        image_tensor = image_tensor.to(device)
        output = self.model(image_tensor, return_attention=True)
        vit_logits = output['vit_logits']
        attention_list = output['attention_list']
        vit_probs = F.softmax(vit_logits, dim=-1)
        pred_class = vit_logits.argmax(dim=1).item()
        confidence = vit_probs[0, pred_class].item()

        n_tokens = attention_list[0].shape[-1]
        n_patches = n_tokens - 1
        spatial = int(np.sqrt(n_patches))

        rollout = torch.eye(n_tokens, device=device)
        for attn in attention_list:
            attn_avg = attn.squeeze(0).mean(dim=0)
            attn_hat = 0.5 * attn_avg + 0.5 * torch.eye(n_tokens, device=device)
            attn_hat = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
            rollout = attn_hat @ rollout

        cls_attention = rollout[0, 1:].cpu().numpy()
        attention_map = cls_attention.reshape(spatial, spatial)
        amin, amax = attention_map.min(), attention_map.max()
        if amax > amin: attention_map = (attention_map - amin) / (amax - amin)
        else: attention_map = np.zeros_like(attention_map)
        attention_map = cv2.resize(attention_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return attention_map.astype(np.float32), pred_class, confidence

def render_overlay(original_rgb, heatmap, colormap, alpha=0.5):
    heatmap_8u = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_8u, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = (1-alpha) * original_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def save_combined_figure(original, gradcam_map, attn_map, out_path, title,
                         gc_pred, gc_conf, ar_pred, ar_conf, true_class, cosine_sim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gc_overlay = render_overlay(original, gradcam_map, cv2.COLORMAP_JET)
    ar_overlay = render_overlay(original, attn_map, cv2.COLORMAP_HOT)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    panels = [
        (original, f'Original MRI\nTrue: {true_class}', 'gray'),
        (gc_overlay, f'Grad-CAM (CNN)\nPred: {gc_pred} ({gc_conf*100:.1f}%)', None),
        (ar_overlay, f'Attention Rollout (ViT)\nPred: {ar_pred} ({ar_conf*100:.1f}%)', None),
    ]
    for ax, (img, label, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')
    fig.text(0.5, -0.02, f'Cosine Similarity: {cosine_sim:.4f}', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def cosine_similarity_maps(gradcam_map, attn_map):
    gc = gradcam_map.flatten().astype(np.float64)
    at = attn_map.flatten().astype(np.float64)
    n1, n2 = np.linalg.norm(gc), np.linalg.norm(at)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(gc, at) / (n1 * n2))

def select_samples(test_dir, model, device, img_size, n_per_class=5, present_classes=None):
    classes = present_classes or CLASS_NAMES
    selected = {cls: [] for cls in classes}
    tf = make_eval_transform(img_size)
    model.eval()
    with torch.no_grad():
        for cls_name in classes:
            cls_dir = os.path.join(test_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            true_idx = CLASS_TO_IDX[cls_name]
            files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
            for fname in files:
                if len(selected[cls_name]) >= n_per_class: break
                path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(path) as img:
                        tensor = tf(img.convert("RGB")).unsqueeze(0).to(device)
                    out = model(tensor, return_attention=False)
                    if out['vit_logits'].argmax(1).item() == true_idx:
                        selected[cls_name].append({'path': path, 'true_idx': true_idx, 'fname': fname})
                except: continue
    for cls, samps in selected.items():
        print(f'  {cls:<14}: {len(samps)} samples')
    return selected

def explain_protocol(protocol, device, n_samples=N_SAMPLES_PER_CLASS):
    cfg = PROTOCOLS[protocol]
    out_dir = os.path.join(EXPLANATIONS_DIR, protocol)
    for sub in ['gradcam', 'attention', 'combined']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    print(f'\n{"="*68}\n  EXPLAINABILITY: {protocol.upper()} (v2.1: brain masking)\n{"="*68}')

    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    img_size = ckpt.get('img_size', 224)
    spatial = img_size // 32
    print(f'  img_size={img_size}px, spatial={spatial}x{spatial}')

    model = build_model(num_classes=4, pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    present_classes = [
        c for c in (ckpt.get('test_classes', CLASS_NAMES))
        if os.path.isdir(os.path.join(cfg['test_dir'], c))
        and len([f for f in os.listdir(os.path.join(cfg['test_dir'], c))
                if f.lower().endswith(('.jpg','.jpeg','.png'))]) > 0
    ]
    print(f'  Classes: {present_classes}')

    gradcam = GradCAM(model)
    rollout = AttentionRollout(model)

    print(f'  Selecting {n_samples} correct samples per class...')
    selected = select_samples(cfg['test_dir'], model, device, img_size,
                              n_per_class=n_samples, present_classes=present_classes)

    print('  Generating explanations (with brain masking)...')
    all_cosine, per_class_sims, records = [], {c: [] for c in present_classes}, []

    for cls_name in present_classes:
        samples = selected.get(cls_name, [])
        for i, sample in enumerate(tqdm(samples, desc=f'  {cls_name}', leave=False)):
            try:
                tensor, original = load_image_tensor(sample['path'], img_size)

                brain_mask = create_brain_mask(original, img_size)

                gc_map_raw, gc_pred, gc_conf = gradcam.generate(tensor, device, img_size,
                                                                  target_class=sample['true_idx'])
                ar_map_raw, ar_pred, ar_conf = rollout.generate(tensor, device, img_size)

                gc_map = apply_brain_mask(gc_map_raw, brain_mask)
                ar_map = apply_brain_mask(ar_map_raw, brain_mask)

                cos_sim = cosine_similarity_maps(gc_map, ar_map)
                all_cosine.append(cos_sim)
                per_class_sims[cls_name].append(cos_sim)

                gc_ov = render_overlay(original, gc_map, cv2.COLORMAP_JET)
                Image.fromarray(gc_ov).save(os.path.join(out_dir, 'gradcam', f'{cls_name}_{i+1:02d}_gradcam.png'))
                ar_ov = render_overlay(original, ar_map, cv2.COLORMAP_HOT)
                Image.fromarray(ar_ov).save(os.path.join(out_dir, 'attention', f'{cls_name}_{i+1:02d}_attention.png'))

                save_combined_figure(
                    original, gc_map, ar_map,
                    os.path.join(out_dir, 'combined', f'{cls_name}_{i+1:02d}_combined.png'),
                    f'{protocol.upper()} — {cls_name.capitalize()} #{i+1} (brain-masked)',
                    CLASS_NAMES[gc_pred], gc_conf, CLASS_NAMES[ar_pred], ar_conf,
                    cls_name, cos_sim,
                )
                records.append({
                    'class': cls_name, 'sample': i+1, 'file': sample['fname'],
                    'gc_pred': CLASS_NAMES[gc_pred], 'gc_conf': round(gc_conf,4),
                    'ar_pred': CLASS_NAMES[ar_pred], 'ar_conf': round(ar_conf,4),
                    'cosine_sim': round(cos_sim,4),
                })
            except Exception as e:
                print(f'  [WARN] {cls_name} #{i+1}: {e}')

    gradcam.remove_hooks()

    overall_mean = float(np.mean(all_cosine)) if all_cosine else 0.0
    overall_std = float(np.std(all_cosine)) if all_cosine else 0.0
    per_class_stats = {}
    for cls in present_classes:
        s = per_class_sims[cls]
        per_class_stats[cls] = {
            'mean': round(float(np.mean(s)),4) if s else 0.0,
            'std': round(float(np.std(s)),4) if s else 0.0,
            'n': len(s),
        }

    summary = {
        'protocol': protocol, 'img_size': img_size, 'brain_masking': True,
        'n_samples': len(records),
        'overall_cosine_mean': round(overall_mean, 4),
        'overall_cosine_std': round(overall_std, 4),
        'per_class': per_class_stats, 'records': records,
    }

    with open(os.path.join(out_dir, 'cosine_similarity_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [
        '='*68, f'  EXPLAINABILITY — {protocol.upper()} (v2.1: brain-masked, {img_size}px)', '='*68, '',
        f'  Spatial: {spatial}x{spatial} = {spatial**2} tokens',
        f'  Samples: {len(records)}  |  Brain masking: ON', '',
        '  COSINE SIMILARITY (Grad-CAM vs Attention Rollout)', '  '+'-'*50,
        f'  Overall: {overall_mean:.4f} +/- {overall_std:.4f}', '',
        f'  {"Class":<14} {"Mean":>8} {"Std":>8} {"N":>4}', '  '+'-'*36,
    ]
    for cls in present_classes:
        s = per_class_stats[cls]
        lines.append(f'  {cls:<14} {s["mean"]:>8.4f} {s["std"]:>8.4f} {s["n"]:>4}')
    lines += ['', '='*68]
    report = '\n'.join(lines)
    with open(os.path.join(out_dir, 'cosine_similarity_report.txt'), 'w') as f:
        f.write(report)
    print(report)
    print(f'\n  {protocol.upper()} done. Cosine sim: {overall_mean:.4f}')
    return summary

def main():
    parser = argparse.ArgumentParser(description='Module 5 — Explainability (v2.1: brain masking)')
    parser.add_argument('--protocol', default='all', choices=['all','merged','lodo1','lodo2','lodo3'])
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES_PER_CLASS)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    if device.type == 'cuda': print(f'  GPU: {torch.cuda.get_device_name(0)}')

    protocols = ['merged','lodo1','lodo2','lodo3'] if args.protocol == 'all' else [args.protocol]
    os.makedirs(EXPLANATIONS_DIR, exist_ok=True)

    all_summaries = {}
    for protocol in protocols:
        try: all_summaries[protocol] = explain_protocol(protocol, device, args.n_samples)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {protocol}: {e}')

    print(f'\n{"="*68}\n  EXPLAINABILITY COMPLETE (brain-masked)\n{"="*68}')
    for p, s in all_summaries.items():
        print(f'  {p.upper():<12} Cosine={s["overall_cosine_mean"]:.4f} +/- {s["overall_cosine_std"]:.4f}')
    print('='*68)
