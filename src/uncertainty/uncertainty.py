
import os, sys, json, argparse
import numpy as np
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2_model.model import build_model
from module3_training.train import (
    BrainTumorDataset, CLASS_NAMES, CLASS_TO_IDX, HARMONIZED_DIR, MODELS_DIR
)

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNCERTAINTY_DIR = os.path.join(PROJECT_ROOT, "outputs", "uncertainty")
N_MC_PASSES     = 50
UNCERTAINTY_THRESHOLD = 0.10
BATCH_SIZE      = 16
N_BINS          = 10

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

def mc_dropout_inference(model, loader, device, n_passes=N_MC_PASSES):
    model.enable_mc_dropout()
    all_pass_probs = []
    for t in tqdm(range(n_passes), desc='  MC passes'):
        pass_probs = []
        with torch.no_grad():
            for images, _ in loader:
                out = model(images.to(device), return_attention=False)
                pass_probs.append(F.softmax(out['vit_logits'], dim=-1).cpu().numpy())
        all_pass_probs.append(np.concatenate(pass_probs, axis=0))
    all_pass_probs = np.stack(all_pass_probs, axis=0)
    true_labels = []
    for _, labels in loader:
        true_labels.extend(labels.numpy())
    return {
        'mean_probs': all_pass_probs.mean(axis=0),
        'std_probs': all_pass_probs.std(axis=0),
        'true_labels': np.array(true_labels),
        'all_passes': all_pass_probs,
    }

def compute_ece(mean_probs, pred_classes, true_labels, n_bins=N_BINS):
    confidences = mean_probs.max(axis=1)
    correctness = (pred_classes == true_labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(true_labels)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        n_bin = mask.sum()
        if n_bin == 0: continue
        ece += (n_bin / N) * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)

def compute_metrics(mc_results, present_classes, threshold=UNCERTAINTY_THRESHOLD):
    mean_probs = mc_results['mean_probs']
    std_probs = mc_results['std_probs']
    true_labels = mc_results['true_labels']
    N = len(true_labels)
    pred_classes = mean_probs.argmax(axis=1)

    records = []
    for i in range(N):
        pred_idx = int(pred_classes[i])
        true_idx = int(true_labels[i])
        max_std = float(std_probs[i].max())
        entropy = float(-np.sum(mean_probs[i] * np.log(mean_probs[i] + 1e-10)))
        records.append({
            'true_class': CLASS_NAMES[true_idx], 'pred_class': CLASS_NAMES[pred_idx],
            'is_correct': pred_idx == true_idx, 'is_flagged': max_std > threshold,
            'max_prob': round(float(mean_probs[i, pred_idx]), 4),
            'max_std': round(max_std, 4), 'entropy': round(entropy, 4),
        })

    per_class = {}
    for cls in present_classes:
        recs = [r for r in records if r['true_class'] == cls]
        if not recs: continue
        stds = [r['max_std'] for r in recs]
        correct_recs = [r for r in recs if r['is_correct']]
        wrong_recs = [r for r in recs if not r['is_correct']]
        per_class[cls] = {
            'n': len(recs), 'mean_std': round(float(np.mean(stds)), 4),
            'std_correct': round(float(np.mean([r['max_std'] for r in correct_recs])), 4) if correct_recs else None,
            'std_incorrect': round(float(np.mean([r['max_std'] for r in wrong_recs])), 4) if wrong_recs else None,
            'n_flagged': sum(1 for r in recs if r['is_flagged']),
        }

    correct_stds = [r['max_std'] for r in records if r['is_correct']]
    wrong_stds = [r['max_std'] for r in records if not r['is_correct']]
    n_flagged = sum(1 for r in records if r['is_flagged'])
    accuracy = sum(1 for r in records if r['is_correct']) / N
    ece = compute_ece(mean_probs, pred_classes, true_labels)

    return {
        'n_images': N, 'present_classes': present_classes,
        'overall_accuracy': round(accuracy, 4),
        'overall_mean_std': round(float(np.mean([r['max_std'] for r in records])), 4),
        'correct_mean_std': round(float(np.mean(correct_stds)), 4) if correct_stds else None,
        'incorrect_mean_std': round(float(np.mean(wrong_stds)), 4) if wrong_stds else None,
        'n_flagged': n_flagged, 'flagged_rate': round(n_flagged / N, 4),
        'ece': round(ece, 4), 'threshold': threshold,
        'per_class': per_class, 'records': records,
    }

def plot_reliability(mean_probs, pred_classes, true_labels, out_path, title, ece):
    confidences = mean_probs.max(axis=1)
    correctness = (pred_classes == true_labels).astype(float)
    bins = np.linspace(0, 1, N_BINS + 1)
    bin_accs, bin_centres = [], []
    for i in range(N_BINS):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        bin_centres.append((bins[i]+bins[i+1])/2)
        bin_accs.append(correctness[mask].mean() if mask.sum() > 0 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title}\nECE = {ece:.4f}', fontsize=13, fontweight='bold')
    ax1.bar(bin_centres, bin_accs, width=0.09, color='steelblue', alpha=0.8, label='Accuracy', zorder=3)
    ax1.plot([0,1],[0,1],'r--',lw=2,label='Perfect'); ax1.fill_between(bin_centres, bin_accs, bin_centres, alpha=0.2, color='red')
    ax1.set_xlabel('Confidence'); ax1.set_ylabel('Accuracy'); ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title('Reliability Diagram')
    ax2.hist(confidences, bins=20, color='teal', alpha=0.7, edgecolor='white')
    ax2.axvline(confidences.mean(), color='red', ls='--', lw=2, label=f'Mean={confidences.mean():.3f}')
    ax2.set_xlabel('Confidence'); ax2.set_ylabel('Count'); ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    ax2.set_title('Confidence Distribution')
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  [SAVED] {out_path}')

def plot_uncertainty_dist(metrics, present_classes, out_path, title):
    records = metrics['records']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    correct_stds = [r['max_std'] for r in records if r['is_correct']]
    incorrect_stds = [r['max_std'] for r in records if not r['is_correct']]
    bns = np.linspace(0, max(r['max_std'] for r in records)+0.01, 30)
    if correct_stds: ax1.hist(correct_stds, bins=bns, alpha=0.7, color='steelblue', label=f'Correct ({len(correct_stds)})', edgecolor='white')
    if incorrect_stds: ax1.hist(incorrect_stds, bins=bns, alpha=0.7, color='tomato', label=f'Incorrect ({len(incorrect_stds)})', edgecolor='white')
    ax1.axvline(metrics['threshold'], color='orange', ls='--', lw=2, label=f'Threshold ({metrics["threshold"]})')
    ax1.set_xlabel('Uncertainty (max std)'); ax1.set_ylabel('Count'); ax1.legend(); ax1.grid(alpha=0.3, axis='y')
    class_stds = [[r['max_std'] for r in records if r['true_class']==cls] for cls in present_classes if any(r['true_class']==cls for r in records)]
    labs = [c.capitalize() for c in present_classes if any(r['true_class']==c for r in records)]
    if class_stds:
        bp = ax2.boxplot(class_stds, labels=labs, patch_artist=True, medianprops={'color':'black','lw':2})
        for p, c in zip(bp['boxes'], ['#2196F3','#F44336','#4CAF50','#FF9800']): p.set_facecolor(c); p.set_alpha(0.7)
        ax2.axhline(metrics['threshold'], color='orange', ls='--', lw=2)
    ax2.set_ylabel('Uncertainty'); ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  [SAVED] {out_path}')

def save_report(protocol, metrics, out_dir):
    m = metrics; pcs = m['present_classes']
    lines = [
        '='*68, f'  UNCERTAINTY — {protocol.upper()} (v2)', '='*68, '',
        f'  MC Passes: T={N_MC_PASSES}  |  Threshold: {m["threshold"]}  |  N={m["n_images"]}', '',
        f'  Accuracy (MC mean):  {m["overall_accuracy"]*100:.2f}%',
        f'  Mean uncertainty:    {m["overall_mean_std"]:.4f}',
        f'  Correct mean std:    {m["correct_mean_std"]:.4f}',
        f'  Incorrect mean std:  {m["incorrect_mean_std"]:.4f}' if m["incorrect_mean_std"] else '  Incorrect mean std:  N/A',
        f'  High-uncertainty:    {m["n_flagged"]}/{m["n_images"]} ({m["flagged_rate"]*100:.1f}%)',
        f'  ECE:                 {m["ece"]:.4f}', '',
        f'  {"Class":<14} {"MeanStd":>8} {"CorrStd":>8} {"WrongStd":>9} {"Flagged":>8}', '  '+'-'*50,
    ]
    for cls in pcs:
        s = m['per_class'].get(cls, {})
        cs = f'{s.get("std_correct",0):.4f}' if s.get('std_correct') is not None else 'N/A'
        ws = f'{s.get("std_incorrect",0):.4f}' if s.get('std_incorrect') is not None else 'N/A'
        lines.append(f'  {cls:<14} {s.get("mean_std",0):>8.4f} {cs:>8} {ws:>9} {s.get("n_flagged",0):>4}/{s.get("n",0)}')
    lines += ['', '='*68]
    report = '\n'.join(lines)
    path = os.path.join(out_dir, 'uncertainty_report.txt')
    with open(path, 'w', encoding='utf-8') as f: f.write(report)
    print(report)

def run_protocol(protocol, device, n_passes=N_MC_PASSES):
    cfg = PROTOCOLS[protocol]
    out_dir = os.path.join(UNCERTAINTY_DIR, protocol)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*68}\n  UNCERTAINTY: {protocol.upper()} (T={n_passes})\n{"="*68}')

    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    img_size = ckpt.get('img_size', 224)
    print(f'  img_size={img_size}px')

    model = build_model(num_classes=4, pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(ckpt['model_state'])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    test_ds = BrainTumorDataset(cfg['test_dir'], transform=eval_tf)
    present_classes = list(test_ds.class_counts.keys())
    print(f'  Test: {len(test_ds)} images, classes: {present_classes}')

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f'  Running {n_passes} MC Dropout passes...')
    mc = mc_dropout_inference(model, loader, device, n_passes)

    print('  Computing metrics...')
    metrics = compute_metrics(mc, present_classes)
    metrics['protocol'] = protocol
    metrics['img_size'] = img_size
    metrics['n_passes'] = n_passes

    json_out = {k: v for k, v in metrics.items() if k != 'records'}
    with open(os.path.join(out_dir, 'uncertainty_summary.json'), 'w') as f:
        json.dump(json_out, f, indent=2)

    pred_classes = mc['mean_probs'].argmax(axis=1)
    plot_reliability(mc['mean_probs'], pred_classes, mc['true_labels'],
                     os.path.join(out_dir, 'reliability_diagram.png'),
                     f'Calibration — {protocol.upper()} (T={n_passes})', metrics['ece'])
    plot_uncertainty_dist(metrics, present_classes,
                          os.path.join(out_dir, 'uncertainty_distribution.png'),
                          f'Uncertainty — {protocol.upper()} (T={n_passes})')
    save_report(protocol, metrics, out_dir)

    print(f'\n  {protocol.upper()}: ECE={metrics["ece"]:.4f}, Flagged={metrics["n_flagged"]}/{metrics["n_images"]}')
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Module 6 — Uncertainty')
    parser.add_argument('--protocol', default='all', choices=['all','merged','lodo1','lodo2','lodo3'])
    parser.add_argument('--n_passes', type=int, default=N_MC_PASSES)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    if device.type == 'cuda': print(f'  GPU: {torch.cuda.get_device_name(0)}')

    protocols = ['merged','lodo1','lodo2','lodo3'] if args.protocol == 'all' else [args.protocol]
    os.makedirs(UNCERTAINTY_DIR, exist_ok=True)

    all_s = {}
    for p in protocols:
        try: all_s[p] = run_protocol(p, device, args.n_passes)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [ERROR] {p}: {e}')

    print(f'\n{"="*68}\n  UNCERTAINTY COMPLETE\n{"="*68}')
    print(f'  {"Protocol":<12} {"MeanStd":>8} {"ECE":>8} {"Flagged":>10} {"Accuracy":>10}')
    print('  '+'-'*50)
    for p, m in all_s.items():
        print(f'  {p.upper():<12} {m["overall_mean_std"]:>8.4f} {m["ece"]:>8.4f} {m["flagged_rate"]*100:>9.1f}% {m["overall_accuracy"]*100:>9.2f}%')
    print('='*68)
