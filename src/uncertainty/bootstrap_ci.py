
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2_model.model import build_model
from module3_training.train import (
    BrainTumorDataset, CLASS_NAMES, CLASS_TO_IDX, HARMONIZED_DIR, MODELS_DIR
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'results')
N_BOOTSTRAP  = 2000
CI_LEVEL     = 0.95
BATCH_SIZE   = 32

PROTOCOLS = {
    'merged': {
        'checkpoint': os.path.join(MODELS_DIR, 'merged', 'best_model.pth'),
        'test_dir':   os.path.join(HARMONIZED_DIR, 'merged_baseline', 'test'),
        'description': 'Merged Baseline',
    },
    'lodo1': {
        'checkpoint': os.path.join(MODELS_DIR, 'lodo1', 'best_model.pth'),
        'test_dir':   os.path.join(HARMONIZED_DIR, 'lodo_round1', 'test'),
        'description': 'LODO Round 1 (Test: Kaggle)',
    },
    'lodo2': {
        'checkpoint': os.path.join(MODELS_DIR, 'lodo2', 'best_model.pth'),
        'test_dir':   os.path.join(HARMONIZED_DIR, 'lodo_round2', 'test'),
        'description': 'LODO Round 2 (Test: Figshare)',
    },
    'lodo3': {
        'checkpoint': os.path.join(MODELS_DIR, 'lodo3', 'best_model.pth'),
        'test_dir':   os.path.join(HARMONIZED_DIR, 'lodo_round3', 'test'),
        'description': 'LODO Round 3 (Test: Mendeley)',
    },
}

def collect_predictions(protocol, device, use_tta=False):
    cfg = PROTOCOLS[protocol]
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    img_size = ckpt.get('img_size', 224)

    model = build_model(num_classes=4, pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    present_classes = [
        c for c in CLASS_NAMES
        if os.path.isdir(os.path.join(cfg['test_dir'], c))
        and len([f for f in os.listdir(os.path.join(cfg['test_dir'], c))
                if f.lower().endswith(('.jpg','.jpeg','.png'))]) > 0
    ]

    ds = BrainTumorDataset(cfg['test_dir'], transform=eval_tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if use_tta:
                import torchvision.transforms.functional as _TF
                views = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.stack([_TF.rotate(img, 10) for img in images]),
                    torch.stack([_TF.rotate(img, -10) for img in images]),
                ]
                acc = None
                for _v in views:
                    _o = model(_v, return_attention=False)
                    _p = F.softmax(_o['vit_logits'], dim=-1)
                    acc = _p if acc is None else acc + _p
                probs = (acc / len(views)).cpu().numpy()
            else:
                out = model(images, return_attention=False)
                probs = F.softmax(out['vit_logits'], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_probs).argmax(axis=1), np.concatenate(all_labels), present_classes

def single_bootstrap_iter(probs, preds, labels, present_classes, rng, max_retries=5):
    N = len(labels)
    present_global_idx = [CLASS_TO_IDX[c] for c in present_classes]
    C = len(present_classes)
    g2l = {g: l for l, g in enumerate(present_global_idx)}

    for _ in range(max_retries + 1):
        idx = rng.integers(0, N, size=N)
        y_boot = labels[idx]; p_boot = preds[idx]; pr_boot = probs[idx]
        if set(np.unique(y_boot)).issuperset(set(present_global_idx)):
            break

    acc = float((y_boot == p_boot).mean())
    mask = np.isin(y_boot, present_global_idx)
    y_loc = np.array([g2l[yi] for yi in y_boot[mask]])
    p_loc = np.array([g2l.get(pi, 0) for pi in p_boot[mask]])
    pr_loc = pr_boot[mask][:, present_global_idx]
    _rowsum = pr_loc.sum(axis=1, keepdims=True)
    pr_loc = np.divide(pr_loc, _rowsum,
                       out=np.full_like(pr_loc, 1.0 / max(C, 1)),
                       where=_rowsum > 0)

    try: mf1 = float(f1_score(y_loc, p_loc, average='macro', zero_division=0, labels=list(range(C))))
    except: mf1 = float('nan')

    if len(np.unique(y_loc)) < C: mauc = float('nan')
    else:
        try: mauc = float(roc_auc_score(y_loc, pr_loc, multi_class='ovr', average='macro', labels=list(range(C))))
        except: mauc = float('nan')

    return acc, mf1, mauc

def bootstrap_ci(probs, preds, labels, present_classes, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=42):
    rng = np.random.default_rng(seed)
    accs, f1s, aucs = [], [], []
    for _ in range(n_bootstrap):
        a, f, au = single_bootstrap_iter(probs, preds, labels, present_classes, rng)
        accs.append(a); f1s.append(f); aucs.append(au)

    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100.0

    def summarise(vals, name):
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            return {'mean': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0, 'valid_pct': 0.0}
        return {
            'mean': round(float(arr.mean()), 4),
            'ci_lo': round(float(np.percentile(arr, lo_pct)), 4),
            'ci_hi': round(float(np.percentile(arr, hi_pct)), 4),
            'valid_pct': round(len(arr)/len(vals)*100, 1),
        }

    return {
        'accuracy': summarise(accs, 'accuracy'),
        'macro_f1': summarise(f1s, 'macro_f1'),
        'macro_auc': summarise(aucs, 'macro_auc'),
        'n_bootstrap': n_bootstrap, 'ci_level': ci_level,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bootstrap CIs for all protocols')
    parser.add_argument('--tta', action='store_true',
                        help='Apply 5-view TTA, matching evaluate.py. Use this so '
                             'the CIs describe the same configuration as the '
                             'point estimates reported in Table III.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')
    print(f'  Bootstrap: B={N_BOOTSTRAP}, CI={int(CI_LEVEL*100)}%  TTA={args.tta}\n')

    all_results = {}
    for protocol in ['merged', 'lodo1', 'lodo2', 'lodo3']:
        print(f'  {protocol.upper()}...')
        probs, preds, labels, present_classes = collect_predictions(protocol, device, use_tta=args.tta)
        ci = bootstrap_ci(probs, preds, labels, present_classes)
        point_acc = float((preds == labels).mean())
        all_results[protocol] = {
            'description': PROTOCOLS[protocol]['description'],
            'n_test': len(labels), 'present_classes': present_classes,
            'point_accuracy': round(point_acc, 4), **ci,
        }
        a = ci['accuracy']; f = ci['macro_f1']; au = ci['macro_auc']
        print(f'    Acc={point_acc*100:.2f}% [{a["ci_lo"]*100:.2f}%,{a["ci_hi"]*100:.2f}%]  '
              f'F1={f["mean"]*100:.2f}%  AUC={au["mean"]*100:.2f}%')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'bootstrap_ci_all_protocols.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    lines = ['='*68, '  BOOTSTRAP 95% CI — ALL PROTOCOLS (v2)', '='*68, '',
             f'  {"Protocol":<14} {"Accuracy":>10} {"95% CI":>22} {"F1":>8} {"AUC":>8}', '  '+'-'*65]
    for p, r in all_results.items():
        a = r['accuracy']; f = r['macro_f1']; au = r['macro_auc']
        lines.append(f'  {p.upper():<14} {r["point_accuracy"]*100:>8.2f}%  '
                     f'[{a["ci_lo"]*100:.2f}%,{a["ci_hi"]*100:.2f}%]  '
                     f'{f["mean"]*100:>7.2f}% {au["mean"]*100:>7.2f}%')
    lines += ['', '='*68]
    report = '\n'.join(lines)
    with open(os.path.join(RESULTS_DIR, 'bootstrap_ci_report.txt'), 'w') as f:
        f.write(report)
    print('\n' + report)
