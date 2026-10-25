
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4_evaluation.evaluate import (
    load_checkpoint, run_inference, compute_metrics,
    bootstrap_ci, plot_confusion_matrix, plot_roc_curves,
    save_comparison_csv, PROTOCOLS, RESULTS_DIR,
)
from module3_training.train import (
    BrainTumorDataset, get_eval_transform,
    CLASS_NAMES, CLASS_TO_IDX, HARMONIZED_DIR, MODELS_DIR,
)
from torch.utils.data import DataLoader, Subset
from module2_model.model import build_model

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
    print('  MODULE 4 — Evaluation Pipeline Verification')
    print('=' * 68)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device.type.upper()}')
    if device.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print()

    all_passed = True
    out_dir    = os.path.join(RESULTS_DIR, '_verify')
    os.makedirs(out_dir, exist_ok=True)

    print('  [CHECK 1] Checkpoint loading...')
    ckpt_path = PROTOCOLS['merged']['checkpoint']
    try:
        model, meta = load_checkpoint(ckpt_path, device)
        all_passed &= check('Checkpoint loads without error', True,
                            f'epoch={meta["epoch"]}  val_acc={meta["val_acc"]:.4f}')
        all_passed &= check('Model in eval mode', not model.training)
    except Exception as e:
        all_passed &= check('Checkpoint loads without error', False, str(e))
        print('\n  CRITICAL: Cannot continue without a checkpoint.')
        return

    print()
    print('  [CHECK 2] Test dataset loading...')
    test_dir = PROTOCOLS['merged']['test_dir']
    test_ds  = BrainTumorDataset(test_dir, transform=get_eval_transform())
    present  = list(test_ds.class_counts.keys())
    all_passed &= check('Test dataset loads', len(test_ds) > 0,
                        f'{len(test_ds)} images, classes={present}')
    all_passed &= check('All 4 classes present in merged test',
                        len(present) == 4, str(present))

    indices = list(range(min(50, len(test_ds))))
    sub_ds  = Subset(test_ds, indices)
    loader  = DataLoader(sub_ds, batch_size=8, shuffle=False, num_workers=0)

    print()
    print('  [CHECK 3+4] Inference...')
    results = run_inference(model, loader, device, present)

    n = len(results['true_labels'])
    all_passed &= check('Inference completes', n == len(indices),
                        f'{n} predictions')
    all_passed &= check('vit_probs shape [N, 4]',
                        results['vit_probs'].shape == (n, 4),
                        f'Got {results["vit_probs"].shape}')
    prob_sums = results['vit_probs'].sum(axis=1)
    all_passed &= check('Softmax probs sum to ~1.0',
                        np.allclose(prob_sums, 1.0, atol=1e-5),
                        f'Max deviation: {np.abs(prob_sums - 1.0).max():.2e}')

    print()
    print('  [CHECK 5-8] Metric computation...')
    metrics = compute_metrics(results, present)
    vit = metrics['vit']
    cnn = metrics['cnn']

    all_passed &= check('Accuracy in [0, 1]',
                        0.0 <= vit['accuracy'] <= 1.0,
                        f'{vit["accuracy"]:.4f}')
    all_passed &= check('Macro F1 in [0, 1]',
                        0.0 <= vit['macro_f1'] <= 1.0,
                        f'{vit["macro_f1"]:.4f}')
    auc_val = vit['macro_auc']
    auc_ok  = (0.0 <= auc_val <= 1.0) or np.isnan(auc_val)
    auc_note = f'{auc_val:.4f}' if not np.isnan(auc_val) else 'nan (expected on tiny subset — OK on full test)'
    all_passed &= check('Macro AUC valid (or nan on tiny subset)', auc_ok, auc_note)
    all_passed &= check('CNN accuracy in [0, 1]',
                        0.0 <= cnn['accuracy'] <= 1.0,
                        f'{cnn["accuracy"]:.4f}')
    all_passed &= check('Per-class metrics for all 4 classes',
                        len(vit['per_class']) == 4,
                        str(list(vit['per_class'].keys())))

    print()
    print('  [CHECK 9] Bootstrap 95% CI (small N simulation)...')
    np.random.seed(42)
    fake_true = np.random.choice([0, 1, 2], size=58)
    fake_pred = fake_true.copy()
    fake_pred[:5] = (fake_pred[:5] + 1) % 3

    try:
        ci_result = bootstrap_ci(
            y_true=fake_true, y_pred=fake_pred,
            class_name='meningioma', class_idx=1,
            n=100,
        )
        ci_ok = (
            isinstance(ci_result['f1'], tuple) and
            len(ci_result['f1']) == 2 and
            ci_result['f1'][0] <= ci_result['f1'][1]
        )
        all_passed &= check('Bootstrap CI returns valid bounds', ci_ok,
                            f'F1 CI: [{ci_result["f1"][0]:.3f}, {ci_result["f1"][1]:.3f}]')
    except Exception as e:
        all_passed &= check('Bootstrap CI runs without error', False, str(e))

    print()
    print('  [CHECK 10] Confusion matrix plot...')
    cm_path = os.path.join(out_dir, 'test_cm.png')
    try:
        cm = np.array(vit['confusion_matrix'])
        plot_confusion_matrix(cm, present, cm_path,
                              'Verify — Confusion Matrix', normalise=True)
        all_passed &= check('Confusion matrix PNG saved',
                            os.path.isfile(cm_path), cm_path)
    except Exception as e:
        all_passed &= check('Confusion matrix plot', False, str(e))

    print()
    print('  [CHECK 11] ROC curve plot...')
    roc_path = os.path.join(out_dir, 'test_roc.png')
    try:
        plot_roc_curves(results, present, roc_path, 'Verify — ROC Curves')
        all_passed &= check('ROC curves PNG saved',
                            os.path.isfile(roc_path), roc_path)
    except Exception as e:
        all_passed &= check('ROC curve plot', False, str(e))

    print()
    print('  [CHECK 12] JSON metrics save/load...')
    json_path = os.path.join(out_dir, 'test_metrics.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(json_path, 'r') as f:
            loaded = json.load(f)
        all_passed &= check('JSON saves and reloads correctly',
                            loaded['vit']['accuracy'] == vit['accuracy'],
                            f'acc={loaded["vit"]["accuracy"]}')
    except Exception as e:
        all_passed &= check('JSON save/load', False, str(e))

    print()
    print('  [CHECK 13] Round 2 separate No Tumor evaluation...')
    from module4_evaluation.evaluate import evaluate_notumor_lodo2, KAGGLE_NOTUMOR_TEST_DIR
    kaggle_notumor = os.path.join(KAGGLE_NOTUMOR_TEST_DIR, "notumor")
    if os.path.isdir(kaggle_notumor):
        try:
            lodo2_ckpt = PROTOCOLS['lodo2']['checkpoint']
            lodo2_model, _ = load_checkpoint(lodo2_ckpt, device)
            result = evaluate_notumor_lodo2(lodo2_model, device)
            valid = (
                isinstance(result, dict) and
                'accuracy' in result and
                0.0 <= result['accuracy'] <= 1.0
            )
            all_passed &= check('Round 2 No Tumor eval runs correctly', valid,
                                f'acc={result.get("accuracy", "N/A")}  '
                                f'n={result.get("n_samples", "N/A")}')
        except Exception as e:
            all_passed &= check('Round 2 No Tumor eval', False, str(e))
    else:
        all_passed &= check('Kaggle No Tumor directory exists', False,
                            f'Not found: {kaggle_notumor}')

    print()
    print('=' * 68)
    if all_passed:
        print('  ALL CHECKS PASSED')
        print()
        print('  Evaluation pipeline is verified and ready.')
        print()
        print('  To run full evaluation (all 4 protocols):')
        print('    python module4_evaluation\\evaluate.py')
        print()
        print('  To run a single protocol:')
        print('    python module4_evaluation\\evaluate.py --protocol merged')
        print('    python module4_evaluation\\evaluate.py --protocol lodo1')
        print('    python module4_evaluation\\evaluate.py --protocol lodo2')
        print('    python module4_evaluation\\evaluate.py --protocol lodo3')
    else:
        print('  ONE OR MORE CHECKS FAILED')
        print('  Fix [FAIL] items above before running full evaluation.')
    print('=' * 68)
    print()
