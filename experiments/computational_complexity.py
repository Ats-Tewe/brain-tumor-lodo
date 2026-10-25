
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module2_model.model import build_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'results')
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_WARMUP     = 20
N_MEASURE    = 100
IMG_SIZE     = 320

class CNNOnly(nn.Module):
    """EfficientNet-B0 + GAP + FC(1280->4) — no ViT."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False,
                                           num_classes=0, global_pool='')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, 4)

    def forward(self, x):
        f = self.backbone(x)
        f = self.gap(f).flatten(1)
        f = self.dropout(f)
        return self.fc(f)

class StandaloneViT(nn.Module):
    """ViT-Small patch16/224 + Dropout + FC(384->4) — no EfficientNet."""
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False,
                                      img_size=IMG_SIZE, num_classes=0)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(384, 4)

    def forward(self, x):
        cls = self.vit(x)
        cls = self.drop(cls)
        return self.head(cls)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_time(model, device, batch_size=1):
    """Measure average inference time per image in milliseconds."""
    model.eval()
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(N_MEASURE):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0 / batch_size)

    times = np.array(times)
    return {
        'mean_ms': round(float(times.mean()), 2),
        'std_ms':  round(float(times.std()), 2),
        'min_ms':  round(float(times.min()), 2),
        'max_ms':  round(float(times.max()), 2),
        'n_runs':  N_MEASURE,
        'batch_size': batch_size,
    }

def main():
    print("\n" + "=" * 68)
    print("  COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("  DualHead Hybrid CNN-ViT Paper — IEEE J-BHI")
    print("=" * 68)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Warmup: {N_WARMUP} runs | Measure: {N_MEASURE} runs")
    print(f"  Image:  {IMG_SIZE}x{IMG_SIZE}x3\n")

    results = {}

    print("  [1/3] DualHead Hybrid CNN-ViT...")
    hybrid = build_model(num_classes=4, pretrained=False, img_size=IMG_SIZE).to(DEVICE)
    total_p, train_p = count_params(hybrid)

    comp = hybrid.count_parameters()
    print(f"    Total params: {total_p:,}")
    for k, v in comp.items():
        print(f"      {k}: {v:,}")

    class HybridWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x, return_attention=False)
            return out['vit_logits']

    wrapper = HybridWrapper(hybrid).to(DEVICE)
    timing = measure_inference_time(wrapper, DEVICE)
    print(f"    Inference: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms/image")

    results['dualhead_hybrid'] = {
        'name': 'DualHead Hybrid CNN-ViT',
        'total_params': total_p,
        'trainable_params': train_p,
        'component_breakdown': comp,
        'inference_time': timing,
    }
    del hybrid, wrapper
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n  [2/3] CNN-only (EfficientNet-B0 + GAP + FC)...")
    cnn = CNNOnly().to(DEVICE)
    total_p, train_p = count_params(cnn)
    print(f"    Total params: {total_p:,}")

    timing = measure_inference_time(cnn, DEVICE)
    print(f"    Inference: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms/image")

    results['cnn_only'] = {
        'name': 'CNN-only (EfficientNet-B0)',
        'total_params': total_p,
        'trainable_params': train_p,
        'inference_time': timing,
    }
    del cnn
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n  [3/3] Standalone ViT-Small (patch16/224)...")
    vit = StandaloneViT().to(DEVICE)
    total_p, train_p = count_params(vit)
    print(f"    Total params: {total_p:,}")

    timing = measure_inference_time(vit, DEVICE)
    print(f"    Inference: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms/image")

    results['standalone_vit'] = {
        'name': 'Standalone ViT-Small',
        'total_params': total_p,
        'trainable_params': train_p,
        'inference_time': timing,
    }
    del vit
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n  [Ref] Plain EfficientNet-B0 (timm default)...")
    eff = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4).to(DEVICE)
    total_p, _ = count_params(eff)
    print(f"    Total params: {total_p:,}")
    timing_eff = measure_inference_time(eff, DEVICE)
    print(f"    Inference: {timing_eff['mean_ms']:.2f} ± {timing_eff['std_ms']:.2f} ms/image")
    results['efficientnet_b0_baseline'] = {
        'name': 'EfficientNet-B0 (plain, reference)',
        'total_params': total_p,
        'inference_time': timing_eff,
    }
    del eff
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n  [Ref] Plain ViT-S/16 (timm default)...")
    vits = timm.create_model('vit_small_patch16_224', pretrained=False,
                             img_size=IMG_SIZE, num_classes=4).to(DEVICE)
    total_p, _ = count_params(vits)
    print(f"    Total params: {total_p:,}")
    timing_vits = measure_inference_time(vits, DEVICE)
    print(f"    Inference: {timing_vits['mean_ms']:.2f} ± {timing_vits['std_ms']:.2f} ms/image")
    results['vit_small_baseline'] = {
        'name': 'ViT-S/16 (plain, reference)',
        'total_params': total_p,
        'inference_time': timing_vits,
    }
    del vits

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'computational_complexity.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] JSON -> {json_path}")

    h = results['dualhead_hybrid']
    c = results['cnn_only']
    v = results['standalone_vit']

    report_lines = [
        "=" * 70,
        "  COMPUTATIONAL COMPLEXITY — MODEL COMPARISON",
        "  IEEE J-BHI Paper: DualHead Hybrid CNN-ViT",
        "=" * 70, "",
        f"  {'Model':<35} {'Params':>12} {'ms/image':>12}",
        "  " + "-" * 62,
        f"  {'DualHead Hybrid CNN-ViT':<35} "
        f"{h['total_params']/1e6:>10.2f}M "
        f"{h['inference_time']['mean_ms']:>10.2f}",
        f"  {'CNN-only (EfficientNet-B0+GAP+FC)':<35} "
        f"{c['total_params']/1e6:>10.2f}M "
        f"{c['inference_time']['mean_ms']:>10.2f}",
        f"  {'Standalone ViT-Small':<35} "
        f"{v['total_params']/1e6:>10.2f}M "
        f"{v['inference_time']['mean_ms']:>10.2f}",
        "  " + "-" * 62, "",
        "  COMPONENT BREAKDOWN (DualHead Hybrid):",
    ]
    for k, val in h['component_breakdown'].items():
        report_lines.append(f"    {k:<35} {val:>12,}")
    report_lines += [
        "", "",
        "  MEASUREMENT CONFIGURATION:",
        f"    Input resolution: {IMG_SIZE}x{IMG_SIZE}",
        f"    Backbone feature grid: {IMG_SIZE//32}x{IMG_SIZE//32} "
        f"({(IMG_SIZE//32)**2} ViT tokens)",
        f"    Warmup iters: {N_WARMUP}   Timed iters: {N_MEASURE}",
        "",
        "  NOTE: The hybrid model is ~{:.1f}x larger than CNN-only".format(
            h['total_params'] / c['total_params']),
        "  but inference overhead stays modest because tokenization reuses the",
        "  shared backbone feature grid rather than re-patching the image.",
        "",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    rpt_path = os.path.join(RESULTS_DIR, 'computational_complexity.txt')
    with open(rpt_path, 'w') as f:
        f.write(report)
    print(f"  [SAVED] Report -> {rpt_path}")
    print("\n" + report)
