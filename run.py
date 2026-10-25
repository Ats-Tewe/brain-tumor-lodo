from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS: dict[str, tuple[str, str]] = {
    "prep.convert":  ("src/data_prep/step1_convert_figshare.py",
                      "Convert the Figshare .mat files to JPEG"),
    "prep.inspect":  ("src/data_prep/step2_inspect_datasets.py",
                      "Count images, check dimensions, find corrupt files"),
    "prep.dedup":    ("src/data_prep/step3_deduplicate.py",
                      "Remove cross-collection near-duplicates (9,992 of 22,328)"),
    "prep.harmonize": ("src/data_prep/step4_harmonize.py",
                       "Resize to 224 px and stratify the Mendeley split"),
    "prep.merged":   ("src/data_prep/step5_merged_baseline.py",
                      "Build the merged baseline split"),

    "train":         ("src/training/train.py",
                      "Train one protocol of the dual-head hybrid"),
    "evaluate":      ("src/evaluation/evaluate.py",
                      "Accuracy, per-class metrics, confusion matrices"),
    "explain":       ("src/explainability/explain.py",
                      "Grad-CAM, attention rollout, cosine agreement"),

    "uncertainty":   ("src/uncertainty/uncertainty.py",
                      "MC dropout predictive spread"),
    "calibrate":     ("src/uncertainty/temperature_scaling_fixed.py",
                      "Temperature scaling, source-val and target k-fold"),
    "bootstrap":     ("src/uncertainty/bootstrap_ci.py",
                      "Bootstrap confidence intervals, B = 2000"),
    "mc.converge":   ("src/uncertainty/mc_dropout_convergence.py",
                      "Justify T = 50 against a T = 200 reference"),

    "ablation.vit":  ("experiments/ablation_vit_standalone_fixed.py",
                      "Standalone ViT-S/32, the controlled initialization test"),
    "ablation.hybrid": ("experiments/hybrid_architecture_experiments.py",
                        "Bridge depth, head configuration, transformer capacity"),
    "pretrained":    ("experiments/pretrained_hybrid_experiment.py",
                      "The ImageNet-initialized hybrid arm"),
    "seeds":         ("experiments/aggregate_seeds.py",
                      "Aggregate the five-seed runs into the variance table"),
    "analyze.ablation": ("experiments/analyze_ablation.py",
                         "Summarize the ablation logs"),
    "complexity":    ("experiments/computational_complexity.py",
                      "Parameter counts and latency"),
    "complexity.pretrained": ("experiments/complexity_pretrained.py",
                              "Latency of the pretrained hybrid"),
    "complexity.vit32": ("experiments/complexity_vit_s32.py",
                         "Latency of the token-matched ViT-S/32"),

    "probe.gradpath": ("experiments/probe_gradpath.py",
                       "Show the stop-gradient leaves backbone gradients at zero"),
    "verify.model":   ("src/model/verify_model.py", "Self-test: architecture"),
    "verify.training": ("src/training/verify_training.py", "Self-test: training"),
    "verify.evaluation": ("src/evaluation/verify_evaluation.py",
                          "Self-test: evaluation"),
    "verify.explain": ("src/explainability/verify_explain.py",
                       "Self-test: explainability"),
    "verify.uncertainty": ("src/uncertainty/verify_uncertainty.py",
                           "Self-test: uncertainty"),
}

def usage() -> int:
    print("usage: python run.py <command> [args...]\n")
    width = max(len(c) for c in COMMANDS)
    group = None
    for name, (path, blurb) in COMMANDS.items():
        head = name.split(".")[0] if "." in name else name
        if head != group:
            print()
            group = head
        print(f"  {name:<{width}}  {blurb}")
    print("\nArguments after the command go straight to the script, so")
    print("  python run.py train --help   shows the training script's own help.")
    return 1

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()

    name = argv[0]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n")
        close = [c for c in COMMANDS if c.startswith(name.split(".")[0])]
        if close:
            print("did you mean:")
            for c in close:
                print(f"  {c}")
            print()
        return usage()

    rel, _ = COMMANDS[name]
    path = ROOT / rel
    if not path.is_file():
        print(f"!! {rel} is missing from this checkout.")
        return 2

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    spec = importlib.util.spec_from_file_location(f"_cmd_{name}", path)
    module = importlib.util.module_from_spec(spec)

    saved = sys.argv
    sys.argv = [name] + argv[1:]
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            return 0
        result = module.main()
        return 0 if result is None else int(result)
    finally:
        sys.argv = saved

if __name__ == "__main__":
    raise SystemExit(main())
