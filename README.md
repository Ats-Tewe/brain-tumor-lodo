# Pretraining, Not Architecture, Determines Cross-Institutional Generalization in Brain Tumor MRI Classification

Code for the paper of the same name.

**Paper:** under review at the Journal of Imaging Informatics in Medicine
**Model weights:** https://huggingface.co/Ha-Te-21/brain-tumor-lodo

---

## What this paper found

Brain tumor classifiers on single public MRI datasets report accuracies above
99%. We asked what survives when the model meets an institution it has never
seen.

1. **Seed alone moved held-out accuracy by 12.99 points** (78.43% to 91.42%)
   across five runs of one fixed configuration. Every architectural difference
   we measured was smaller than that, so the architecture comparison we set out
   to make could not be made at all.
2. **Initialization resolved, and by a wide margin.** On a token-matched
   ViT-S/32 where nothing varied but the starting weights, ImageNet
   initialization was worth 26.97 points (95.47% against 68.50%). It also cut
   the between-seed standard deviation from 4.90 to 0.65.
3. **Post-hoc calibration moves the wrong way under shift.** A temperature
   fitted on source validation data sharpens the predictive distribution; the
   shifted target needs softening. The standard correction made calibration
   56% and 83% worse on the two shifted rounds.

And about the data rather than the models: **44.75% of the pooled corpus was
inter-collection duplicates.** Two of the three public collections lost roughly
two thirds of their content to near-duplicates of the third. A
leave-one-dataset-out evaluation that skips deduplication holds out an
institution that is already in its training set.

## Everything runs through one command

```bash
python run.py                 # list every command
python run.py train --help    # a script's own options
```

`run.py` dispatches to the right module and passes your arguments through
untouched.

## Reproducing the paper

```bash
pip install -r requirements.txt
```

**Preprocessing**, in this order. See `data/README.md` for where to get the
three collections.

```bash
python run.py prep.convert      # Figshare .mat -> JPEG
python run.py prep.inspect      # counts, dimensions, corrupt files
python run.py prep.dedup        # removes 9,992 of 22,328
python run.py prep.harmonize
python run.py prep.merged
```

**Training and evaluation.** GPU required; the full campaign is roughly 40
GPU-hours on an RTX 4090.

```bash
bash scripts/run_all_experiments.sh     # all four LODO protocols
bash scripts/run_randinit_seeds.sh      # the five-seed variance arm
python run.py evaluate --protocol lodo3
```

**The analyses behind the paper's claims:**

```bash
python run.py ablation.vit        # controlled initialization test, 26.97 pp
python run.py seeds               # the five-seed variance table
python run.py calibrate           # temperature scaling, both fitting modes
python run.py uncertainty         # MC dropout predictive spread
python run.py explain             # Grad-CAM and attention rollout
python run.py probe.gradpath      # shows backbone gradients are exactly zero
```

`results/` holds the recorded outputs of our own runs, so you can compare
without retraining.

## Layout

```
run.py             single entry point
src/
  common/          configuration, data loading, metrics, model adapter
  data_prep/       steps 1-5: convert, inspect, deduplicate, harmonize, split
  model/           the dual-head hybrid CNN-Transformer
  training/        training loop, schedule, loss
  evaluation/      accuracy, per-class metrics, confusion matrices
  explainability/  Grad-CAM, attention rollout, cosine agreement
  uncertainty/     MC dropout, temperature scaling, bootstrap intervals
experiments/       ablations, complexity timing, seed aggregation
scripts/           shell drivers for the full campaign
results/           recorded run outputs
checkpoints/       how to fetch the trained weights
data/              how to obtain the three source collections
```

The `verify_*.py` files are self-tests: each checks that its module behaves as
the paper describes. Run them with `python run.py verify.model` and so on.

## Environment

```
Python 3.10+, PyTorch 2.6.0+cu124, timm, scikit-learn, Pillow
```

`requirements.txt` pins what we ran; `requirements_server.txt` is the CUDA
build used on the training machine.

## Two things to know before you reproduce this

**The Mendeley collection is a re-aggregation, not an institution.** Filename
analysis finds at least four provenances inside it, including images from the
Kaggle collection and machine-generated augmented variants. Our deduplication
removes 7,996 of its 12,064 images for exactly that reason. Pool these
collections without deduplicating and your held-out "institution" is already in
your training set, and your transfer number means nothing.

**LODO Round 2 supports no transfer claim.** Figshare has no no-tumor class, so
that round evaluates three classes, its test set is 78% glioma, and its
meningioma estimate rests on 58 images. We report it for completeness and
exclude it from every cross-institutional conclusion. Do the same.

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The three source image collections carry their own
licenses; see `data/README.md`.
