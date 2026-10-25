# Trained model weights

Not in this repository: `best_model.pth` is 310 MB and GitHub hard-blocks any
file over 100 MB. They are hosted on Hugging Face.

**https://huggingface.co/Ha-Te-21/brain-tumor-lodo**

## What is archived

Seed 42 throughout. Each is the selected checkpoint, not the last epoch, so
each corresponds to a number in the paper.

| file | protocol | accuracy |
|---|---|---|
| `merged_best_model.pth` | Merged baseline | 99.68% |
| `lodo1_best_model.pth` | LODO Round 1, Kaggle held out | 94.25% |
| `lodo2_best_model.pth` | LODO Round 2, Figshare held out | 99.34%, three classes |
| `lodo3_best_model.pth` | LODO Round 3, Mendeley held out | 94.12% |

`lodo3` is the model behind every calibration, uncertainty and explanation
result in the paper.

The other four seeds of the variance analysis are not archived; `results/`
carries their recorded metrics, which is what the F-test is computed from.

## Downloading

```python
from huggingface_hub import snapshot_download
snapshot_download("Ha-Te-21/brain-tumor-lodo", local_dir="checkpoints")
```

Or the whole repository:

```bash
git clone https://huggingface.co/Ha-Te-21/brain-tumor-lodo checkpoints
```

## Loading

```python
import torch
state = torch.load("checkpoints/lodo3_best_model.pth", map_location="cpu")
model.load_state_dict(state.get("model_state_dict", state))
model.eval()
```

Two things that will bite you:

**Input size is 320, not 224.** Images are stored at 224 and upsampled at load
time. Feeding 224 directly changes the token grid from 10x10 to 7x7 and the
position embeddings will not match.

**Normalization is ImageNet statistics**, not per-image. See
`src/common/data.py`.

## Verifying

`SHA256SUMS` is in the Hugging Face repository.

```bash
sha256sum -c SHA256SUMS
```
