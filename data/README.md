# Obtaining the data

No images are committed here. All three collections are public; download them
yourself and place them as shown below, then run the preprocessing steps.

## The three sources

**Kaggle — Brain Tumor MRI Dataset (Nickparvar)**
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Place under `datasets/kaggle/` with its `Training/` and `Testing/` folders.
7,200 images in our copy, balanced at 1,400 train and 400 test per class.
Note the released set is 7,023 and unbalanced; ours was rebalanced before use.

**Figshare — brain tumor dataset (Cheng et al.)**
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
DOI: 10.6084/m9.figshare.1512427
3,064 images, distributed as MATLAB `.mat` files. Run
`src/data_prep/step1_convert_figshare.py` to convert them to JPEG under
`datasets/figshare_converted/`.

**Mendeley Data — brain tumor collection**
https://data.mendeley.com/datasets/zwr4ntf94j/1
Place under `datasets/mendeley/` with its `Train/` and `Test/` folders.
12,064 images.

## A warning about the Mendeley collection

It is a public re-aggregation, not a single-institution release. Filename
analysis finds at least four provenances inside it:

| pattern | origin |
|---|---|
| `Tr-me_1234.jpg`, `Te-gl_*.jpg` | the Kaggle collection above |
| `41 no.jpg`, `44no.jpg` | the Chakrabarty no-tumor convention |
| `M_123_BR_.jpg`, `_DA_`, `_HF_`, `_RO_`, `_SP_`, `_VF_` | machine-generated augmented variants |
| `1107.jpg`, `m1 (1).jpg` | two further unlabelled sources |

In one class folder alone, 1,061 of 2,000 files carry Kaggle's naming.

This is why `step3_deduplicate.py` exists and why it removes 7,996 of the
12,064. **Do not skip it.** Without deduplication a leave-one-dataset-out split
holds out an institution that is already two thirds inside the training set,
and the resulting number is not a measure of transfer.

## Expected layout

```
datasets/
  kaggle/Training/{glioma,meningioma,notumor,pituitary}/
  kaggle/Testing/{glioma,meningioma,notumor,pituitary}/
  figshare/                     raw .mat files
  figshare_converted/           written by step1
  mendeley/Train/{glioma,meningioma,notumor,pituitary}/
  mendeley/Test/{glioma,meningioma,notumor,pituitary}/
```

## What preprocessing produces

| step | effect |
|---|---|
| step1 | converts Figshare `.mat` to JPEG |
| step2 | counts, dimensions, corrupt-file check |
| step3 | **removes 9,992 of 22,328** as cross-collection near-duplicates (256-bit pHash, Hamming distance 2, same-class guard) |
| step4 | resizes to 224 px, stratifies the Mendeley 80/20 split |
| step5 | builds the merged baseline split |

After step3: Kaggle 7,200 kept, Figshare 1,068 kept (1,996 removed),
Mendeley 4,068 kept (7,996 removed). Total 12,336.
