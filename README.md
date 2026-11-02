# GEMIL

## Environment

Use Python 3.10 or 3.11. Run the following commands inside `GEMIL-release`:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data preparation

Obtain the slides from [CAMELYON16](https://camelyon16.grand-challenge.org/),
[BRACS](https://www.bracs.icar.cnr.it/), and [GDC](https://portal.gdc.cancer.gov/)
(TCGA-LUAD and TCGA-LUSC). Only **medical_ssl** features are supported:
[Lunit DINO ViT-S/16](https://github.com/lunit-io/benchmark-ssl-pathology),
`dino_vit_small_patch16_ep200.torch`, 384 dimensions per patch, projected to
128 dimensions by GEMIL. The original experiments used non-overlapping
256×256 patches; BRACS used 10× features, and TCGA-NSCLC used features extracted
after stain normalization. Prepare the original pre-extracted features; feature
extraction and stain normalization are outside this release.

For each dataset, provide one aggregate HDF5 file with a group per slide ID:
`/<slide_id>/feat` (finite float array `[N,384]`), `/<slide_id>/coords` (`[N,2]`),
and group attribute `label` (integer). CAMELYON16 labels are normal=0/tumor=1;
NSCLC labels are LUAD=0/LUSC=1. BRACS requires the original seven-type labels
0–6, mapped by the loader as `{0,1,2}→0`, `{3,4}→1`, `{5,6}→2`.

The author's supplied `GEMIL_WSI_Splits_v2.csv` is split into
`splits/camelyon.csv`, `splits/bracs.csv`, and `splits/nsclc.csv`.
Slide IDs and source order are preserved; `wsi_id` is renamed to `slide_id`
and Train/Validation/Test are mapped to train/val/test.
Accepted columns: `slide_id,split` (split values `train`, `val`, `test`;
`split_info` is also accepted), or three columns `train,val,test` containing
slide IDs with blank cells for unequal lengths. Labels are read from HDF5.
The loader checks counts, duplicate IDs, split overlaps and feature availability;
it never generates a new split.

| Config | Train | Val | Test |
|---|---:|---:|---:|
| `configs/camelyon.yaml` | 242 | 27 | 129 |
| `configs/bracs.yaml` | 389 | 62 | 86 |
| `configs/nsclc.yaml` | 885 | 46 | 120 |

## Training

Replace the HDF5 paths with your local paths. Output directories must be new.
`--seed` overrides the config seed; it does not change the supplied split.

```bash
python gemil.py --config configs/camelyon.yaml --features /path/to/camelyon_medical_ssl.h5 --splits splits/camelyon.csv --output runs/camelyon --seed 4
python gemil.py --config configs/bracs.yaml --features /path/to/bracs_medical_ssl.h5 --splits splits/bracs.csv --output runs/bracs --seed 4
python gemil.py --config configs/nsclc.yaml --features /path/to/nsclc_medical_ssl_norm.h5 --splits splits/nsclc.csv --output runs/nsclc --seed 4
```

Configs preserve the effective seed-4 checkpoint settings. Training writes
`checkpoint-best.pth`, `checkpoint-last.pth`, config, metrics and TensorBoard
events. The historical best-checkpoint rule uses **test AUC + test F1** and is
preserved for consistency (see `AUDIT.md`). `--device cpu` forces CPU execution;
the default is automatic CUDA/CPU selection. Set `n_worker: 0` in the config
when multiprocessing is unavailable.

## Testing

```bash
python gemil.py --config configs/camelyon.yaml --features /path/to/camelyon_medical_ssl.h5 --splits splits/camelyon.csv --eval-only --checkpoint runs/camelyon/checkpoint-best.pth
python gemil.py --config configs/bracs.yaml --features /path/to/bracs_medical_ssl.h5 --splits splits/bracs.csv --eval-only --checkpoint runs/bracs/checkpoint-best.pth
python gemil.py --config configs/nsclc.yaml --features /path/to/nsclc_medical_ssl_norm.h5 --splits splits/nsclc.csv --eval-only --checkpoint runs/nsclc/checkpoint-best.pth
```

Testing reports validation/test AUC, accuracy (percent), F1 and classification
loss. It loads both model weights and their config compatibility checks; it
requires the same dataset config and HDF5 labels as training. Original project
checkpoints with a `model` state dictionary are supported. WSI data, features
and checkpoints are not distributed here.
