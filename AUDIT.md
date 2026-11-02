# Release audit — 2026-11-02

## Sources and retained code

This directory is an independent copy; the source project was not edited.
`SOURCE_MANIFEST.json` records SHA-256 hashes of the original source files and
the three seed-4 reference checkpoints used to establish effective settings.
It contains hashes and relative source paths only, without checkpoint data.

- `architecture/moe_stkim.py`, `tam.py`, `network.py`: original files retained
  byte for byte. GMEA is implemented inside `MoEACMIL.forward`, with
  expert-specific gated attention, gate-weighted auxiliary logits and
  gate-weighted slide representations. The historical names `MoEACMIL`,
  `Attention_Gated`, `Slide_classifier`, and parameter/module keys are kept.
- `engine.py`: training/evaluation functions extracted from
  `S4_WSI_classification__MoE_fstkim.py`. Training has the same Python AST.
  Evaluation removes only the unused `div_loss` gate statistic and its logging.
- `utils/utils.py`: required definitions extracted without semantic changes
  from `logs/reviewer_R1C2_expert_complementarity/model_source_snapshot/utils/utils.py`.
  This snapshot's SHA-256 is
  `89933dc885799cec914a25dcb2c6ea90262f3d63f3e874a6cc4abc3857c706fd`, matching
  `reviewer_experiments/component_controls_20261025/server_snapshot/plan.json`.
  The top-level project's utility source is absent, but the archived source
  resolves this gap. Unused imports/functions were removed. A dependency-free
  top-k accuracy helper preserves the original timm percentage convention.
- `data.py`: aggregate HDF5 loading, original HDF5 labels, BRACS label mapping,
  eager in-memory bags, coordinates, and per-item return values are preserved.
  Fixed CSVs replace original hard-coded JSON/metadata locations and random
  split fallback. It checks counts, uniqueness, split overlap, features and
  labels. No bags are sampled, normalized, padded, truncated or rebalanced.
  CAMELYON16/NSCLC preserve the supplied CSV's per-split order (which must match
  the original JSON list order for exact reruns); BRACS preserves the original
  HDF5 key iteration order within each split.
- `gemil.py`: single training/testing entry. Keeps batch size, AdamW,
  minibatch scheduler calls, losses, evaluation and checkpoint-selection rule.
  Adds functional `--eval-only`, strict checkpoint loading, configuration
  compatibility checks and explicit input/output paths. The original
  `--eval-only` flag was parsed but never acted upon.
- `configs/*.yaml`, `requirements.txt`, `README.md`, `splits/*.csv`,
  `.gitignore`: effective configs, runtime dependencies and usage instructions.

No baseline models, experimental entry points, ablation/sensitivity runners,
attention exporters, expert/gate analysis, historical logs, caches, original
WSIs, feature files, checkpoints or archives are distributed. The three exact
model source files retain their historical optional implementations for source
fidelity; the release config loader supports only the full four-expert model,
medical SSL, `n_expand=0`, and training-only stratified masking.

## Effective settings verified from reference checkpoints

Reference paths are
`logs/tensorboard/<dataset>_medical_ssl_<n_class>_moe_exp_fcloss_stkim_4/saved_models/checkpoint-best.pth`.

| Setting | CAMELYON16 | BRACS | TCGA-NSCLC |
|---|---|---|---|
| `dataset` | `camelyon` | `bracs` | `nsclc` |
| `n_class` | 2 | 3 | 2 |
| `D_feat` / `D_inner` | 384 / 128 | 384 / 128 | 384 / 128 |
| `n_token` / attention hidden width | 4 / 128 | 4 / 128 | 4 / 128 |
| `train_epoch` | 100 | 100 | 50 |
| `lr` / `wd` | 1e-4 / 1e-5 | 1e-4 / 1e-5 | 1e-4 / 1e-5 |
| classification criterion | CrossEntropyLoss | weighted FocalLoss, gamma=2 | CrossEntropyLoss |
| reference best epoch (zero-based) | 53 | 15 | 6 |
| original feature directory | `roi_feats` | `roi_feats_x10` | `roi_feats_norm` |

Common: B=1, seed=4, warmup_epoch=0, min_lr=0, n_worker=8,
pin_memory=false, n_shot=-1, stratified_group_ratios=[0.3,0.4,0.3],
stratified_mask_rates=[0.1,0.1,0.1], use_stratified_stkim=true,
n_masked_patch=0, mask_drop=0.6. TAM uses rank partitioning and uniformly
samples `floor(tier_size * mask_rate)` indices without replacement, replacing
their scores with -1e9 before softmax. It is disabled for validation/testing.
GMEA uses mean projected instance features for the slide gate. The projection
is bias-free Linear(384,128)+ReLU, without residual blocks or expansion.

The optimizer is constructed with lr=0.001 as in the original entry; the
per-minibatch scheduler sets it to 1e-4 before the first update and then applies
cosine decay. Focal alpha is `N_train / (n_class * class_count)` for present
classes, and 1.0 for absent classes. Loss is auxiliary classification plus
slide classification plus positive gate entropy with coefficient 1 and
epsilon=1e-8. Evaluation uses slide logits and classification loss only.
Accuracy is a percentage; AUROC uses torchmetrics' multiclass macro one-vs-rest
implementation, including two-class datasets. F1 is **micro** (the original
`torchmetrics.F1Score` wrapper default in version 1.1.0).

## Naming, manuscript and protocol issues kept visible

1. Current YAMLs in the source project contain n_token=8 and n_masked_patch=10;
   the original parser defaults overwrite them with 4 and 0. Checkpoint configs
   and state tensor shapes confirm 4/0. Released configs store effective values;
   optional CLI defaults no longer silently overwrite them.
2. The source BRACS YAML says 80 epochs, lr=1e-5 and no focal loss. The actual
   BRACS-537 seed-4 checkpoint and generated multiseed training config say
   100 epochs, lr=1e-4 and focal loss. The release follows the checkpoint.
3. `paper/sn-article.tex:545` describes focal loss generally, whereas the
   CAMELYON16 and NSCLC reference checkpoints explicitly disable it. No loss
   setting was changed to reconcile this statement.
4. The original entry selects `checkpoint-best.pth` using **test AUC + test F1**
   and evaluates the test set every epoch. This uses test labels for model
   selection. The same rule is retained, explicitly documented, and has not
   been silently replaced by validation selection. Later reviewer/ablation
   protocols use other selection rules and are excluded from this release.
5. The current local manuscript describes stochastic TAM masking and agrees
   with the selected configs. The source also offers a `range_shift` alternative;
   it is not used by these checkpoints and is not accepted by the release
   configs. No historical `tam_rho` has been invented.
6. `patch_gate_network` is registered but never called by the original forward.
   It is kept to preserve all checkpoint keys; its parameters receive no
   gradients. Only `gate_network` routes experts in the actual forward.
7. This is the seed-4 main-experiment configuration evidenced by the three
   checkpoints, not a claim that every historical run used identical settings.

## Data indices

Existing read-only snapshots inspected at
`reviewer_experiments/asmil_mergeslide_20261101/results_snapshot/splits/{camelyon,bracs,nsclc}.json`:

| Snapshot | Train | Val | Test | Duplicate IDs / cross-split overlap |
|---|---:|---:|---:|---:|
| CAMELYON16 | 242 | 27 | 129 | 0 |
| BRACS | 389 | 62 | 86 | 0 |
| TCGA-NSCLC | 885 | 46 | 120 | 0 |

BRACS' separate `bash_logs_bracs537_clean_multiseed_audit/bracs537_split_audit.csv`
also reports 389/62/86 for medical SSL. These checks establish local snapshot
counts and slide-ID disjointness, not patient independence or an identity match
to future files. The author subsequently supplied `GEMIL_WSI_Splits_v2.csv`. It was split by
dataset into `splits/camelyon.csv`, `splits/bracs.csv`, and `splits/nsclc.csv`,
preserving slide IDs, source order and split assignments. `wsi_id` was renamed
to `slide_id`; Train/Validation/Test were mapped to train/val/test. All outputs
passed the actual release CSV validator with the counts above and no duplicate
or overlapping slide IDs. Matching IDs to real HDF5 features remains unverified.

Final publication comparison: CAMELYON16 v2 matches the local snapshot's IDs,
assignments and order exactly. BRACS v2 moves BRACS_1614 and BRACS_1632 into
training and BRACS_1972 and BRACS_1973 into validation; the test set matches.
For NSCLC, relative to the snapshot, the supplied train/val/test sets contain
11/4/10 different IDs respectively, with an equal number removed from each.
The author-supplied v2 assignments are retained as requested. Counts and
slide-ID disjointness pass; this does not establish that the reference
checkpoints were trained on these v2 assignments or reproduce their accuracy.


## Runtime verification

Passed on Windows, Python 3.11.3, torch 2.0.0+cpu, numpy 1.26.0, h5py 3.9.0,
PyYAML 6.0.2, torchmetrics 1.1.0 and TensorBoard 2.13.0. Dependencies were
installed in a temporary virtual environment with system-site-packages disabled;
`python -m pip check` reported `No broken requirements found`.

Verification used a separate copy of this release outside the project directory.
The CLI subprocesses therefore did not rely on the source project's imports.
For each of the three dataset configs:

- Completed one synthetic epoch, backward propagation, AdamW updates, validation
  and test evaluation, best/last checkpoint saving, strict model reloading,
  optimizer state reloading and independent CLI testing.
- Used two or three small synthetic bags of 64 patches × 384 features, reused
  under distinct synthetic slide IDs to exercise the full required split counts.
  Only the temporary configs changed `train_epoch` to 1 and `n_worker` to 0;
  dimensions, experts, TAM, losses and optimizer settings were unchanged.
- Loaded the actual original seed-4 checkpoint, ran independent CLI evaluation
  with synthetic inputs, and confirmed all 39 state keys match exactly. Parameter
  counts are 184,086 (binary) and 184,731 (BRACS), including the unused patch gate.
- Compared original/release outputs in both train and eval modes with matched
  RNG states: maximum absolute difference **0**. Compared all parameter gradients
  for the original objective: maximum absolute difference **0**, including
  agreement on parameters that receive no gradients.
- Confirmed reloaded predictions equal predictions before serialization,
  gradients/losses/metrics are finite, the two CSV interfaces agree, and bad split
  counts and repeated/overlapping IDs are rejected.

The three core source files are byte-identical; training and all eight extracted
utility definitions are AST-identical to their verified sources. Accuracy was
also checked against the function in the original pinned timm 0.9.2 distribution
for several batch sizes and top-k values; outputs agree exactly. Focal loss and
cosine scheduler results agree exactly with the archived original definitions.
Final source-hash checks passed. No synthetic files, test scripts, generated
checkpoints, events, caches or environment files remain in this release.

These tests verify implementation/serialization behavior, not real-data accuracy.

## Public repository preparation

The owner confirmed MIT licensing for original GEMIL contributions and the
software authors, in this order: Jingjun Gu and Yijing Liu. `LICENSE` and
`CITATION.cff` record these choices. Author affiliations and Jingjun Gu's ORCID
are supplied by the owner; the ORCID checksum is valid. Yijing Liu's ORCID is
not supplied and is omitted. Professional email addresses are not included.
`.zenodo.json` supplies the same author order, software version 1.0.0 and MIT
license, plus the owner's funding acknowledgement and corresponding-author
designation. Grant numbers are recorded as text, without inventing external
funder/project registry identifiers. Zenodo gives this JSON metadata precedence
over CITATION.cff; the shared fields are checked for consistency.

Required upstream MIT notices for the identified ACMIL/DTFD-MIL portions are
preserved in `THIRD_PARTY_NOTICES.md`. Lunit pretrained weights are not included.
The owner confirmed that the GitHub repository is enabled in Zenodo. The active
GitHub release webhook and a successful Zenodo ping response were verified
before preparing the first formal release, v1.0.0.

## Remaining items

- Author-provided split CSVs were added and validated on 2026-11-02.
- No real feature HDF5 files were found in the project. Therefore real-feature
  prediction/performance reproduction cannot be certified locally.
- CUDA execution is not validated on this CPU-only machine.
