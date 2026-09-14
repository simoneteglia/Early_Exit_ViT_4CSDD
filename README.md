# Early_Exit_ViT_4CSDD

Early-exit Vision Transformer for deepfake detection on the
[CSS deepfake dataset](https://huggingface.co/datasets/simoneteglia/css-deepfake-dataset),
where the exit decision is shaped by human-centered factors — content
sensitivity, user susceptibility to misinformation, policy — on top of the
prediction confidence. The early-exit machinery and the interactive
threshold explorer are ports of ETA-DyNN's `ee_cnn` and `activity_emulator`
(see `../DESIGN.md` for the full design).

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # the dataset is gated; or export HF_TOKEN=...
```

## Pipeline

All scripts run from this directory. Every script accepts `--synthetic N`
to use N random images instead of the dataset (smoke tests).

1. **Train** the early-exit ViT (pretrained **DINOv3** ViT-S/16 backbone by default —
   `--backbone vit_base_patch16_dinov3.lvd1689m` for ViT-B — side exits after
   blocks 4 and 8 of 12, joint multi-exit BCE loss). The split manifest
   (stratified on label × sensitivity tier, 70/15/15) is created on first use.
   ```bash
   python train.py --manifest manifests/splits.json --out-dir runs/ee_vit --epochs 10 --batch-size 64 --amp
   python train.py --manifest manifests/splits.json --out-dir runs/full_vit --disable-ee   # plain-ViT baseline
   ```
2. **Calibrate**: per-exit beta calibration on the validation split, base
   `(lower, upper)` thresholds per side exit, and `scores.pkl` (the reference
   distributions the explorer shows).
   ```bash
   python calibrate.py --checkpoint runs/ee_vit/best.pt --manifest manifests/splits.json
   ```
3. **Extract** test-set scores and per-exit costs into `experiments.db`.
   ```bash
   python extract.py --checkpoint runs/ee_vit/best.pt --manifest manifests/splits.json \
       --db experiments.db --experiment exp_1 --dataset-codename css_test
   ```
4. **Explore** thresholds interactively (screen 1 of the ETA-DyNN emulator,
   generalised to N exits, with sliders for user susceptibility and policy
   strictness, margin coefficients, and a content-sensitivity tier filter).
   ```bash
   python emulator/main.py --scores runs/ee_vit/scores.pkl --db experiments.db \
       --experiment exp_1 --dataset css_test --thresholds runs/ee_vit/thresholds.json
   ```

## Modules

| File | Role |
|---|---|
| `vit.py` | Original from-scratch ViT (kept for reference) |
| `ee_vit.py` | `EarlyExitViT` on a timm backbone (DINOv3 `Eva` or classic `VisionTransformer`), `multi_exit_loss`, per-sample early-exit `infer` |
| `data.py` | HF dataset loading, stratified split manifests, transforms, datasets/loaders |
| `human_factors.py` | Threshold policy `m = clip(m0 + α·s_c + β·s_u + γ·p)` widening the abstention band; MIST loading |
| `calibration.py` | `Calibrator` (beta / isotonic), threshold objective and grid search |
| `metrics.py` | Per-exit metrics, ECE, score collector |
| `experiments_db.py` | SQLite schema (`ee_vit` / `full_vit` runs, samples with CSS score, per-phase costs) |
| `emulator/` | PyQt5 + matplotlib threshold explorer |

## Threshold semantics

At side exit *e* with calibrated score `s = P(fake)` and effective thresholds
`lower_e' = lower_e·(1−m)`, `upper_e' = upper_e + (1−upper_e)·m`:
`s < lower_e'` → answer *real*, `s > upper_e'` → answer *fake*, otherwise
continue. The final head always answers (`s > 0.5`). `m = 0` is confidence-only
early exit; `m → 1` forces full depth.

User susceptibility `s_u` comes from a MIST score (Maertens et al., 2023,
OSF `r7phc`): `human_factors.user_sensitivity(mist, reference_population)`
returns `1 − percentile`, so better discernment means a lower margin.
