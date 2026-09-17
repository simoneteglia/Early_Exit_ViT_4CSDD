# Early_Exit_ViT_4CSDD

Early-exit Vision Transformer for deepfake detection on the
[CSS deepfake dataset](https://huggingface.co/datasets/simoneteglia/css-deepfake-dataset),
where the decision to stop at an intermediate exit depends not only on the
prediction confidence but on **human-centered factors**: the sensitivity of
the content, the susceptibility of the user to misinformation, and the
policy in force. The early-exit machinery and the interactive tool are ports
of ETA-DyNN's `ee_cnn` and `activity_emulator` (bees on an energy-harvesting
edge device), re-targeted at mis/disinformation on social networks. The
design rationale is in `../DESIGN.md`.

The repository contains two things:

1. a **training / calibration / extraction pipeline** that produces, for a
   test set, the calibrated per-exit scores and per-exit inference costs of
   the early-exit ViT (stored in `experiments.db`);
2. an **interactive explorer** with two screens: *threshold exploration*
   (how the human-centered thresholds shape accuracy, exit usage and cost) and
   *propagation simulation* (how the detector changes the spread of fake and
   real content in a social network).

---

## 1. Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # the dataset is gated; or export HF_TOKEN=...
```

Python ≥ 3.10. GPU recommended for training; the explorer runs anywhere
(PyQt5 + matplotlib). All commands below run from this directory.

## 2. Model

`EarlyExitViT` ([ee_vit.py](ee_vit.py)) wraps a pretrained timm ViT —
**DINOv3 ViT-S/16** by default (`vit_small_patch16_dinov3.lvd1689m`; use
`--backbone vit_base_patch16_dinov3.lvd1689m` for ViT-B, or any DeiT/ViT) —
and attaches **side exits** after intermediate encoder blocks (after blocks 4
and 8 of 12 by default; `--exit-points`). Each exit is a LayerNorm + Linear
head on the pooled tokens (CLS + mean patch tokens for DINO-style backbones,
CLS for DeiT) producing one logit; `sigmoid(logit) = P(fake)`. The exits and
the final head are trained **jointly** with a weighted sum of BCE losses,
as in ETA-DyNN's `Joint_EE_MobileNetV3`.

At inference (`EarlyExitViT.infer`) a sample walks the exits in order; at
each side exit its calibrated score is compared with that exit's thresholds:
`score < lower` → answer *real*, `score > upper` → answer *fake*, otherwise
continue to the next block. Samples that receive an answer are dropped from
the batch, so early exiting saves real compute. The final head always
answers (`score > 0.5`).

## 3. Pipeline

Every script accepts `--synthetic N` to use N random images instead of the
dataset (smoke tests), and `--max-per-class N` for a balanced subsample.

**Step 1 — Train.** Creates the split manifest on first use (70/15/15,
stratified on label × content-sensitivity tier; manifests are versioned in
`manifests/`). The dataset's third class ("tampered", `label_idx = 2`) is
filtered out — the detector is binary real/fake.

```bash
python train.py --manifest manifests/splits.json --out-dir runs/ee_vit --epochs 10 --batch-size 64 --amp
python train.py --manifest manifests/splits.json --out-dir runs/full_vit --disable-ee   # plain-ViT baseline
```

**Step 2 — Calibrate.** Fits one beta calibrator per exit on the validation
split (as in ETA-DyNN), searches base `(lower, upper)` thresholds per side
exit with ETA-DyNN's objective (answer quality blended with the fraction of
samples answered), and writes `scores.pkl` (the reference score
distributions shown on screen 1), `thresholds.json`, `calibrators.pkl` and a
calibration report (accuracy, AUC and ECE before/after calibration).

```bash
python calibrate.py --checkpoint runs/ee_vit/best.pt --manifest manifests/splits.json
```

**Step 3 — Extract.** Runs the test split, stores per-sample calibrated
scores of every exit in `experiments.db` (`ee_vit` rows with `exit_idx`, and
`full_vit` rows for the final head), together with each sample's content
sensitivity, tier and scenario, and the **cumulative inference cost of
reaching each exit** (duration, and CPU/GPU/RAM energy via codecarbon),
measured with `force_exit` exactly as ETA-DyNN measures per-stage cost.

```bash
python extract.py --checkpoint runs/ee_vit/best.pt --manifest manifests/splits.json \
    --db experiments.db --experiment exp_1 --dataset-codename css_test
```

**Step 4 — Explore** (the tool, described in §4):

```bash
python emulator/main.py --scores runs/ee_vit/scores.pkl --db experiments.db \
    --experiment exp_1 --dataset css_test --thresholds runs/ee_vit/thresholds.json
```

Options: `--reference-split {train,val,test}` (distribution shown in the
histograms; default train), `--policy file.json` (initial margin
coefficients, e.g. `{"m0": 0, "alpha": 0.5, "beta": 0.3, "gamma": 0.2, "m_max": 1}`).

**Step 5 — Simulate headlessly** (same engine as screen 2, for batch
experiments; writes `<out>_results.json` and `<out>_trends.png`):

```bash
python simulate.py --db experiments.db --experiment exp_1 --dataset css_test \
    --thresholds runs/ee_vit/thresholds.json --items 60 --out results/sim
```

---

## 4. The explorer

The explorer is a two-screen PyQt5 application. Screen 1 is where an expert
or policy maker sets *how the detector decides*; screen 2 shows *what that
decision does* to the spread of content in a network. "Proceed to
simulation" carries the screen-1 settings into screen 2; "Back to
thresholds" returns.

### 4.1 The human-centered exit rule (shared by both screens)

Three factors set a **margin** `m ∈ [0, 1]`:

```
m = clip( m0 + α·s_c + β·s_u + γ·p ,  0 ,  m_max )
```

- `s_c` — content sensitivity of the item, the dataset's CSS score (1–4)
  normalised to [0, 1];
- `s_u` — susceptibility of the user to misinformation, in [0, 1], derived
  from a MIST score (Maertens et al., 2023; OSF `r7phc`) as `1 − percentile`
  within the reference population — better discernment means lower `s_u`;
- `p` — policy strictness in [0, 1] (e.g. 0 normally, close to 1 during an
  election);
- `m0, α, β, γ, m_max` — coefficients set by the expert.

The margin **widens the abstention band** of every side exit around the base
thresholds found by calibration:

```
lower' = lower · (1 − m)          upper' = upper + (1 − upper) · m
```

so `m = 0` is confidence-only early exit and `m → 1` forces full depth: the
more sensitive the content, the more susceptible the user or the stricter
the policy, the more confidence an early exit must show before it is allowed
to answer. The fourth factor, the **prediction confidence**, is what gets
compared with the widened band. Because `s_c` is a property of each item,
thresholds are per sample; because `s_u` is a property of each user, the
same item can be answered by different exits for different users.

### 4.2 Screen 1 — threshold exploration

A replica of ETA-DyNN's first emulator screen, generalised to *N* side
exits and extended with the human factors.

**Control box "Human-centered factors"** (top):

- *Content sensitivity tier* — filters the whole screen to items of one
  tier (`1-NONE`, `2-LOW`, `3-MODERATE`, `4-HIGH`, one per CSS level) or
  `ALL`; every plot and table below is recomputed on that subset.
- *Source dataset* — filters by the source the images come from (the prefix
  of the sample id). Useful because in the CSS dataset real and fake images
  largely come from different sources; only the mixed source gives a
  shortcut-free estimate.
- *User susceptibility `s_u`* and *Policy strictness `p`* sliders — the
  user and policy factors for the whole screen (one user at a time).
- *Margin coefficients* `m0, α, β, γ, m_max` (tooltips give their meaning).
- A status line with the resulting margin statistics on the selected samples.

Any change recomputes everything live.

**One panel per side exit** ("Model: ee_vit / Exit: e"), left to right:

- *Histogram* of the calibrated `P(fake)` on the reference split, with the
  **blue (lower)** and **red (upper)** base thresholds as draggable lines and
  a **grey band** showing the effective, widened band at the current mean
  margin.
- *Training confusion matrix and metrics* — on the reference samples this
  exit would answer; "Valid answers" is the fraction it answers.
- *Testing confusion matrix and metrics* — on the test samples actually
  answered at this exit in the waterfall (not already answered by an earlier
  exit); "Answered here" is this exit's share, "Percentage returned" is
  cumulative.

**Global Results** panel:

- *Inference costs* table: summed inference cost of the early-exit system
  (each sample charged the cumulative cost of the exit that answered it) vs.
  the always-full-depth baseline, and the percentage saved — duration and
  total/CPU/GPU/RAM energy. Preprocessing is identical for both and is
  excluded.
- *Confusion matrix and metrics* of the whole system's answers.
- *Exit distribution*: share of samples answered by each exit and by the
  final head — the plot to watch while moving the sliders.

**Buttons**: *Reset base thresholds* (back to `thresholds.json`), *Compare
thresholds performance* (grid search over the base thresholds under the
current human-factor setting; writes `thresholds_search_data.pkl` with
thresholds, energy saving and accuracy per combination), *Proceed to
simulation*.

### 4.3 Screen 2 — propagation simulation

The analogue of ETA-DyNN's activity emulator: instead of bees arriving at a
hive under an energy budget, content items spread in a social network with
the detector in the loop. The question it answers is how much
mis/disinformation reach the detector removes, at what inference cost, and
with what collateral damage to legitimate content.

**Model** ([propagation.py](propagation.py)):

- *Network*: a Barabási–Albert graph (scale-free, so a few high-degree
  users act as influencers). Each user carries a susceptibility `s_u`
  sampled from the MIST population (the OSF data file when given; otherwise a
  stand-in Normal(13, 3.5) on the 0–20 scale, converted to `1 − percentile`).
- *Content*: each test item (with its recorded per-exit scores and costs) is
  one **Independent-Cascades** run (NDlib). A user who shares the item gets
  one chance to pass it to each neighbour, who shares it with probability

  ```
  P(v shares) = p_share · g(s_u(v), label) · (1 − w · flagged(item, v))
  ```

  where `g = 1 + κ(2·s_u − 1)` for fake items (susceptible users share fakes
  more) and 1 for real items.
- *Detector*: runs once per (item, exposed user) with that user's `s_u`, the
  item's `s_c` and the policy `p` — i.e. exactly the screen-1 rule — using the
  scores stored in `experiments.db`. A "fake" verdict applies the
  **intervention** `w`: a warning that multiplies the share probability by
  `(1 − w)`, or a block when `w = 1`. It applies identically to false
  positives, which is how the suppression of real content enters.
- *Cost*: each detector run is charged the inference cost of the exit that
  answered; savings are reported relative to the full-ViT scenario (cost is
  reported, not budgeted).
- *Scenarios*, run on the same seeds and random streams per item so that
  differences are due to the detector alone: **No detector**, **Full ViT**
  (final head for everyone), **Early exit (confidence)** (`m = 0`), **Early
  exit (human-centered)** (screen-1 margin).
- *Assumptions*: items are independent; a user shares an item at most once;
  the network is static; susceptibility is independent of network position;
  no belief/correction dynamics beyond the single-shot sharing rule.

**Layout**:

- *Header*: the thresholds, coefficients and policy inherited from screen 1
  (the screen-1 user slider is not used — every node has its own `s_u`).
- *Propagation settings* (left): users and BA edges per node; `p_share`;
  susceptibility effect `κ`; intervention `w`; seeds per item and seed
  strategy (random users or top-degree hubs); number of test items (balanced
  real/fake, spread over tiers); max steps; random seed; optional MIST file.
  *Run simulation* runs in the background with a progress bar.
- *Network* (top right): one cascade at a time, chosen with the **Item** and
  **Scenario** selectors and scrubbed with the **Step** slider. Under the
  selectors, a line compares the selected item across the four scenarios
  (users reached, steps, flagged or not), and says so explicitly when the
  scenarios are identical — which is the expected outcome for an item the
  detector never flags. Node colours: grey = unaware, orange = sharing now,
  red = already shared; node size = degree; green ring = seed; **black ring =
  users for whom the detector flagged the item** (the warning at work for a
  fake, a false positive for a real item). Mouse wheel zooms, left-drag pans,
  the toolbar offers pan/zoom-box/Home/save, and hovering a node shows its
  id, degree, `s_u`, state and the detector's decision for that user (which
  exit answered, and its verdict). The view is kept while scrubbing steps or
  switching scenario.
- *Reach curves* (bottom left): mean cumulative reach over items, per
  scenario, for fake items (the benefit) and real items (curves below "no
  detector" reveal suppression by false positives).
- *Results table* (bottom right), one row per scenario: fake reach and its
  reduction vs. no detector; real reach and its loss; detector runs (one per
  exposed user); inference time and energy with savings vs. the full ViT;
  share of runs answered by each exit.

Reading the table as an experiment: fake reduction is the benefit, real loss
the harm, inference cost the price. Rows 3 vs. 4 isolate the human-centered
margin; rows 2 vs. 3–4 show the accuracy/cost trade-off of early exiting.

---

## 5. Data notes

- The CSS dataset's images come from several **source datasets**, most of
  which contain only real or only fake images; only `rrdataset` is mixed. A
  detector can therefore partly recognise the source instead of fakeness —
  evaluate on the mixed source (screen-1 *Source dataset* filter, or a
  held-out-source split) for an honest number.
- Content-sensitivity tiers are derived from the 1–4 CSS score (one tier per
  level); tier and label are correlated in the data, so per-tier metrics
  partly reflect class balance.
- Exit scores are strongly bimodal (most samples get `P(fake)` near 0 or 1,
  and those are almost always right), so the margin mechanism mainly acts on
  the minority of mid-range samples. A depth floor driven by `m` is the
  candidate extension if human factors must also override confident early
  answers.

## 6. Modules

| File | Role |
|---|---|
| `vit.py` | Original from-scratch ViT (kept for reference) |
| `ee_vit.py` | `EarlyExitViT` on a timm backbone (DINOv3 `Eva` or classic `VisionTransformer`), `multi_exit_loss`, per-sample early-exit `infer` |
| `data.py` | HF dataset loading, stratified split manifests, transforms, datasets/loaders |
| `human_factors.py` | Threshold policy `m = clip(m0 + α·s_c + β·s_u + γ·p)` widening the abstention band; CSS tiers; MIST loading |
| `calibration.py` | `Calibrator` (beta / isotonic), threshold objective and grid search |
| `metrics.py` | Per-exit metrics, ECE, score collector |
| `experiments_db.py` | SQLite schema (`ee_vit` / `full_vit` runs, samples with CSS score, per-phase costs) |
| `train.py`, `calibrate.py`, `extract.py` | Pipeline steps 1–3 |
| `propagation.py` | NDlib Independent-Cascades propagation with the detector in the loop; scenarios, cost accounting |
| `simulate.py` | Headless propagation experiment (JSON results + reach curves) |
| `plot_history.py` | Training-history dashboard from `history.json` |
| `emulator/main.py` | Explorer entry point (screen 1 + screen 2) |
| `emulator/thresholds_window.py`, `confidence_plot.py`, `summary_plot.py` | Screen 1 |
| `emulator/simulation_window.py` | Screen 2 |
