# Pianoid: 3D Pianist Body Motion Generation from MIDI

**Joonhyung Bae, Juhan Nam**
Graduate School of Culture Technology, KAIST, Daejeon, Republic of Korea

[Project page](https://joonhyungbae.github.io/pianoid/) ·
[Paper (PDF)](https://joonhyungbae.github.io/pianoid/assets/pianoid_ismir2026.pdf) ·
[Video](https://youtu.be/GaFwqai4PsE) ·
[ISMIR 2026 page](https://ismir2026program.ismir.net/poster_287.html) ·
[Checkpoints](https://github.com/joonhyungbae/pianoid/releases/latest)

Official code for *Pianoid: 3D Pianist Body Motion Generation from MIDI*
(ISMIR 2026, poster P10-8). Pianoid is a four-stage hierarchical pipeline that
generates 3D 26-joint pianist motion from symbolic MIDI.

[![Pianoid video](https://img.youtube.com/vi/GaFwqai4PsE/hqdefault.jpg)](https://youtu.be/GaFwqai4PsE)

## Quickstart (reproduce the paper's headline numbers)

Five steps. Total time ≈ 30–40 min on one GPU once MOSA is in hand. No training required.

```bash
# 1. Environment (creates conda env `pianoid` and activates it)
bash setup.sh && conda activate pianoid

# 2. Download MOSA mocap (≈ 40 GB) from Zenodo and unpack into MOSA_dataset/
#    https://zenodo.org/records/11393449 (CC-BY-4.0, request-access flow on
#    Zenodo) → result: MOSA_dataset/yp/<piece>/<performer>/t<trial>/<...>_motion.csv
#    (14 pieces × 8 performers, 272 sequences total)

# 3. Preprocess (one-time, ≈ 15 min after MOSA is on disk)
python scripts/preprocess/build_splits.py
python src/stage1/s05_preprocess_motion.py
python src/stage1/s12_precompute_training_data.py --onset_window 0.5 --output_dir MOSA_dataset/training_cache_W05
python src/stage1/s12_precompute_training_data.py --onset_window 1.0 --output_dir MOSA_dataset/training_cache_W10
python src/stage1/s12_precompute_training_data.py --onset_window 2.0 --output_dir MOSA_dataset/training_cache_B

# 4. Download released checkpoints (≈ 188 MB, unpacks to checkpoints/release/)
bash scripts/download_checkpoints.sh

# 5. Reproduce the paper (~10–20 min on a single GPU)
bash inference.sh
```

To train every stage from scratch instead of downloading checkpoints, swap step
4 for `bash train.sh` (≈ 3–5 days on a single GPU); step 5 still works.

## What this repo reproduces

`bash inference.sh` regenerates every paper headline number listed below from
the released checkpoints. The "Reproduced" column is from a fresh run on a
single A6000; the bootstrap CI (10 k resamples, seed 42) is reported by
[`src/evaluation/compute_full_pipeline_er.py`](src/evaluation/compute_full_pipeline_er.py).
Run-to-run variation comes from DDIM sampling stochasticity at Stage 2; every
paper number falls inside its own reproduction's 95% CI.

| Paper item | Paper value | Reproduced (point, 95% CI) | Output file |
|---|---|---|---|
| Table 2 — MPJPE  | 92.1 mm | **92.15 mm** [88.95, 96.54] | `results/full_pipeline_er.json` (also `results/full_pipeline.json`) |
| Table 2 — ER-range | 1.12 | **1.134** [1.082, 1.188] | `results/full_pipeline_er.json` |
| Table 2 — ER-speed | 0.92 | **0.938** [0.868, 1.017] | `results/full_pipeline_er.json` |
| Table 2 — ER-jerk  | 0.34 | **0.335** [0.311, 0.361] | `results/full_pipeline_er.json` |
| Table 2 — WZR      | 0.76 | **0.762** [0.710, 0.820] | `results/full_pipeline_er.json` |
| Table 2 — AccR     | 0.44 | **0.440** [0.413, 0.468] | `results/full_pipeline_er.json` |
| Table 2 — **Δ₁**   | **0.42** | **0.428** [0.394, 0.465] | `results/full_pipeline_er.json` |
| Table 2 — FID, Jitter, VelCorr | (paper text) | FID 0.028, \|ΔJ\| 23.95×10⁻⁵, VelCorr 0.107 | `results/full_pipeline.json` |
| Table 4 — per-bodypart MPJPE | (paper Fig. 4 / Table 4) | matches within ≤ 1 mm | `experiments/benchmark/bodypart_metrics.json` |
| §3 F1–F6 + Table 1 — inter-body-part correlations, opposing-dynamics, FDR | (paper §3) | matches | `results/music_body_analysis.json`, `results/expressiveness_analysis.json`, `results/delta1_axis_correlation_and_perf.json`, `experiments/analysis/body_decomposition_analysis.json` |

### Beyond the headline (baselines + ablations)

Following standard ML release practice, only Pianoid's own checkpoints are
distributed — baseline checkpoints (Li2018, MDM-adapted, EDGE-adapted,
Single-Stage MDM+zs, BiLSTM-backbone) are **not** redistributed, but every
adaptation is implemented in this repo and trainable end-to-end:

- **Table 2 baseline rows** — `bash scripts/baselines.sh all` trains all four
  Table 2 baselines under the paper's Table 5 (`tab:hyperparams`) protocol;
  `python experiments/benchmark/evaluate_all.py` then writes one JSON row per
  baseline to `experiments/benchmark/unified_comparison.json`.
- **Table 3 ablations (A–D)** — per-block scripts under
  [`scripts/ablation/`](scripts/ablation/) (see the
  [Optional sub-experiments](#optional-sub-experiments) table for the exact
  command per block).

Both paths reuse the same dataset + Stage 1–4 code as the headline, so any
discrepancy reviewers find is on real training, not on protocol drift.

## Pipeline

| Stage | Input → Output | Model | Released checkpoint |
|-------|---------------|-------|---------------------|
| S1 | MIDI → wrists (2 joints) | BiLSTM + Z-axis log-ratio loss | `checkpoints/release/stage1_wrist/model.pt` |
| S2 | wrists → backbone (10 joints: head, torso, shoulders) | MDM + z-score, DDIM-45 | `checkpoints/release/stage2_backbone/model.pt` |
| S3 | wrists + chest + shoulders → elbows (2 joints) | BiLSTM, shoulder-relative, trained on predicted shoulders | `checkpoints/release/stage3_elbow/model.pt` |
| S4 | spine → lower body (10 joints) | FiLM (spine-only) | `checkpoints/release/stage4_lower/model.pt` |

End-to-end inference on the MOSA test split is wrapped by [inference.sh](inference.sh) (Pipeline → [src/pipeline/full_pipeline.py](src/pipeline/full_pipeline.py)). Per-stage training scripts that produce these checkpoints live under [scripts/train/](scripts/train/) and are orchestrated by [train.sh](train.sh).

## Installation

```bash
bash setup.sh            # creates conda env `pianoid` (Python 3.11)
conda activate pianoid
```

ARIA is **optional**. The final Pianoid model and all paper headline numbers do
not use ARIA; it is only needed to reproduce the Stage 4 `FiLM + ARIA` ablation
row in Table 3(D). Enable it with:

```bash
INSTALL_ARIA=1 bash setup.sh
```

## Dataset

[MOSA](https://zenodo.org/records/11393449): 26-joint mocap at 60 fps, 8 performers (yp01–yp08), 272 sequences, Henle grades 3–9. MIDI is reconstructed from note annotations (per-note velocity not available).

After unpacking the Zenodo archive so that
`MOSA_dataset/yp/<piece>/<performer>/t<trial>/<...>_motion.csv` exists,
run the preprocessing block from the Quickstart. The expected layout
post-preprocessing is `MOSA_dataset/{splits.json, yp/ (raw),
preprocessed/, training_cache_W05/, training_cache_W10/, training_cache_B/}`.
Each stage script's `--cache_dir` defaults to the relevant variant. Paper split
(158/34/80) details: train = yp01–yp05 (except yp04/yp05 on pieces {be1, tc1});
val = yp06; test = yp07 + yp08 + yp04/yp05 on {be1, tc1}.

## Released checkpoints

Trained checkpoints that reproduce the paper numbers are attached to the
[v1.0.0 GitHub Release](https://github.com/joonhyungbae/pianoid/releases/tag/v1.0.0)
as a single ~188 MB zip. The Quickstart's
[`scripts/download_checkpoints.sh`](scripts/download_checkpoints.sh) fetches it,
unpacks under `checkpoints/release/`, and verifies SHA256:

```
checkpoints/release/
├── stage1_wrist/     model.pt (139 MB) — BiLSTM h512 l6, v3 loss
├── stage2_backbone/  model.pt (30 MB)  — MDM zscore h256 l8, DDIM 45 at inference
├── stage3_elbow/     model.pt (35 MB)  — BiLSTM h256 l6, predicted-train
├── stage4_lower/     model.pt (0.25 MB)— FiLM h128 l4, spine-only
└── SHA256SUMS.txt
```

Each `model.pt` is a dict loadable with `torch.load(..., weights_only=False)`;
keys: `model_state_dict`, `config`, `val_mpjpe_mm`, `epoch`, plus `norm_stats`
for Stage 2. (Re-organize freshly trained checkpoints into this layout with
`python src/project_utils/release_checkpoints.py --mode copy`.)

## Reproducing the paper

Two top-level entry points cover the entire paper. `inference.sh` requires the
released checkpoints (one-shot download via `scripts/download_checkpoints.sh`);
`train.sh` produces them from scratch.

```bash
bash inference.sh    # Reproduce paper numbers (no training, ~10–20 min)
bash train.sh        # Train everything from scratch (~3–5 days)
```

`inference.sh` runs:
1. **§3 analysis** (F1–F6 correlations, Table 1) — five scripts under `experiments/analysis/`.
2. **§5.2 Pianoid full pipeline** — `src/pipeline/full_pipeline.py` with DDIM 45 → `results/full_pipeline.json` (Table 2 row "Pianoid": MPJPE 92.1, ER-rng 1.12, ER-spd 0.92, ER-jrk 0.34, WZR 0.76, AccR 0.44, Δ₁ 0.42).
3. **§5.2 Table 4** per-bodypart MPJPE — `experiments/benchmark/compute_bodypart_metrics.py`.

`train.sh` runs Stage 1 → 2 → 3 → 4 in order, then Table 2 baselines (Li2018, MDM-adapted, EDGE-adapted). After completion, `bash inference.sh` regenerates the headline numbers from the freshly trained checkpoints.

### Optional sub-experiments

These are not part of the headline (`inference.sh` does not call them) and require their own training runs.

| Paper item | Run | Output |
|---|---|---|
| §5.2 Table 2 baselines (separately) | [`bash scripts/baselines.sh all`](scripts/baselines.sh) → [`python experiments/benchmark/evaluate_all.py`](experiments/benchmark/evaluate_all.py) | `experiments/benchmark/unified_comparison.json` |
| §5.3 Table 3(A) Stage 2 ablation (regression vs. diffusion, DDIM sweep) | [`bash scripts/ablation/stage2_monolithic.sh`](scripts/ablation/stage2_monolithic.sh), [`bash scripts/train/stage2.sh sweep`](scripts/train/stage2.sh) | `results/stage2_*.json` |
| §5.3 Table 3(B) Stage 3 ablation (anatomical / predicted-train 2×2 + MDM) | [`bash scripts/ablation/stage3_rows.sh`](scripts/ablation/stage3_rows.sh) | `results/stage3_*.json` |
| §5.3 Table 3(C) Stage 1 wrist loss (v1/v2/v3) | [`bash scripts/ablation/stage1_v1v2.sh`](scripts/ablation/stage1_v1v2.sh), [`bash scripts/ablation/stage1_loss_sweep.sh`](scripts/ablation/stage1_loss_sweep.sh) | `results/stage1_*.json` |
| §5.3 Table 3(D) Stage 4 lower body | [`bash scripts/train/stage4.sh`](scripts/train/stage4.sh) | `results/stage4_ablation.json` |
| §5.2 Bootstrap CIs (multi-seed Stage 3) | [`bash scripts/ablation/stage3_multiseed.sh`](scripts/ablation/stage3_multiseed.sh) | `results/stage3_multiseed.json` |

## Repository layout

```
pianoid/
├── README.md, setup.sh, requirements.txt, environment.yml
├── inference.sh                Reproduce all paper numbers (no training)
├── train.sh                    Full from-scratch training cascade
├── src/                        All Python code
│   ├── data/, models/, evaluation/, diffusion_lib/, project_utils/
│   ├── stage{1,2,3,4}/         Per-stage training/inference internals
│   └── pipeline/               full_pipeline.py, phase0_precompute.py (Stage 1 → wrist predictions)
├── scripts/                    Per-stage and sub-experiment orchestration
│   ├── train/stage{1,2,3,4}.sh Per-stage trainers (called by train.sh)
│   ├── baselines.sh            Table 2 baselines (Li2018, MDM-adapted, EDGE-adapted)
│   ├── ablation/               Paper Table 3 ablation sweeps
│   └── preprocess/             build_splits.py
├── experiments/                Paper-specific analysis
│   ├── analysis/               §3 F1–F6
│   ├── benchmark/              §5 Table 2
│   ├── vis/                    Figures
│   └── explore/                Supporting exploration
├── vendor/                     Vendored external code (MDM, EDGE for Table 2 baseline; aria for Table 3(D) ablation)
├── MOSA_dataset/               Processed MOSA cache (not tracked)
├── results/                    Ablation result JSONs (not tracked)
└── checkpoints/                Trained weights (not tracked)
```

## Citation

```bibtex
@inproceedings{bae2026pianoid,
  title     = {Pianoid: 3{D} Pianist Body Motion Generation from {MIDI}},
  author    = {Bae, Joonhyung and Nam, Juhan},
  booktitle = {Proceedings of the 27th International Society for
               Music Information Retrieval Conference (ISMIR)},
  address   = {Abu Dhabi, UAE},
  year      = {2026},
}
```

## Acknowledgments

Built on top of [MDM](https://github.com/GuyTevet/motion-diffusion-model) (Tevet et al., ICLR 2023) and uses the [MOSA dataset](https://zenodo.org/records/11393449).

## License

Code is released under the [MIT License](LICENSE). Vendored code under
[`vendor/`](vendor/) keeps its original license (MDM, EDGE, aria; see each
`vendor/*/LICENSE`). The MOSA motion data is CC BY 4.0 and is not
redistributed here.
