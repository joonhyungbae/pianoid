# Explore 02: Hand Input Conditioning Sweep — Plan

## 1. Goals

- **Fix the architecture**: the **best Mamba** validated in Explore 01 (hand_mamba, config_10aef044: hidden_dim=512, num_layers=4, d_state=16).
- **Vary only the input**: with the same model and the same data, explore **which input gives the best performance**.
- **Conditions compared**: MIDI only / ARIA only / MIDI+ARIA.

---

## 2. Where ARIA / MIDI live in the repo

### 2.1 Data (training_cache)

- **Path**: `MOSA_dataset/training_cache` (or the npz produced by `src/stage1/s12_precompute_training_data.py`).
- **npz keys**:
  - `joints`: (T, 26, 3)
  - `active_keys`, `key_velocity`, `key_onset`: derived from MIDI
  - **`aria_embed`**: (512,) global, or (num_windows, 512) per-window. **Already saved.**

### 2.2 Where ARIA is consumed (for reference)

- **Loading**: in `src/stage1/s12_precompute_training_data.py`, per sample the `aria_embedding_temporal.npy` or `aria_embedding.npy` is loaded and saved into the npz as `aria_embed`.
- **Hand-side usage**:
  - `HandGeneratorNet`: `aria_embed` (B, 512) -> AriaEncoder -> (B, T, hidden), concatenated with MIDI then fused.
  - `MusicAlignedHandMDM`: when `use_aria=True`, ARIA (B, 512) is projected and added to the conditioning.
  - `HandMDM` (hierarchical_mdm): ARIA (B, 512) used as global conditioning.

### 2.3 The current Mamba (Explore 01)

- **HandMambaRegression**: input is **MIDI only** (piano_roll + velocity + onset -> 264-dim). ARIA not used.
- **Data**: the previous `stage1_1_hand_architecture/dataset.py` (now removed) did not load `aria_embed`.

---

## 3. Input-condition definitions

| Condition ID | Description | Input dim (per frame) | Note |
|--------|------|------------------------|------|
| **midi_only** | MIDI only (same as Explore 01) | 264 | piano_roll(88) + velocity(88) + onset(88) |
| **aria_only** | ARIA only (one vector per segment, broadcast over T) | 512 | One (512,) per sequence; for temporal ARIA, pick by window index |
| **midi_aria** | MIDI + ARIA concat | 264 + 512 = 776 | per-frame: [midi_264 ; aria_512] (ARIA broadcast as the same vector) |

- **Fairness**: same Mamba backbone, same data splits and windows, same loss and metrics. **Only the input vector changes.**

---

## 4. Implementation summary

### 4.1 Data

- **File**: `experiments/explore/stage1_2_hand_input_conditioning/dataset.py` (or an extension of stage1_1's dataset).
- **Extra load**: `aria_embed` from npz.
  - shape (512,) -> reused identically as (B, 512) regardless of segment/window.
  - shape (num_windows, 512) -> pick the segment vector via window index `start // stride_aria` -> (B, 512).
- **Returns**: in addition to existing `hand` and `midi_features`, return `aria_embed`: (B, 512) (stacked at batch time).

### 4.2 Model

- **Option A (recommended)**: a **single Mamba with the input dimension swapped**.
  - `input_dim` ∈ {264, 512, 776}. Everything else identical (hidden_dim=512, num_layers=4, d_state=16).
  - `midi_only`: input_proj(264 -> hidden); `aria_only`: input_proj(512 -> hidden); `midi_aria`: input_proj(776 -> hidden).
  - Each run uses **only one** input condition. Specify via config: `input_conditioning: "midi_only" | "aria_only" | "midi_aria"`.
- **Option B**: a single model that always takes a 776-dim input. For "MIDI only" zero out the ARIA part, for "ARIA only" zero out the MIDI part.
  -> Same parameter count and architecture across all three conditions.

- **Conclusion**: Option B is simpler — a **single checkpoint structure** runs the three experiments via input masking.
  Option A keeps the input projection dimension different per condition while leaving the Mamba/output identical.

### 4.3 Training and evaluation

- **Fixed**: optimizer, scheduler, epoch count, early stopping, train/val/test split, metrics (MPJPE, Onset Align, Jitter, Vel Corr).
- **Sweep**: 3 runs — `midi_only`, `aria_only`, `midi_aria`.
- **Output**: per-run best val loss, test-set metrics, and (optionally) an aggregate Borda score.

---

## 5. Suggested directory / scripts

```
experiments/explore/
  stage1_2_hand_input_conditioning/
    README.md              # goals, condition table, how to run
    dataset.py             # Hand data with ARIA (returns aria_embed)
    train_sweep.py         # sweeps the three input conditions on the same Mamba
    evaluate.py            # test-set evaluation and table output
    docs/
      results.md           # (post-experiment) per-condition metrics and conclusions
```

- **Dependency**: the Mamba in `src.models.hand_baselines` (or a stage1_2 wrapper).

---

## 6. Execution order (summary)

1. **Data**: confirm the training_cache npz has `aria_embed`. If missing, regenerate via s12 etc.
2. **Dataset**: load ARIA and return per-window (512,) or (B, 512).
3. **Model**: fix Mamba input to 776-dim and feed [midi; 0], [0; aria], or [midi; aria] per run (or change input_dim per condition: 264/512/776).
4. **Sweep**: train `midi_only`, `aria_only`, and `midi_aria` separately, saving the best checkpoint each.
5. **Evaluation**: compute metrics on the same test set and tabulate "which input is best".

---

## 7. Success criteria

- Training and evaluating the three conditions on **the same Mamba architecture and the same data** must give a numerical answer for "MIDI only vs ARIA only vs MIDI+ARIA" via **test MPJPE (and Onset / Jitter / Vel Corr)**.
- (Optional) If MIDI+ARIA is significantly better, this is the basis for fixing the Hand-stage input to MIDI+ARIA in the downstream pipeline.

This plan defines the sweep "with the current best Mamba architecture, which input is the best?" in a single shot.
