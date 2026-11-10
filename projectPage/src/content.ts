// ─────────────────────────────────────────────────────────────
// Single source of truth for everything the project page states
// about the paper. Edit here, not in the components.
// ─────────────────────────────────────────────────────────────

export const PAPER = {
  title: 'Pianoid: 3D Pianist Body Motion Generation from MIDI',
  venue: 'ISMIR 2026',
  venueLong:
    'Proceedings of the 27th International Society for Music Information Retrieval Conference, Abu Dhabi, UAE',
  authors: [
    { name: 'Joonhyung Bae', affiliation: 1 },
    { name: 'Juhan Nam', affiliation: 1 },
  ],
  affiliations: ['Graduate School of Culture Technology, KAIST, Daejeon, Republic of Korea'],
  session: 'Poster P10-8',
  tagline:
    'The first system to generate 3D 26-joint full-body pianist motion from MIDI alone.',
}

// Set a URL to make the button appear. Empty string = button hidden.
export const REPO = 'https://github.com/joonhyungbae/pianoid'

export const LINKS = {
  paper: 'assets/pianoid_ismir2026.pdf',
  code: REPO,
  checkpoints: `${REPO}/releases/latest`,
  video: 'https://youtu.be/GaFwqai4PsE',
  ismir: 'https://ismir2026program.ismir.net/poster_287.html',
  data: 'https://zenodo.org/records/11393449',
  bibtexAnchor: '#bibtex',
}

export const VIDEO = {
  id: 'GaFwqai4PsE',
  title: 'Pianoid: 3D Pianist Body Motion Generation from MIDI',
}

// Shown verbatim in the "Code" section; mirrors the README Quickstart.
export const QUICKSTART = `git clone ${REPO}.git && cd pianoid
bash setup.sh && conda activate pianoid

# MOSA mocap from Zenodo -> MOSA_dataset/yp/..., then one-time preprocessing
python scripts/preprocess/build_splits.py
python src/stage1/s05_preprocess_motion.py
for w in 0.5:W05 1.0:W10 2.0:B; do
  python src/stage1/s12_precompute_training_data.py \\
    --onset_window \${w%%:*} --output_dir MOSA_dataset/training_cache_\${w#*:}
done

bash scripts/download_checkpoints.sh   # released weights, ~188 MB
bash inference.sh                      # reproduces Table 2, ~10-20 min`

export const ABSTRACT = `Pianist performance is a full-body activity, with torso sway, shoulder lift, and head tilt conveying expression alongside the notes. Yet computational models of pianist body motion have so far addressed only hands or 2D upper bodies. We present Pianoid, the first 3D 26-joint pianist body motion system from MIDI. Pianoid is a four-stage hybrid pipeline whose body partition and diffusion placement are derived from a coupling analysis of the MOSA performance corpus. MPJPE (Mean Per Joint Position Error) rewards the near-static collapse of mean-regressed baselines. To surface this failure mode we report per-axis expressiveness ratios aggregated as Δ₁, validated by a dual perceptual study (non-expert N = 28, expert N = 9). Non-expert viewers prefer Pianoid over every baseline and ablation, while still preferring the ground truth. Experts attribute the residual gap to fine timing and articulation rather than under-expression. Pianoid establishes note-level pianist body motion as a tractable MIR task via data-derived hierarchical modeling with selective diffusion.`

export const HIGHLIGHTS = [
  {
    heading: 'Body partition derived from data',
    body: 'Shoulders couple far more tightly to the torso (|r| 0.65–0.87) than to the ipsilateral wrist (0.25–0.48), on 14 of 14 pieces. This inverts the SMPL/MMM “arms” partition that prior hierarchical motion synthesizers inherit, and moving shoulders into the backbone raises elbow Z-range 10×.',
  },
  {
    heading: 'Diffusion only where the target is one-to-many',
    body: 'Torso sway is genuinely one-to-many (9.1× performer variance), so Stage 2 uses a diffusion model. Wrists, elbows, and the lower body are unimodal once conditioned, so they use deterministic regression. Diffusion at the elbow costs +211 mm MPJPE.',
  },
  {
    heading: 'MPJPE rewards mean-collapse, Δ₁ exposes it',
    body: 'The best-MPJPE baseline is a near-static skeleton (wrist-Z 0.17× GT). Per-axis expressiveness ratios aggregated as Δ₁ separate the two, and on probes where MPJPE inverts the ranking, only Δ₁ matches expert judgment.',
  },
]

// ─── Table 2 of the paper (80 test sequences) ───────────────
export type BenchmarkRow = {
  label: string
  note?: string
  ours?: boolean
  gt?: boolean
  cells: string[]
  best: number[]   // column indices rendered bold
  second: number[] // column indices rendered underlined
}

export const BENCHMARK: {
  columns: string[]
  rows: BenchmarkRow[]
  footnote: string
} = {
  columns: ['Method', 'MPJPE ↓', 'ER-rng', 'ER-spd', 'ER-jrk', 'WZR', 'AccR', 'Δ₁ ↓'],
  // best / second flags drive the bold + underline styling
  rows: [
    { label: 'Li2018', note: '★', cells: ['76.7', '0.66', '2.92', '13.1', '0.17', '10.5', '1.36'], best: [0], second: [] },
    { label: 'MDM-adapted', cells: ['88.2', '0.82', '0.46', '0.25', '0.38', '0.25', '0.79'], best: [], second: [1, 4] },
    { label: 'EDGE-adapted', cells: ['80.2', '0.71', '0.51', '0.69', '0.27', '0.54', '0.46'], best: [3, 5], second: [0, 2, 6] },
    { label: 'Pianoid (ours)', ours: true, cells: ['92.1', '1.12', '0.92', '0.34', '0.76', '0.44', '0.42'], best: [1, 2, 4, 6], second: [3, 5] },
    { label: 'Ground truth', gt: true, cells: ['—', '1.00', '1.00', '1.00', '1.00', '1.00', '0.00'], best: [], second: [] },
  ],
  footnote:
    '★ Li2018’s 13× GT jerk follows from extending its 2D/8-joint architecture to 3D/26-joint output, not a claim against the original 2D/8-joint setting. All expressiveness ratios target 1. Bold = best, underline = second.',
}

// ─── Non-expert 2AFC study (N = 28), paper §5.4 ─────────────
export const PREFERENCE = {
  caption:
    'Share of non-expert 2AFC trials preferring Pianoid, per opponent (Holm-adjusted GLMM, p < 0.001).',
  rows: [
    { label: 'vs. Li2018', value: 94.9 },
    { label: 'vs. MDM-adapted', value: 93.3 },
    { label: 'vs. BiLSTM-on-backbone', value: 91.1 },
    { label: 'vs. EDGE-adapted', value: 89.5 },
    { label: 'vs. Single-Stage MDM+zs', value: 87.2 },
  ],
  gt: {
    label: 'Ground truth vs. Pianoid',
    value: 78.3,
    note: 'Ground truth still wins head-to-head. Bradley–Terry places Pianoid (−1.28) between ground truth (0) and every baseline or ablation (−3.20 to −4.19).',
  },
}

export const EXPERT = {
  caption:
    'Nine practicing pianists (10–30 yr experience, median 17.5) diagnosed the residual gap to ground truth in a thematic interview study.',
  themes: [
    { id: 'T2', agreement: '7/9', text: 'Lost anticipation before sforzando attacks, and absent vertical wrist articulation on repeated octaves.' },
    { id: 'T3', agreement: '5/9', text: 'Mistimed trunk drift on lyrical probes. A phase deficit rather than an amplitude one, since Pianoid’s trunk carries 2.22× ground-truth energy.' },
    { id: 'T4', agreement: '4/9', text: 'Attentional-anchor split. Non-experts watch trunk and head, experts watch wrist, gaze, and elbow.' },
    { id: 'T1', agreement: '3/9', text: 'Hidden signatures, namely absent gaze and elbow rigidity.' },
  ],
  takeaway:
    'The gap experts perceive is fine timing and articulation, not under-expression.',
}

export const BIBTEX = `@inproceedings{bae2026pianoid,
  title     = {Pianoid: 3{D} Pianist Body Motion Generation from {MIDI}},
  author    = {Bae, Joonhyung and Nam, Juhan},
  booktitle = {Proceedings of the 27th International Society for
               Music Information Retrieval Conference (ISMIR)},
  address   = {Abu Dhabi, UAE},
  year      = {2026},
}`

// ─── Comparison viewer ──────────────────────────────────────
export type Variant = {
  key: string
  label: string
  file: string // .bin filename inside public/data/{idx:05d}/
  hint: string
  citation?: string
}

// Order follows lay 2AFC preference vs. Pianoid (paper §5.4):
// pianoid → monolithic (87.2%) → edge (89.5%) → bilstm (91.1%) → mdm (93.3%) → li2018 (94.9%)
export const VARIANTS: Variant[] = [
  { key: 'pianoid', label: 'Pianoid (ours)', file: 'gen.bin', hint: '' },
  { key: 'monolithic', label: 'Single-Stage MDM+zs', file: 'gen_monolithic.bin',
    hint: 'hierarchy ablation — same tricks, monolithic (2AFC: 87.2% prefer Pianoid)' },
  { key: 'edge', label: 'EDGE-adapted', file: 'gen_edge.bin',
    hint: 'Tseng et al. 2023, MIDI-conditioned (2AFC: 89.5%)', citation: 'tseng2023edge' },
  { key: 'bilstm', label: 'BiLSTM-on-backbone', file: 'gen_bilstm.bin',
    hint: 'diffusion ablation — deterministic backbone (2AFC: 91.1%)' },
  { key: 'mdm', label: 'MDM-adapted', file: 'gen_mdm.bin',
    hint: 'Tevet et al. 2023, MIDI-conditioned (2AFC: 93.3%)', citation: 'tevet2023human' },
  { key: 'li2018', label: 'Li2018', file: 'gen_li2018.bin',
    hint: 'CNN-LSTM, ISMIR 2018 (2AFC: 94.9%)', citation: 'li2018skeleton' },
]

// Curated showcase indices: the 12 clips for which all baseline .bin files were
// generated and deployed. Ordered by Pianoid MPJPE ascending (best → worst).
//   60 Mozart mo3 68.4 · 35 Chopin ch1 83.2 · 53 Mozart mo1 83.3 ·
//   25 Brahms br1 83.9 · 64 Mozart mo4 84.5 · 74 Tchaikovsky tc1 85.8 ·
//   38 Chopin ch2 87.5 · 45 Debussy de1 88.8 · 4 Bach ba1 90.9 ·
//   31 Brahms br2 90.9 · 21 Beethoven be1 92.1 · 8 Bach ba2 97.1
export const CURATED_IDX = [60, 35, 53, 25, 64, 74, 38, 45, 4, 31, 21, 8]
