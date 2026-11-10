# Stage 2 distal head error sweep — outcome

Goal: reduce the head MPJPE (production: 111.1 mm) by varying Stage 2
capacity. Variants trained from `scripts/ablation/stage2_distal_head.sh`
on the same 158/34/80 split with seed 42.

| Variant            | hidden | layers | epochs | val MPJPE | full-pipe MPJPE | head MPJPE |
|--------------------|--------|--------|--------|-----------|-----------------|------------|
| v_base             | 256    | 8      | 500    | 30.41     | —               | —          |
| **v_h512**         | 512    | 8      | 500    | **27.52** | 96.76           | **126.1**  |
| v_l12              | 256    | 12     | 500    | 28.85     | —               | —          |
| v_ep1k             | 256    | 8      | 1000   | 29.69     | —               | —          |
| (release h256)     | 256    | 8      | 500    | 28.27     | 92.07           | 111.1      |

**Outcome: no improvement.** v_h512 had the lowest val MPJPE in
*z-score-normalized* space, but in absolute coordinates after
denormalisation it was strictly worse on every backbone joint:

  Overall MPJPE   92.07 → 96.76 (+4.7)
  Head            111.1 → 126.1 (+15.0)
  Shoulders        88.0 →  97.5 (+9.5)
  Torso            79.1 →  86.8 (+7.7)
  Elbows           82.9 →  88.4 (+5.5, cascade)

The larger model converged to a more conservative diffusion sampler
(smoother trajectories), the opposite of what reducing distal cascade
error would require. Adding capacity is not the right axis here.

**Decision**: keep the h256 release checkpoint. Distal head error remains
the documented limitation in the paper (§6 Discussion).

**What would actually help (left for future work)**: a head-specific loss
weight, anchored anti-collapse losses on the head joints, or splitting the
backbone stage into a separate head-only refinement.
