# E3 Eye-Region vs E3 Face-Only — Engineering Baseline Comparison

**Status:** read-only analysis. No source code was modified; no training was run; no experiment artifacts were committed.
**Runs compared:**
- Reference (frozen) = `data/experiments/engineering_baseline_v1_E3` — ResNet50 face-only input, frozen at mean **9.7642°**
- Candidate = `data/experiments/engineering_baseline_v1_E3_eye_region` — same Protocol B recipe (seed 0, epochs=3, batch 8), only variable changed: model/input = `resnet50_face_eyes`

**Verification basis:** all values re-read directly from each fold's `metrics.json` and both `run_summary.json` files. Both runs: `{"completed": 15, "skipped": 0, "failed": 0}` — all 15 LOPO folds present.

## Per-fold comparison

| Fold | Test n | Eye-region deg | E3 face deg | abs Δ (eye−face) | % Δ |
|---|---|---|---|---|---|
| p00 | 2,927 | 6.2170 | 6.6179 | −0.4010 | −6.06% |
| p01 | 2,904 | 7.8317 | 7.9358 | −0.1041 | −1.31% |
| p02 | 2,916 | 10.1768 | 9.7631 | +0.4137 | +4.24% |
| p03 | 2,929 | 14.6048 | 17.2214 | **−2.6166** | −15.19% |
| p04 | 2,860 | 6.6526 | 5.1924 | +1.4602 | +28.12% |
| p05 | 2,870 | 12.5480 | 11.6722 | +0.8758 | +7.50% |
| p06 | 2,877 | 8.1797 | 9.5330 | −1.3533 | −14.20% |
| p07 | 2,843 | 9.7560 | 10.5060 | −0.7501 | −7.14% |
| p08 | 2,767 | 11.9498 | 11.5931 | +0.3567 | +3.08% |
| p09 | 2,719 | 9.3624 | 8.3151 | +1.0473 | +12.60% |
| p10 | 2,194 | 10.0494 | 11.6352 | −1.5858 | −13.63% |
| p11 | 2,262 | 8.3426 | 8.4814 | −0.1388 | −1.64% |
| p12 | 1,601 | 8.3741 | 8.8154 | −0.4413 | −5.01% |
| p13 | 1,498 | 14.1675 | 10.5278 | +3.6397 | +34.57% |
| p14 | 1,500 | 12.6658 | 8.6533 | **+4.0125** | **+46.37%** |

## Aggregates

| Statistic | Eye-region | E3 face-only (frozen) |
|---|---|---|
| Folds | 15 | 15 |
| Mean angular error | **10.0585°** | **9.7642°** |
| Population std dev | ±2.5211° | ±2.6747° |
| Sample std dev (ddof=1) | ±2.6096° | ±2.7685° |
| Minimum fold | 6.2170° (p00) | 5.1924° (p04) |
| Maximum fold | 14.6048° (p03) | 17.2214° (p03) |

## Comparison vs E3

- Folds improved: **8 / 15** · folds worsened: **7 / 15** · unchanged: **0**
- Mean paired difference (eye − face): **+0.2943°**
- Mean absolute difference across folds: **1.2798°**
- Relative change of means: **+3.01% (worse)**
- Largest improvement: **p03 −2.6166° (−15.19%)** — the fold that was worst for face-only
- Largest degradation: **p14 +4.0125° (+46.37%)**, followed by p13 +3.6397° (+34.57%)

## Statistical note

No statistical hypothesis test was performed in this analysis. Accordingly, **no claim of statistical significance is made** about any difference between the two configurations. The near-even 8/7 fold split and per-fold swings of ±2.6–4.0° indicate the aggregate difference (+0.29°) is small relative to fold-to-fold variability (sample SD of paired differences ≈ 1.79°).

## Interpretation

The eye-region crop does **not** improve over the frozen E3 face-only reference on this protocol: the mean is 0.2943° higher (+3.01%), with highly discordant per-fold behavior rather than a consistent shift. Notably, the two largest degradations (p13, p14) are the folds with the smallest held-out test sets (n ≈ 1,500). The single biggest gain (p03, −15.19%) occurs exactly where the face-only baseline was weakest (17.22°), suggesting the eye-region input may help hardest subjects while hurting others — but with n = 15 correlated folds and no test performed, this remains an observation, not an established effect.

**Recommendation:** do not adopt the eye-region variant as reference on current evidence; retain E3 face-only as frozen baseline. If pursued further, run additional seeds before drawing conclusions.
