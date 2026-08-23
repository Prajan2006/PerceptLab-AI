# E1 vs E3 — Project Engineering Baseline Comparison

**Status:** read-only analysis. No source, configuration, dataset, or experiment artifacts were modified.
**Runs compared:**
- E1 = `data/experiments/engineering_baseline_v1` (Protocol B, 1 epoch/fold) — frozen reference
- E3 = `data/experiments/engineering_baseline_v1_E3` (identical recipe, only variable changed: epochs=3)

**Verification basis:** all values below re-read directly from each fold's `metrics.json`, `history.json`, and both `run_summary.json` files.
- Both runs: `{"completed": 15, "skipped": 0, "failed": 0}` — all 15 LOPO folds present, none skipped/failed.
- Every E1 history has exactly 1 epoch entry; every E3 history has exactly 3 entries (`epoch = [1, 2, 3]`) — confirmed per fold.
- All 15 E3 checkpoints load on CUDA (epoch=3 restored, inference `(1, 3)`, finite).
- `engineering_baseline_v1` remains untouched: 46 files / 15 directories; newest artifact mtime 2026-08-23 08:01, pre-dating the E3 run start (11:22).

## Per-fold comparison

| Fold | Train n | Test n | E1 deg | E3 deg | abs Δ | % Δ | E1 val loss | E3 final val loss |
|---|---|---|---|---|---|---|---|---|
| p00 | 34,740 | 2,927 | 7.087 | 6.618 | −0.469 | −6.6% | 0.006796 | 0.006380 |
| p01 | 34,763 | 2,904 | 11.005 | 7.936 | −3.069 | −27.9% | 0.017924 | 0.007695 |
| p02 | 34,751 | 2,916 | 10.356 | 9.763 | −0.593 | −5.7% | 0.013024 | 0.011458 |
| p03 | 34,738 | 2,929 | 13.756 | 17.221 | **+3.465** | **+25.2%** | 0.023436 | 0.033533 |
| p04 | 34,807 | 2,860 | 7.916 | 5.192 | −2.724 | −34.4% | 0.008663 | 0.003638 |
| p05 | 34,797 | 2,870 | 12.830 | 11.672 | −1.158 | −9.0% | 0.018538 | 0.016074 |
| p06 | 34,790 | 2,877 | 11.751 | 9.533 | −2.218 | −18.9% | 0.017167 | 0.012214 |
| p07 | 34,824 | 2,843 | 11.482 | 10.506 | −0.976 | −8.5% | 0.016719 | 0.014048 |
| p08 | 34,900 | 2,767 | 13.314 | 11.593 | −1.721 | −12.9% | 0.022829 | 0.017417 |
| p09 | 34,948 | 2,719 | 9.346 | 8.315 | −1.031 | −11.0% | 0.011428 | 0.008545 |
| p10 | 35,473 | 2,194 | 11.081 | 11.635 | +0.554 | +5.0% | 0.015356 | 0.016571 |
| p11 | 35,405 | 2,262 | 8.746 | 8.481 | −0.265 | −3.0% | 0.010242 | 0.009205 |
| p12 | 36,066 | 1,601 | 10.411 | 8.815 | −1.596 | −15.3% | 0.015559 | 0.010444 |
| p13 | 36,169 | 1,498 | 15.628 | 10.528 | −5.100 | −32.6% | 0.028300 | 0.013905 |
| p14 | 36,167 | 1,500 | 13.357 | 8.653 | −4.704 | −35.2% | 0.020479 | 0.009706 |

Final losses above are the last recorded epoch of each run (E1 epoch 1; E3 epoch 3). Full 3-epoch trajectories are in each `history.json`.

## Aggregates

| Statistic | E1 (1 epoch) | E3 (3 epochs) |
|---|---|---|
| Mean angular error | **11.2045°** | **9.7642°** |
| Population std dev | ±2.2633° | ±2.6747° |
| Sample std dev | ±2.3428° | ±2.7685° |
| Minimum fold | 7.087° (p00) | 5.192° (p04) |
| Maximum fold | 15.628° (p13) | 17.221° (p03) |

- Folds improved: **13 / 15** · folds worsened: **2 / 15** (p03, p10)
- Largest improvements: p13 −5.100° (−32.6%), p14 −4.704° (−35.2%), p01 −3.069° (−27.9%)
- Largest degradations: p03 +3.465° (+25.2%), p10 +0.554° (+5.0%)
- Paired Wilcoxon signed-rank test across the 15 folds: **W = 16.0, p = 0.010254** (significant improvement at α = 0.05)
- Validation-loss change agrees in sign with angular-error change in **15 / 15 folds**

## Overfitting / instability evidence
- **p03:** validation loss rose across E3 epochs (0.0250 → 0.0409 → 0.0335) while train loss fell (0.00543 → 0.00366), and angular error degraded from 13.756° to 17.221° — a clear overfitting/instability signature on this held-out subject under longer training.
- **p10:** milder version — val loss rose by epoch 3 (0.0121 → 0.0166), angular error slightly worse (+0.554°).
- The other 13 folds show flat-to-falling val loss over the three epochs; no instability there.

## Interpretation

**Does 1 → 3 epochs provide enough evidence to adopt 3 epochs as the next reference configuration?**

Yes — as the project's new *working* reference configuration, within these explicit limits:
1. Directional consistency: 13/15 folds improved; mean improved −1.4403° (−12.9% relative); paired Wilcoxon p ≈ 0.010 against the frozen baseline; the locked metric and validation loss agree on every fold.
2. Not universal: p03 degrades materially (+25%, with a rising val-loss trajectory), p10 mildly (+5%). Fold spread widened (pop-std 2.26° → 2.67°). Three epochs is therefore *not* claimed to be universally optimal, and intermediate durations (e.g., 2 epochs) were not tested.
3. Scope limits: single seed (0), single split family, n = 15 correlated folds; the Wilcoxon p-value is supportive evidence, not proof of generalization beyond this dataset/recipe.

**Recommendation:** adopt `epochs=3` as the reference training duration for subsequent experiments (recorded as an explicit project decision), keep `engineering_baseline_v1` frozen as the historical 1-epoch baseline, and treat subject-level instability (p03-style val-loss divergence) as a standing observation motivating future investigation (e.g., validation-based epoch selection policy — which would be a protocol decision requiring approval).

---
*Analysis script: `%TEMP%\opencode\e1e3_analysis.py`; machine-readable dump: `%TEMP%\opencode\e1e3_dump.json`. No experiment directories were modified.*
