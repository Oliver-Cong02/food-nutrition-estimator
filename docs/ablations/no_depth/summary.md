# Ablation: No-Depth (RGB-only)

**Date:** 2026-04-26
**Run id:** `ablation_no_depth_seed42`
**Compared against:** `main_seed42` (RGB-D)
**Held-out test set:** rgb_test_ids ∩ available, n=507
**Wall-clock:** ~10 minutes (single A6000)

## Results table

| Metric | Main (RGB+D) | Ablation (RGB-only) | Δ (main − ablation) | 95% Bootstrap CI | Significant? |
|---|---|---|---|---|---|
| **kcal MAE** | 61.86 | **57.58** | **−4.19** | [−6.90, −1.64] | ★ **depth HURTS** |
| **mass MAE** | **37.89** | 46.56 | **+8.64** | [+5.94, +11.33] | ★ **depth HELPS** |
| fat MAE | 4.69 | 4.53 | −0.15 | [−0.30, +0.01] | not sig |
| carb MAE | 6.15 | 6.08 | −0.07 | [−0.26, +0.15] | not sig |
| protein MAE | 5.99 | 5.86 | −0.14 | [−0.44, +0.15] | not sig |
| ingr F1 (micro) | 0.331 | 0.318 | +0.013 | — | depth marginally helps |
| top-5 ingr IoU | 0.230 | 0.233 | −0.003 | — | tie |

(Bootstrap: paired, n=1000 resamples, seed=0. "Significant" = 95% CI excludes 0.)

## Interpretation

The two significant effects point in opposite directions, exactly as a physics-motivated reading of the modalities would predict:

- **Mass benefits from depth (+8.6 g, 23% lower MAE).** Mass is a volumetric quantity, and depth gives the model direct geometric access to dish-volume cues that RGB alone has to infer indirectly via shading and perspective. This is what RGB-D is supposed to deliver and the result confirms the spec's §3.2 design choice for the dual-stream architecture, **for the mass head specifically**.

- **kcal is hurt by depth (−4.2 kcal, 7% higher MAE).** kcal correlates more strongly with ingredient identity and macronutrient composition than with absolute portion volume — once you know the dish is "salad" vs "fried rice," per-gram density dominates. RGB carries that compositional signal cleanly. The depth stream introduces capacity that the model has to spend; in the 50-epoch budget, that capacity competes with RGB's ingredient signal. With more training (or a better-calibrated derived-kcal pathway), depth could plausibly *help* kcal too — but in this slice of the design space it doesn't.

- **Macronutrients (fat / carb / protein) are unaffected** within bootstrap noise. They're pulled along by ingredient identity, and depth doesn't change ingredient identity meaningfully (F1 changes by only 0.013).

## Implications for the spec / future work

- The 50/50 average of direct + derived kcal in §3.2 is currently dropped at eval time (see `evaluate.py:kcal_avg = preds_kcal_direct`). The derived path would add value once the ingredient classifier reaches F1 ≥ 0.6 and the mass head suppresses absent-slot outputs.
- A natural follow-up is a **head-specific depth gate**: route depth into mass head, suppress it from the kcal head. Quick win available within the same compute budget.
- The mass-vs-kcal trade-off might shrink with longer training or a smaller depth encoder (current ConvNeXt-Tiny depth branch may be over-parameterized for the 2479-dish train set).

## Files

- Predictions: `docs/runs/ablation_no_depth_seed42/eval/predictions.csv`
- Eval metrics: `docs/runs/ablation_no_depth_seed42/eval/eval_results.json`
- Significance: `docs/ablations/no_depth/significance.json`
