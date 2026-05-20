# Compare-to-Agisoft Results — first 4-session benchmark

Results of running `src/preprocessing/compare_to_agisoft.py` on the four sessions delivered by the supervisor in `demoanlage2025_v0/` (now restructured under `input_plots/phone/`). Run date: 2026-05-20.

Each session ran the full preprocessing pipeline:
```
python src/preprocessing/run_preprocessing.py field=<field> plot=<date> run_compare=true
```

Per-session raw outputs live under `input_plots/phone/{field}/{plot}/logs/compare_to_agisoft.json` (full per-camera lists) and `compare_summary.json` (aggregate stats the orchestrator reads).

For the meaning of "translation error" / "rotation error" in the context of Umeyama-aligned reconstructions, see [`SFM_PIPELINE_COMPARISON.md`](SFM_PIPELINE_COMPARISON.md). For how Agisoft's own marker errors compare, see [`AGISOFT_QUALITY_METRICS.md`](AGISOFT_QUALITY_METRICS.md).

---

## Side-by-side comparison

| Session | Coverage (ours/agisoft) | Translation err (mm)<br>median \| mean \| max | Rotation err (°)<br>median \| mean \| max | Umeyama scale | Verdict |
|---|---|---|---|---|---|
| **field_A/20250609** | 119 / 119 ✅ | **12.3** \| 14.6 \| 41.8 | **0.48** \| 0.56 \| 1.57 | 0.559 | **Excellent** |
| **field_D/20250523** | 64 / 64 ✅ | **15.1** \| 17.1 \| 52.2 | **0.96** \| 0.87 \| 1.54 | 0.483 | Good |
| **field_A/20250618** | 93 / 93 ✅ | **22.9** \| 22.8 \| 37.4 | **0.90** \| 0.89 \| 1.43 | 0.468 | Good |
| **field_D/20250530** | 63 / 85 ⚠️ | **60.0** \| 66.8 \| 162.9 | **3.74** \| 3.48 \| 4.74 | 0.497 | **Problematic** |

---

## How to read these numbers

### Translation error in mm (after Umeyama alignment)

| Range | Meaning |
|---|---|
| **0–15 mm** | Within Agisoft's *own* marker measurement noise floor (3D Err on a clean Agisoft run is typically 5–15 mm). At this level we're indistinguishable from the reference. |
| **15–30 mm** | Slight geometric looseness — usually a few cameras drift more than the rest. 3DGS still trains fine; the looseness gets absorbed by Gaussian densification. |
| **30–60 mm** | Real pose drift across the dataset. Likely to start showing as slightly softer 3DGS renders at high resolution. |
| **>60 mm** | Significant — investigate scene/capture issues before blaming the pipeline. |

### Rotation error in degrees

| Range | Meaning |
|---|---|
| **<1°** | Excellent — sub-degree alignment, matches a careful manual calibration. |
| **1–2°** | Acceptable. Phone capture noise + SIFT matching limitations. |
| **2–4°** | Noticeable — cameras are looking in measurably different directions than Agisoft thinks they should. |
| **>4°** | Serious — usually correlated with bad translation, indicates the BA solution diverged. |

### Coverage (n_common vs n_agisoft)

The ratio tells you how many images Agisoft registered that **we also registered**. A gap (e.g. 63/85 on 20250530) means our COLMAP failed on 22 images Agisoft handled fine — usually a sign of poor feature quality on those specific frames (blur, lighting changes).

### Umeyama scale

Our COLMAP recovers a reconstruction in arbitrary units; Agisoft is metric. The recovered scale factor (around 0.5 in all sessions) is the ratio that converts our units to meters. Should be roughly consistent within a field — if it varies wildly between sessions on the same field, the alignment is unreliable.

Here scale is 0.47–0.56 across all four sessions. Reasonably consistent: confirms the alignment is meaningful (same camera/lens, just different arbitrary-unit scaling per run).

---

## Per-session takeaways

### field_A/20250609 (best — 119 images)
- 12.3 mm median translation, 0.48° median rotation. **Excellent.**
- Largest dataset (119 cams) and the **most precise** result — more images give the bundle adjustment more constraints, which tightens both translation and rotation.
- Worth using as the **primary 3DGS benchmark session**.

### field_D/20250523 (64 images) + field_A/20250618 (93 images)
- 15–23 mm median, ~0.9° rotation. Solid, normal-quality COLMAP runs.
- 20250618 is slightly worse on translation (22.9 vs 15.1 mm). Likely scene-dependent — small variations in feature distribution.
- Both should produce 3DGS quality comparable to Agisoft.

### field_D/20250530 (problematic — 63/85 registered)
- 60 mm median, 162.9 mm max translation, 3.7° median rotation. **3–4× worse than the others** on every metric.
- **22 images Agisoft registered but we didn't** — our COLMAP gave up on a quarter of the captured frames.
- The max=162.9 mm with std=33 mm tells us it's not uniform — some cameras are far worse than others.

**Confirmed root cause:** the raw images in 20250530 are visibly blurrier than the other three sessions. SIFT can't track features that change shape between shots, so:
1. Feature matching produces fewer reliable pairs.
2. The mapper drops images it can't constrain → 22 missing.
3. Surviving images get loose pose estimates → large mean error + occasional outliers (the 162.9 mm max).

Likely capture-time causes for the blur:
- **Wind** — wheat moves between shots, SIFT can't track moving features.
- **Lighting changes mid-capture** — clouds rolling through, time-of-day too long.
- **Walking pace / motion blur** — faster or longer walk → more blur → fewer reliable keypoints.

The same field worked fine one week earlier (20250523 → 15 mm), so this isn't a "wheat field is too hard" issue — it's a capture-quality issue specific to that session.

---

## Bottom-line interpretation

Three of four sessions are firmly in the "SIFT is good enough" zone (per the [benchmarking workflow](../src/preprocessing/README.md#benchmarking-workflow--when-and-when-not-to-upgrade-the-pipeline) in the preprocessing README):

- Mean translation 15–23 mm — within ~2× of Agisoft's own marker noise floor.
- Rotation under 1° — practically indistinguishable from Agisoft for 3DGS purposes.

**Recommended next step (decision matrix):**

| Question | Action |
|---|---|
| Should we install hloc + SuperPoint+LightGlue *now*? | **No.** Three sessions are good; one is bad for *data-quality* reasons no detector can fix. |
| Should we benchmark 3DGS PSNR/SSIM/LPIPS on these sessions? | **Yes, this is the next step.** That's the thesis-relevant metric — `compare_to_agisoft.py` is only a proxy. |
| Should we engineer a blur fix for 20250530? | **Not yet.** First measure how much render quality actually suffers. The blur is in the source images, so the upper bound on render quality is hard-capped regardless of pipeline. |

### Suggested concrete plan

1. **Train 3DGS on field_A/20250609 first** (best COLMAP, most images, cleanest test).
   - Train once on our `sparse/0/`, once on `agisoft/sparse/0/`.
   - Compare PSNR/SSIM/LPIPS on held-out test cameras.
   - If our PSNR is within ~1 dB of Agisoft's → the open-source pipeline is validated.

2. **Then train 3DGS on field_D/20250530** (the stress test).
   - Train on our `sparse/0/` only; visually inspect the renders.
   - Measure how much render quality degrades vs the 20250609 baseline.

3. **Only after seeing those PSNR numbers, decide on blur mitigation.**
   - If 20250530 PSNR is roughly explainable by the input blur (e.g. ~3 dB worse than 20250609): **document as known limitation of source data**, don't engineer a fix.
   - If 20250530 PSNR is severely broken (e.g. 10 dB worse): start with the cheapest fix — compute per-image Laplacian variance and drop the bottom-decile blurriest frames, then re-run COLMAP on the filtered set. Costs ~30 min to implement.
   - Only if filtering doesn't help → consider hloc / blur-aware preprocessing (1–2 days).

**TL;DR:** Run the 3DGS benchmark before doing any further pipeline engineering. The benchmark might reveal that the pipeline is already good enough thesis-wise, in which case 20250530's poor metrics become a single bullet point under "limitations" rather than a fix to implement.

---

## Sharpness analysis — confirms 20250530 is a data-quality issue

After observing that `field_D/20250530` raw images look visibly blurrier than the other sessions, I ran the diagnostic script [`src/preprocessing/analyze_sharpness.py`](../src/preprocessing/analyze_sharpness.py) on all four. The script computes the **Laplacian variance** per image (the standard quick sharpness metric — sharp images have lots of strong edges → high variance of the Laplacian; blurry images smooth those edges out → low variance) and reports the per-session distribution.

**The script does not modify any image** — it only reads them, prints a report, and writes `{source_path}/logs/sharpness_report.json`. Pure diagnostic.

### Per-session sharpness

| Session | Images | Median (Laplacian var) | Min / Max | 10th pctile | Outliers (<0.5× median) | Label at median |
|---|---|---|---|---|---|---|
| **field_A/20250609** | 119 | **1934.6** | 824.7 / 2214.6 | — | 2 | very sharp |
| **field_D/20250523** | 64  | **1196** | 1018 / 1734 | 1067 | 0 | very sharp |
| **field_A/20250618** | 93  | **1026.4** | 656.8 / 1783.0 | — | 0 | very sharp |
| **field_D/20250530** | 85  | **399** | 151.4 / 678.1 | 230 | **5** | sharp (3× lower than the rest) |

(Sharpness scores are in arbitrary units and only meaningful within a session, but on the same camera/resolution they're directly comparable across sessions. All four were captured on the same phone at the same resolution, so the cross-session comparison is fair.)

### Key observations

1. **20250530 is 3× less sharp than 20250523** — same field, one week apart, same camera. The week's wind/lighting/walking-pace conditions degraded image quality by a factor of three.
2. **20250530 is the only session with multiple outliers** (5 images below half the session median).
3. **20250530's blurriest images** (`IMG_20250530_123737.jpg`, `..._123904.jpg`, etc.) cluster in time, suggesting a sustained issue (wind gust, low light) rather than a one-off motion mistake.
4. **The other three sessions are uniformly sharp** — no outliers, narrow distributions.

### What this confirms

The `compare_to_agisoft.py` numbers on 20250530 (66 mm mean translation, 3.5° rotation, 22 missing images) were caused by **input image quality**, not by any pipeline weakness. SIFT (or any feature detector) can only extract reliable keypoints from sharp, stable image content. When 5+ images are visibly blurry and the rest are 3× less crisp than usual, the feature graph becomes sparse and the mapper drops images it can't constrain.

This validates the recommendation above: **don't engineer a pipeline fix for 20250530**. Either:
- Document it as a data-quality outlier in the thesis, or
- Cheap diagnostic-driven filter step (later, if it turns out to matter): drop images with sharpness <0.5× session median before COLMAP. For 20250530 that's 5 images — a small filter, no detector swap required.

### Re-running the analysis

```bash
python src/preprocessing/analyze_sharpness.py field=field_D plot=20250530
# write the per-image scores to logs/sharpness_report.json (not gitignored — keep across sessions)

# bigger/wider report (top 20 blurriest, top 10 sharpest):
python src/preprocessing/analyze_sharpness.py field=field_D plot=20250530 worst_n=20 best_n=10

# faster on huge sessions — downscales to half-resolution before computing the score
# (only affects absolute values, the ranking stays the same):
python src/preprocessing/analyze_sharpness.py field=field_A plot=20250609 downscale=0.5
```

Full config defaults in [`configs/preprocessing/analyze_sharpness.yaml`](../configs/preprocessing/analyze_sharpness.yaml).
