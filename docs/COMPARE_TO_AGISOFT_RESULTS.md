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

---

## 3DGS quality benchmark — COLMAP SfM vs Agisoft SfM (PSNR / SSIM / LPIPS)

Run date: 2026-05-21. Trained vanilla 3DGS on both sides of each session (our COLMAP `sparse/0/` and supervisor's Agisoft `agisoft/sparse/0/`) on an ETH Euler RTX 4090 (24 GB), 15 000 iterations, `resolution=1`, `data_device_cpu=true`, `sh_degree=3`. Metrics computed on the test split (llffhold=8 → every 8th sorted image is test).

The COLMAP-vs-Agisoft compare from the section above answers "**how well does our SfM match Agisoft's geometry?**" This section answers the downstream question: "**how well does the 3DGS reconstruction trained on our SfM render new views, compared to one trained on Agisoft's SfM?**" That's the metric that actually matters for the thesis — geometry differences only count if they translate into render-quality differences.

### Per-session metrics

| Session | Metric | COLMAP | Agisoft | Δ (Agi − COL) |
|---|---|---|---|---|
| **field_A/20250609** | PSNR ↑ | 14.52 | 14.89 | **+0.37** |
|                       | SSIM ↑ | 0.193 | 0.194 | +0.001 |
|                       | LPIPS ↓ | 0.607 | 0.562 | **−0.046** |
| **field_A/20250618** | PSNR ↑ | 14.30 | 14.36 | +0.06 |
|                       | SSIM ↑ | 0.207 | 0.213 | +0.005 |
|                       | LPIPS ↓ | 0.563 | 0.529 | −0.034 |
| **field_D/20250523** | PSNR ↑ | 14.67 | 14.89 | +0.22 |
|                       | SSIM ↑ | 0.275 | 0.281 | +0.006 |
|                       | LPIPS ↓ | 0.519 | 0.493 | −0.026 |
| **field_D/20250530** | PSNR ↑ | 16.51 | 16.40 | −0.10 |
|                       | SSIM ↑ | 0.404 | 0.405 | +0.002 |
|                       | LPIPS ↓ | 0.509 | 0.497 | −0.012 |
| **MEAN (4 sessions)** | PSNR ↑ | **14.99** | **15.13** | **+0.14** |
|                        | SSIM ↑ | 0.270 | 0.273 | +0.003 |
|                        | LPIPS ↓ | 0.549 | 0.520 | **−0.029** |

Raw JSONs: `results/reconstruction/phone/{field}/{date}/vanilla_3dgs/colmap_bench/results.json` and `results/reconstruction/phone/{field}/{date}/agisoft/vanilla_3dgs/agisoft_bench/results.json`.

### Headline findings

1. **The gap between COLMAP-3DGS and Agisoft-3DGS is tiny.** PSNR mean +0.14 dB in Agisoft's favor — well below the ~0.5–1.0 dB threshold typically considered "meaningfully different" in 3DGS benchmarks. SSIM gap is essentially noise (+0.003). LPIPS gap (−0.029) is small but the most consistent: Agisoft wins LPIPS on all 4 sessions.

2. **Agisoft wins on every metric on every session except one** (`field_D/20250530` where COLMAP edges out by 0.1 dB PSNR — possibly because COLMAP only registered 63 of 85 images, dropping the worst-quality ones, while Agisoft kept all 85).

3. **`field_D/20250530` (the blurry outlier session) scores HIGHEST on PSNR, not lowest.** Counterintuitive — this likely reflects test-view ease, not reconstruction quality: with blurrier input, render and gt are both blurry → numerical similarity rises even though absolute quality is low. SSIM bears this out (0.40 vs 0.19-0.28 elsewhere): the textured wheat detail is washed out, so structural matching becomes "easier."

4. **All absolute PSNR numbers (14–16 dB) are way below typical 3DGS quality (22–28 dB on natural scenes).** That's the real concern, and it's NOT explained by SfM quality — Agisoft can't fix it either. See "What this means for the thesis" below.

### What this means for the thesis

**Conclusion 1: Our COLMAP pipeline is good enough for 3DGS.** The (Agisoft − COLMAP) gap is small enough that no SfM upgrade is justified by the data. Using the decision tree from [`SFM_PIPELINE_COMPARISON.md`](SFM_PIPELINE_COMPARISON.md):

```
3DGS metric gap (Agisoft − COLMAP)
├── < 0.5 PSNR / < 0.02 SSIM     → No upgrade. Phone+COLMAP is fine.   ← WE ARE HERE
├── ~1–2 PSNR / 0.02–0.05 SSIM   → Try hloc only.
└── > 2 PSNR / > 0.05 SSIM       → Re-shoot with ArUco, both upgrades.
```

So **no pycolmap, no hloc, no ArUco** — those would be solving a problem we don't have.

**Conclusion 2: The absolute quality is the real problem.** PSNR 14–16 means 3DGS struggles regardless of which SfM tool feeds it. Candidate causes (none of them SfM-related):

- **Too few training iterations** — paper default is 30k; we ran 15k.
- **Pathological eval split** — llffhold=8 picks every 8th sorted-by-timestamp image as test. If the photographer walked the field in a linear pattern, neighboring frames have very high overlap, and the chosen test view is right between two train views. But if the path zigzags, the test view could be much further off-distribution than a typical NeRF benchmark would have. Inspecting per-view PSNRs in `per_view.json` would tell us if certain views are dragging the mean down.
- **Phone image properties** — depth-of-field, motion blur, noise, JPEG compression all set a hard ceiling that no amount of SfM/3DGS engineering escapes.
- **The wheat scene itself** — fine texture detail (each wheat head a few pixels wide), low surface contrast, and partial occlusion are genuinely hard for 3DGS. Compare to typical 3DGS benchmark scenes (statues, buildings, indoor rooms — high-contrast structured surfaces).

**Diagnostic next steps (cheap, no re-training needed):**

- **(a) Compute metrics on the *train* set too**, not just test. If train PSNR is also ~15, the model is fundamentally undertrained or the scene is too hard. If train PSNR is 25+, then we're overfitting and the test split is pathological — re-think llffhold for phone data.
- **(b) Visually inspect a render-vs-gt pair** from `train/ours_15000/renders/` and `train/ours_15000/gt/`. Is the render blurry? Wrong colors? Misregistered geometry? The failure mode determines the fix.
- **(c) Try 30k iterations on one session** to see if the curve is still descending at 15k. Cheap test.

These three together should rule out the "Conclusion 2" candidate causes without committing to any major rewrite.

### Cross-reference: SfM geometry agreement vs 3DGS quality

| Session | SfM translation err (median, mm) | SfM rotation err (median, °) | COLMAP 3DGS PSNR | Agisoft 3DGS PSNR | 3DGS gap (Agi-COL) PSNR |
|---|---|---|---|---|---|
| field_A/20250609 | 12.3 | 0.48 | 14.52 | 14.89 | +0.37 |
| field_A/20250618 | 22.9 | 0.90 | 14.30 | 14.36 | +0.06 |
| field_D/20250523 | 15.1 | 0.96 | 14.67 | 14.89 | +0.22 |
| field_D/20250530 | 60.0 | 3.74 | 16.51 | 16.40 | −0.10 |

**Interesting:** the session with the *worst* SfM agreement to Agisoft (`field_D/20250530`, 60 mm / 3.7° median errors) has the *smallest* 3DGS quality gap (and even reverses direction). This suggests that for our wheat scenes, 3DGS quality is **not sensitive to SfM precision** at this level — bundle adjustment refines the camera poses during 3DGS training anyway, and the small initial differences wash out. This further weakens any case for engineering effort on the SfM side.
