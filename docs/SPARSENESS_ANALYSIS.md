# SfM sparseness analysis — FIP vs phone

**Goal:** decide, per dataset, whether the 3DGS densification should lean toward **MCMC**
(robust on sparse / limited views) or **AbsGrad** (recovers fine detail on better-constrained
data). See [`DENSIFICATION_OPTIONS.md`](DENSIFICATION_OPTIONS.md) for what those methods are.

**Script:** [`src/analysis/analyze_sparseness.py`](../src/analysis/analyze_sparseness.py) (read-only).
**Raw numbers:** [`analysis_results/sparseness.json`](analysis_results/sparseness.json).
Re-run with `python src/analysis/analyze_sparseness.py`.

## Results

| dataset | imgs | pts3D | track len (mean/med) | % seen by only 2 imgs | obs/img | reproj px | view angle° |
|---|--:|--:|--:|--:|--:|--:|--:|
| fip/plot_461 | 36 | 31949 | 3.35 / 2.0 | 61% | 2977 | 0.00 | 13.2 |
| fip/plot_462 | 36 | 33954 | 3.45 / 2.0 | 60% | 3251 | 0.00 | 13.2 |
| fip/plot_463 | 36 | 33713 | 3.35 / 2.0 | 60% | 3133 | 0.00 | 13.2 |
| fip/plot_464 | 36 | 30848 | 3.79 / 2.0 | 59% | 3246 | 0.00 | 13.2 |
| fip/plot_465 | 36 | 34305 | 3.43 / 2.0 | 61% | 3266 | 0.00 | 13.2 |
| fip/plot_466 | 36 | 33806 | 3.53 / 2.0 | 59% | 3316 | 0.00 | 13.2 |
| fip/plot_467 | 36 | 34377 | 3.43 / 2.0 | 58% | 3273 | 0.00 | 13.2 |
| phone/field_A/20250609 | 119 | 7523 | 3.52 / 3.0 | 22% | 223 | 1.24 | 70.5 |
| phone/field_A/20250618 | 93 | 8085 | 3.35 / 3.0 | 28% | 291 | 1.32 | 67.5 |
| phone/field_D/20250523 | 64 | 7388 | 3.07 / 3.0 | 35% | 354 | 1.29 | 65.2 |
| phone/field_D/20250530 | 63 | 9348 | 3.14 / 3.0 | 28% | 465 | 1.24 | 63.0 |

## What each column means

- **track len** = how many images observe each 3D point. The core multi-view-overlap signal.
  Higher = each point is better triangulated.
- **% seen by only 2 imgs** = fraction of 3D points with the *minimum* possible track (2 views) —
  these are the **fragile, weakly-triangulated** points. Higher = more fragile geometry.
- **obs/img** = how many of an image's 2D keypoints matched a 3D point. Higher = denser 2D
  feature support per image.
- **reproj px** = mean reprojection error. (FIP shows 0.00 because the FIP `sparse/` is exported
  from Agisoft, which doesn't populate the per-point error field — ignore it for FIP.)
- **view angle°** = mean pairwise angle between camera optical axes. **Low = all cameras look
  from a similar direction (angularly LIMITED); high = diverse viewpoints.**

## The headline: FIP and phone are sparse in OPPOSITE ways

**FIP = narrow-angle but dense-matching.**
- View angle only **13.2°** → all 36 cameras sit in a tight overhead cone (the cable-rig captures
  top-down). This is the classic **"limited views"** situation — almost no angular diversity, so
  depth/geometry is weakly constrained.
- But **~3000 obs/img** and ~32k points → 2D feature matching is rich (the coded ground markers +
  high-res 12 MP help).
- **~60% of points are 2-view only** → lots of fragile geometry despite the dense matching.

**Phone = wide-angle but sparse-matching.**
- View angle **63–70°** → walking over the plot captures genuinely diverse directions → geometry
  is *better* angularly constrained than FIP.
- But only **~220–465 obs/img** and ~7–9k points → far sparser 2D feature support (repetitive
  vegetation + handheld + no markers give COLMAP much less to match).
- Fewer 2-view points (22–35%), so the points that *do* survive are a bit better triangulated.

### Direct answer to "more phone images ⇒ denser?"
**No.** `field_A/20250609` has the **most** images (119) yet the **fewest** observations/image
(223) and only 7.5k points. More images did **not** make it denser in the sense 3DGS cares about
(multi-view feature support per surface point). Image count and the thing that constrains 3DGS
(overlap + matchable features) are different axes — this dataset proves it.

## What this means for densification choice

Both datasets are **under-constrained**, just along different axes — and both have a property that
favors **MCMC** somewhere:

- **FIP** is the textbook **limited-views** case (13° cone, 60% 2-view points). MCMC's robustness
  on limited angular coverage is most likely to pay off here, and its fixed budget tames the
  unbounded Gaussian growth that over-fits the narrow viewpoint range. → **MCMC is the prime
  candidate on FIP.**
- **Phone** has good angular coverage but sparse feature support and few 3D points → the risk is
  default densification **over-spawning** Gaussians in weakly-matched vegetation. MCMC's
  "don't exceed the budget, relocate instead" also helps here, while **AbsGrad** is attractive
  for pulling the fine wheat-head/awn detail out of the regions that *are* well-supported.

**Recommended experiment (no code yet):** run the 3-way A/B/C (vanilla Default vs Default+AbsGrad
vs MCMC) on **one FIP plot and one phone session** — they stress different sparseness regimes, so
testing both tells us whether one method wins universally or whether FIP and phone want different
strategies. Compare PSNR/SSIM/LPIPS, training time, peak VRAM, **and** the segmentation
`eval_2d/metrics_2d.json` (detail that doesn't survive into correct head segmentation isn't worth
much).

## Caveats

- `view angle°` is a coarse proxy (mean pairwise optical-axis angle); it captures "overhead cone
  vs orbit" well but not occlusion or baseline length.
- FIP `reproj px` is unavailable (Agisoft export), so don't compare reprojection across the two
  datasets — only within phone.
- Track length here counts the SfM tracks, not the final 3DGS visibility; it's an input-quality
  proxy, which is exactly what we want for choosing a densification prior.
