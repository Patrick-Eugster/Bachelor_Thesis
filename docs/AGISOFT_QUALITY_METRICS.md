# Agisoft Quality Metrics — 3D Error vs Distance Error

Quick reference for what the marker-error metrics in `marker_errors_summary.csv` actually mean. This is the CSV produced by the supervisor's `7-agisoft_compute_marker_errors.py` script and visualized in `10_agisoft_calibration_quality_analysis.ipynb`. The CSV is the only practical way to pick which Agisoft `sparse/` sessions to trust as benchmarking references for our COLMAP output — see [`SFM_PIPELINE_COMPARISON.md`](SFM_PIPELINE_COMPARISON.md) for that context.

The two main metrics are **3D Error** and **Distance Error**. They sound similar but measure different things.

---

## 3D Error (m)

**Per single marker.** Where Agisoft *thinks* the marker is in 3D space, vs. where it *actually* is (surveyed real-world coordinates).

```
Marker 1:
   surveyed GPS position:     (X=2600100.123, Y=1200200.456, Z=540.789)   ← ground truth
   reconstructed position:    (X=2600100.135, Y=1200200.468, Z=540.781)   ← Agisoft's answer

   3D Error for marker 1 = distance between those two points = 0.018 m = 18 mm
```

Repeat for every marker, take the average → that's `3D Error (m) mean` in the CSV.

**One marker. Absolute position. In real-world coordinates (Swiss CH1903+/LV95, EPSG:2056).**

The ground truth comes from the supervisor's surveyed marker CSV — each marker's `(Easting, Northing, Elevation)` measured once, presumably with GPS or differential surveying.

---

## Distance Error (m)

**Per pair of markers.** Distance between two markers as reconstructed, vs. distance between them as measured by hand.

```
Marker pair (1 ↔ 2):
   ruler measurement:              4.000 m   ← ground truth
   reconstructed distance:         4.018 m   ← Agisoft's answer

   Distance Error for this pair = |4.018 − 4.000| = 0.018 m = 18 mm
```

Repeat for every marker pair, take the average → that's `Avg Distance Error (m)` in the CSV.

**Two markers. Relative distance between them. No GPS involved.**

The ground truth comes from `metadata/markers/Demoanlage-2025-markers-manual-distances.xlsx` — the supervisor measured every marker pair once with a ruler/tape when the markers were first installed. Since the markers are permanent physical objects that don't move between sessions, these reference distances are **fixed for the whole study** — the same numbers are used to evaluate every session's reconstruction.

---

## The crucial difference

| | 3D Error | Distance Error |
|---|---|---|
| Operates on | one marker at a time | a pair of markers |
| Compares to | surveyed real-world XYZ (needs GPS) | ruler measurement (just a tape) |
| Coordinate frame | absolute (Swiss CH1903+) | relative (just a length) |
| What it catches | global drift / wrong georef | distorted internal geometry / wrong scale |
| Depends on GPS quality? | yes | no |

---

## Why a reconstruction can have low Distance Error but high 3D Error

A reconstruction can have all the **relative distances correct** but the **whole thing shifted/rotated in space**:

```
real markers:               ●───4m───●───4m───●     (at positions A, B, C in Switzerland)
reconstruction:                       ●───4m───●───4m───●   ← all distances perfect (Dist Err = 0)
                                                              ← but shifted 50 cm east (3D Err = 0.5 m)
```

Internal geometry: perfect. Absolute georef: drifted. This happens when GPS / marker survey positions are noisy but the SfM is doing a good job internally.

The other way around — high Distance Error but low 3D Error — is essentially impossible: if every marker is in its correct absolute position, then distances between them must also be correct. So they usually move together, but 3D Error is the noisier of the two because it inherits any error in the surveyed reference positions.

---

## Why the same field gets different errors on different dates

The markers don't move between captures. The ruler measurements are constant. **What varies is the reconstruction's accuracy on each session**, driven by capture conditions:

- Number of images captured (more = stronger constraint)
- Camera angles and how well they cover the plot
- Lighting (sun angle, clouds → affects feature contrast)
- Wind moving plants between shots (breaks feature consistency)
- How many markers were clearly visible in each image
- Image sharpness, focus, motion blur
- Random initialization of bundle adjustment

So same plot, same markers, very different errors across sessions. That's why the CSV exists — capture-day quality dominates, not the field itself. Pick sessions by their per-session quality metrics, not by field name.

Example: field_D/20250530 has 8.4 mm Distance Error (excellent capture day). field_D/20250618 has 47.4 mm Distance Error (something went wrong that day, even though the field and markers were identical).

---

## Which one matters for benchmarking our COLMAP output

**Distance Error.** Because:

- It doesn't depend on GPS quality — which we don't have for our COLMAP output anyway.
- It directly tests the property we care about: did the reconstruction get the relative geometry right at the correct scale?
- For phenotyping (measuring wheat heads in real units), correct relative distances are what we need — not absolute georeferencing.

3D Error is useful only if you specifically need to place the reconstruction at the right real-world location (e.g. lining up with another georeferenced dataset). For us it's secondary.

When picking Agisoft sessions to benchmark our COLMAP against, sort by `Avg Distance Error (m)` ascending and pick the lowest. That's the cleanest reference.

---

## Quick interpretation guide

| Distance Error | Quality |
|---|---|
| **< 10 mm** | Excellent — trustworthy reference for benchmarking |
| **10–20 mm** | Good — usable |
| **20–50 mm** | Mediocre — reconstruction is somewhat distorted |
| **> 50 mm** | Bad — avoid for ground truth |
| **> 1 m** | Reconstruction is essentially broken (e.g. field_C/20250613 has 2.9 m) |

| Reproj Error | Quality |
|---|---|
| **1–2 px** | Tight — camera poses and 3D positions agree very well |
| **3–5 px** | Acceptable |
| **> 10 px** | Geometric inconsistency, likely bad feature matches |
| **> 100 px** | Reconstruction has degenerate solution; ignore |

(For reference, `Reproj Error (px)` is the third metric in the CSV — it measures how well the 3D model and the 2D detections agree, purely internally. No external ground truth needed. Lower = camera poses and 3D positions are mutually consistent. Useful as a diagnostic but doesn't directly measure "is the reconstruction correct.")
