"""Shared coded-target code utilities: rotation canonicalization, Hamming, and ID filtering.

The 12-bit coded markers carry an integer code read off a ring of 12 samples. Because the marker
is a circle with no fixed "up", the decoder canonicalizes each read to the MINIMUM integer over all
12 rotations (see cctdecode B2I) — so every code we handle here is already rotation-canonical.

Used by the detectors (v8 ID filter) and later by triangulation (same manifest / Hamming logic).
Full write-up: docs/MARKER_CODE_STRUCTURE.md.

IMPORTANT: the set of "legal necklaces" (352 rotation-canonical 12-bit values) does NOT separate
real markers from junk — the all-ones junk codes (511, 1023, ...) are themselves legal necklaces, and
every decoded code is canonical so it always passes. The real junk filter is the per-plot MANIFEST
(the actual codes deployed, e.g. decoded from the spec PDF). The necklace set + Hamming helpers are
kept for the future generator tool (pick codes far apart) and for flagging near-neighbour misreads.
"""

N_BITS = 12


def _rotations(v, N=N_BITS):
    """All N cyclic left-rotations of the N-bit value v."""
    return [((v << r) | (v >> (N - r))) & ((1 << N) - 1) for r in range(N)]


def canonicalize(v, N=N_BITS):
    """Rotation-canonical form = the smallest integer over all N rotations (matches cctdecode B2I)."""
    return min(_rotations(v, N))


def legal_necklaces(N=N_BITS):
    """The set of rotation-canonical N-bit codes (binary necklaces). 352 for N=12.
    NOTE: this is NOT the junk filter — see module docstring."""
    return {canonicalize(v, N) for v in range(1 << N)}


def hamming(a, b, N=N_BITS):
    """Rotation-aware Hamming distance: min bit-difference over all rotations of a vs b.
    Needed because codes are canonical-minimum forms, not aligned bit strings."""
    return min(bin(r ^ b).count("1") for r in _rotations(a, N))


def kept_ids(id_views, mode, manifest=None, keep_top_k=0, min_views=1):
    """Decide which decoded IDs to keep, given how many views each was seen in.

    mode:
      'manifest' — keep only IDs in the deployed code set (drops junk + near-neighbour misreads
                   like 117; the principled default, manifest sourced from the spec PDF).
      'view'     — keep the keep_top_k most-seen IDs above the min_views floor (legacy heuristic;
                   fragile when a junk ID is seen in many views).
      'none'     — keep everything.
    min_views is an absolute floor applied in every mode."""
    if mode == "none":
        return set(id_views)
    if mode == "manifest":
        man = {int(x) for x in (manifest or [])}
        return {i for i in id_views if int(i) in man and id_views[i] >= min_views}
    if mode == "view":
        order = sorted(id_views, key=lambda k: -id_views[k])
        if keep_top_k and keep_top_k > 0:
            order = order[:keep_top_k]
        return {i for i in order if id_views[i] >= min_views}
    raise ValueError(f"unknown id_filter mode: {mode}")


def split_detections(per_image, kept):
    """Split per-image detections into (kept, dropped) by the kept-id set. Dropped keeps locations
    so a wrongly-filtered marker can be recovered later."""
    keep = {int(k) for k in kept}
    filtered = {f: [d for d in dets if int(d["id"]) in keep] for f, dets in per_image.items()}
    dropped = {f: [d for d in dets if int(d["id"]) not in keep]
               for f, dets in per_image.items()
               if any(int(d["id"]) not in keep for d in dets)}
    return filtered, dropped
