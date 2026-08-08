"""Single source of truth for the 3DGS train/test (eval) split.

Both training (gaussians/scene/dataset_readers.py) and the standalone checker
(preprocessing/check_split.py) call compute_eval_split() so a verification can never
drift from what training actually does.

Split rules, in priority order:
  1. PINNED test list — split by NAME membership (identity). This is robust to
     COLMAP registration drift and to image-count differences across methods, so two
     reconstructions of the same session always test on the same physical views.
     Pin sources: FIP transforms.json `test_filenames` (the paper's split), or
     phone phone_split.json `test_views` (made by make_phone_split.py from input_uniform/).
     Membership is suffix-robust (see _norm_stem) so the SAME pin matches both our COLMAP
     naming and Agisoft's `_<N>` ingestion-renamed naming — the two R1 arms hold out the
     same physical views.
  2. FIP naming fallback — every name ends in `_cam_NN`: cam_NN > 10 = test
     (cam_11, cam_12). Matches transforms.json, kept for safety if the pin is absent.
  3. llffhold-8 fallback — phone with no pin: every 8th sorted image = test.
"""
import os
import re
import json

# FIP image names look like FPWW036_SR0461_FIP2_cam_11 — the cam index is the tail.
FIP_PAT = re.compile(r'_cam_\d+$')

# A full phone image stem: IMG_<date8>_<time6> with an optional ~N dedup marker. Used to decide
# whether a trailing _<N> is Agisoft's ingestion index (safe to strip) vs the time group (keep).
_PHONE_STEM = re.compile(r'^IMG_\d{8}_\d{6}(?:~\d+)?$')


def _stem(name):
    """basename without extension — matches dataset_readers' image_name (split on the first dot)."""
    return os.path.basename(str(name)).split(".")[0]


def _norm_stem(name):
    """Stem with a trailing Agisoft ingestion index (_0, _37, ...) removed, so the SAME physical view
    matches across our COLMAP naming (IMG_20250627_102602) and Agisoft's (IMG_20250627_102602_0).
    Only strips when the remainder is a full phone stem, so it never eats the 6-digit time group,
    FIP _cam_NN, or ~N dedup markers — for those it returns the plain stem unchanged."""
    s = _stem(name)
    m = re.match(r'^(.*)_\d+$', s)
    if m and _PHONE_STEM.match(m.group(1)):
        return m.group(1)
    return s


def pin_applies(image_names, pin_test):
    """True iff the pin matches at least one of image_names (suffix-robust). Lets callers detect the
    'pin found but different naming' case without duplicating the normalization."""
    if not pin_test:
        return False
    pin = {_norm_stem(n) for n in pin_test}
    return any(_norm_stem(n) in pin for n in image_names)


def pin_missing(image_names, pin_test):
    """Pinned test views (normalized) that are NOT present in image_names — the comparability killer
    when a reconstruction dropped a held-out view. Empty if no pin. Suffix-robust like pin_applies."""
    if not pin_test:
        return []
    present = {_norm_stem(n) for n in image_names}
    return sorted({_norm_stem(n) for n in pin_test} - present)


def load_pin_test(source_path):
    """Return the set of pinned TEST stems for this reconstruction, or None if there is no pin.

    Looks for transforms.json (FIP, key `test_filenames`) then phone_split.json (phone, key
    `test_views`). Searches source_path AND its parent so an agisoft/ subfolder can share the
    session-level phone_split.json. Returns None on any read/parse problem (caller falls back)."""
    base_dir = os.path.normpath(source_path)
    for base in (base_dir, os.path.dirname(base_dir)):
        tj = os.path.join(base, "transforms.json")
        if os.path.isfile(tj):
            try:
                d = json.load(open(tj))
                if d.get("test_filenames"):
                    return {_stem(f) for f in d["test_filenames"]}
            except Exception:
                pass
        pj = os.path.join(base, "phone_split.json")
        if os.path.isfile(pj):
            try:
                d = json.load(open(pj))
                if d.get("test_views"):
                    return {_stem(f) for f in d["test_views"]}
            except Exception:
                pass
    return None


def is_fip_naming(names):
    """True when EVERY name ends in _cam_NN (so the FIP cam-index split applies)."""
    names = [_stem(n) for n in names]
    return bool(names) and all(FIP_PAT.search(n) for n in names)


def compute_eval_split(image_names, pin_test=None, llffhold=8):
    """Split image names into (train, test) lists of stems. Deterministic (sorts by name first).

    pin_test: optional iterable of test names/paths → split purely by membership (identity).
    No pin → FIP cam-index split if all names are FIP-style, else llffhold-8. This is the exact
    logic training uses; the checker imports the same function so the two cannot disagree."""
    names = sorted(_stem(n) for n in image_names)
    if pin_test:
        # suffix-robust membership: normalize both sides so the same pin (COLMAP stems) matches
        # Agisoft's _<N>-renamed views. Returned lists keep the ACTUAL stems so dataset_readers can
        # still match them to the on-disk filenames.
        pin = {_norm_stem(n) for n in pin_test}
        train = [n for n in names if _norm_stem(n) not in pin]
        test = [n for n in names if _norm_stem(n) in pin]
        return train, test
    if is_fip_naming(names):
        train = [n for n in names if int(n.split('_')[-1]) <= 10]
        test = [n for n in names if int(n.split('_')[-1]) > 10]
    else:
        train = [n for i, n in enumerate(names) if i % llffhold != 0]
        test = [n for i, n in enumerate(names) if i % llffhold == 0]
    return train, test


def split_method_label(image_names, pin_test=None):
    """Short human label for which rule produced the split — used in log lines / reports."""
    if pin_test:
        return "pinned (transforms.json / phone_split.json)"
    return "FIP cam-index (cam_11/cam_12 = test)" if is_fip_naming(image_names) else "llffhold-8"


def read_colmap_image_names(sparse_dir):
    """Read the registered image names (stems) from a COLMAP sparse model.

    sparse_dir holds images.txt/images.bin (e.g. <path>/sparse/0). Prefers images.txt — its
    non-comment lines alternate header/points2D, so every 2nd line's last token is the filename —
    which keeps this dependency-free (no torch). Falls back to the binary reader only if there is
    no txt, so check_split.py stays lightweight in the common (export_text) case."""
    txt_path = os.path.join(sparse_dir, "images.txt")
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            nonc = [l for l in f if l.strip() and not l.startswith("#")]
        return sorted({_stem(nonc[i].split()[-1]) for i in range(0, len(nonc), 2)})
    bin_path = os.path.join(sparse_dir, "images.bin")
    if os.path.isfile(bin_path):
        from gaussians.scene.colmap_loader import read_extrinsics_binary
        extr = read_extrinsics_binary(bin_path)
        return sorted({_stem(im.name) for im in extr.values()})
    raise FileNotFoundError(f"no images.txt or images.bin in {sparse_dir}")
