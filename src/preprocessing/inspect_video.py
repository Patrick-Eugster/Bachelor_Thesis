"""Read-only diagnostic: inspect MP4 (or any) video metadata via ffprobe.

Tells you everything that matters for deciding whether to extract frames from a
video for COLMAP / 3DGS: codec, resolution, pixel format + bit depth + chroma
subsampling, bitrate, fps, duration, total frames, and the I-frame (keyframe)
count / GOP size. Optionally compares the video resolution against one of your
current JPGs to catch a downscale that happened during extraction.

Does NOT modify or extract anything — pure metadata read.

Usage:
    # one file
    python src/preprocessing/inspect_video.py input_plots/phone/field_A/20250609/video/clip.mp4

    # a whole folder (scans recursively for video files)
    python src/preprocessing/inspect_video.py input_plots/phone/field_A/20250609/video/

    # also break down I / P / B frame counts (slower — decodes the stream)
    python src/preprocessing/inspect_video.py path/to/clip.mp4 --frame-types

    # compare video res against a current extracted jpg (detect downscaling)
    python src/preprocessing/inspect_video.py path/to/clip.mp4 --compare-jpg path/to/IMG_0001.jpg

    # write the report as JSON next to the video(s)
    python src/preprocessing/inspect_video.py path/to/video/ --json-out
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# file extensions we treat as videos when scanning a folder
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts"}


def check_ffprobe():
    """Make sure ffprobe is on PATH, otherwise nothing here works.
    Exits with a clear message instead of a cryptic subprocess error."""
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not found. Install ffmpeg first (e.g. `apt-get install ffmpeg`).")
        sys.exit(1)


def run_ffprobe_json(args):
    """Run ffprobe with -of json and return the parsed dict.
    args is the list of ffprobe flags AFTER the program name."""
    cmd = ["ffprobe", "-v", "error", "-of", "json"] + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        # ffprobe writes the real reason to stderr — surface it, don't swallow it
        raise RuntimeError(f"ffprobe failed: {' '.join(cmd)}\n{out.stderr.strip()}")
    return json.loads(out.stdout)


def parse_fraction(s):
    """ffprobe gives rates/ratios as 'num/den' strings (e.g. '30000/1001').
    Turn that into a float, guarding against the '0/0' that shows up for some streams."""
    try:
        num, den = s.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except (ValueError, AttributeError):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0


def pix_fmt_info(pix_fmt):
    """Map a libav pixel-format name to (bit depth, chroma subsampling) for the report.
    We only need the common phone-video cases; anything else is reported as-is."""
    if not pix_fmt:
        return "?", "?"
    # bit depth: '...10le' / '...10be' means 10-bit, otherwise assume 8-bit
    depth = "10-bit" if "10" in pix_fmt else ("12-bit" if "12" in pix_fmt else "8-bit")
    # chroma subsampling lives in the name: yuv420 / yuv422 / yuv444
    if "420" in pix_fmt:
        chroma = "4:2:0 (chroma half-res)"
    elif "422" in pix_fmt:
        chroma = "4:2:2"
    elif "444" in pix_fmt:
        chroma = "4:4:4 (full chroma)"
    else:
        chroma = pix_fmt
    return depth, chroma


def count_keyframes(path):
    """Count I-frames (keyframes) by reading PACKET flags only — no decoding, so this is fast.
    Keyframe packets carry a 'K' in their flags field. Returns (keyframes, total_packets)."""
    data = run_ffprobe_json([
        "-select_streams", "v:0",
        "-show_entries", "packet=flags",
        str(path),
    ])
    packets = data.get("packets", [])
    keys = sum(1 for p in packets if "K" in p.get("flags", ""))
    return keys, len(packets)


def count_frame_types(path):
    """Break down I / P / B frame counts by reading FRAME pict_type — this decodes the
    stream so it's slower. Only called when --frame-types is passed. Returns a dict."""
    data = run_ffprobe_json([
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        str(path),
    ])
    counts = {"I": 0, "P": 0, "B": 0, "?": 0}
    for f in data.get("frames", []):
        t = f.get("pict_type", "?")
        counts[t] = counts.get(t, 0) + 1
    return counts


def get_jpg_resolution(jpg_path):
    """Read a JPG's pixel dimensions with ffprobe (no extra deps).
    Used to compare current extracted-frame resolution against the source video."""
    data = run_ffprobe_json([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        str(jpg_path),
    ])
    s = data["streams"][0]
    return int(s["width"]), int(s["height"])


def human_size(num_bytes):
    """Pretty-print a byte count as KB/MB/GB."""
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def inspect_one(path, do_frame_types=False, compare_jpg=None):
    """Probe a single video file and return a dict of everything we care about.
    Keeps the raw numbers; the printing is done separately so JSON stays clean."""
    info = {"file": str(path), "size_bytes": path.stat().st_size}

    data = run_ffprobe_json(["-show_format", "-show_streams", str(path)])
    fmt = data.get("format", {})
    # pick the first video stream
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not vstreams:
        info["error"] = "no video stream"
        return info
    v = vstreams[0]

    width, height = int(v.get("width", 0)), int(v.get("height", 0))
    depth, chroma = pix_fmt_info(v.get("pix_fmt"))
    duration = float(fmt.get("duration", 0) or v.get("duration", 0) or 0)
    # avg_frame_rate is the real average; r_frame_rate is the base/ideal rate
    fps = parse_fraction(v.get("avg_frame_rate", "0/0")) or parse_fraction(v.get("r_frame_rate", "0/0"))
    # nb_frames is often present for mp4; otherwise estimate from fps * duration
    nb_frames = v.get("nb_frames")
    nb_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else (int(round(fps * duration)) if fps and duration else 0)

    info.update({
        "container": fmt.get("format_name", "?"),
        "codec": v.get("codec_name", "?"),
        "codec_long": v.get("codec_long_name", "?"),
        "profile": v.get("profile", "?"),
        "width": width,
        "height": height,
        "megapixels": round(width * height / 1e6, 2),
        "pix_fmt": v.get("pix_fmt", "?"),
        "bit_depth": depth,
        "chroma": chroma,
        "fps": round(fps, 3),
        "duration_s": round(duration, 2),
        "frames_estimated": nb_frames,
        "video_bitrate": v.get("bit_rate") or fmt.get("bit_rate"),
        "has_audio": len(astreams) > 0,
    })

    # bits-per-pixel-per-frame: a rough "how hard is this compressed" number.
    # photo JPEGs sit ~1-2 bpp; phone video is typically ~0.05-0.15 bpp (much harsher).
    br = info["video_bitrate"]
    if br and fps and width and height:
        info["bits_per_pixel"] = round(float(br) / (fps * width * height), 4)

    # keyframe / GOP info (fast, packet-based)
    try:
        keys, total_pkts = count_keyframes(path)
        info["keyframes"] = keys
        info["packets"] = total_pkts
        info["avg_gop"] = round(total_pkts / keys, 1) if keys else None
    except RuntimeError as e:
        info["keyframe_error"] = str(e)

    # optional full I/P/B breakdown (slow)
    if do_frame_types:
        try:
            info["frame_types"] = count_frame_types(path)
        except RuntimeError as e:
            info["frame_types_error"] = str(e)

    # optional comparison against a current extracted jpg
    if compare_jpg:
        try:
            jw, jh = get_jpg_resolution(compare_jpg)
            # classify the relationship instead of only checking exact equality:
            #  - same res            -> jpg is the full video frame
            #  - different aspect    -> NOT a frame of this video (different source)
            #  - jpg bigger, same AR -> jpg is a higher-res still, not a downscale
            #  - jpg smaller, same AR-> jpg was downscaled from the video (extra loss)
            v_ar = round(width / height, 3) if height else 0
            j_ar = round(jw / jh, 3) if jh else 0
            same_ar = abs(v_ar - j_ar) < 0.02
            if jw == width and jh == height:
                verdict = "same_resolution"
            elif not same_ar:
                verdict = "different_aspect"
            elif jw * jh > width * height:
                verdict = "jpg_larger"
            else:
                verdict = "downscaled"
            info["jpg_compare"] = {
                "jpg": str(compare_jpg), "jpg_width": jw, "jpg_height": jh,
                "jpg_aspect": j_ar, "video_aspect": v_ar, "verdict": verdict,
            }
        except (RuntimeError, KeyError, IndexError) as e:
            info["jpg_compare"] = {"error": str(e)}

    return info


def print_report(info):
    """Print a boxed, human-readable summary of one video's metadata,
    matching the yolo_sam-style SUMMARY blocks used elsewhere in the project."""
    line = "=" * 64
    print(line)
    print(f" VIDEO: {Path(info['file']).name}")
    print(line)
    if "error" in info:
        print(f"  ERROR: {info['error']}")
        print(line)
        return

    print(f"  container     : {info['container']}")
    print(f"  codec         : {info['codec']}  ({info['profile']})")
    print(f"  resolution    : {info['width']} x {info['height']}  ({info['megapixels']} MP)")
    print(f"  pixel format  : {info['pix_fmt']}  -> {info['bit_depth']}, {info['chroma']}")
    print(f"  fps           : {info['fps']}")
    print(f"  duration      : {info['duration_s']} s")
    print(f"  frames (~)    : {info['frames_estimated']}")
    if info.get("video_bitrate"):
        mbps = float(info["video_bitrate"]) / 1e6
        print(f"  video bitrate : {mbps:.2f} Mbps")
    if "bits_per_pixel" in info:
        print(f"  bits/pixel    : {info['bits_per_pixel']}  (photo JPEG ~1-2 ; phone video ~0.05-0.15)")
    print(f"  file size     : {human_size(info['size_bytes'])}")
    print(f"  audio track   : {'yes' if info.get('has_audio') else 'no'}")

    if "keyframes" in info:
        print(f"  keyframes (I) : {info['keyframes']} of {info['packets']} packets"
              f"  (avg GOP {info['avg_gop']})")
        print(f"                  -> only ~{info['keyframes']} clean standalone frames; "
              f"the rest are predicted (motion-blur prone)")
    if "frame_types" in info:
        ft = info["frame_types"]
        print(f"  frame types   : I={ft.get('I',0)}  P={ft.get('P',0)}  B={ft.get('B',0)}")

    if "jpg_compare" in info:
        jc = info["jpg_compare"]
        if "error" in jc:
            print(f"  jpg compare   : ERROR ({jc['error']})")
        else:
            jres = f"{jc['jpg_width']}x{jc['jpg_height']}"
            vres = f"{info['width']}x{info['height']}"
            v = jc["verdict"]
            if v == "same_resolution":
                print(f"  jpg compare   : MATCH — jpg {jres} == video res (likely a full video frame)")
            elif v == "different_aspect":
                print(f"  jpg compare   : DIFFERENT SOURCE — jpg {jres} (AR {jc['jpg_aspect']}) "
                      f"vs video {vres} (AR {jc['video_aspect']}); aspect ratios differ "
                      f"-> jpg is NOT a frame of this video (native photo)")
            elif v == "jpg_larger":
                print(f"  jpg compare   : jpg {jres} is HIGHER-res than video {vres} (same AR) "
                      f"-> jpg is a native still, not a downscale of this video")
            else:  # downscaled
                print(f"  jpg compare   : DOWNSCALED — jpg {jres} < video {vres} (same AR) "
                      f"-> extracted jpg lost resolution (extra quality loss!)")
    print(line)


def collect_videos(path):
    """Return a sorted list of video files. If path is a file, just that file;
    if a directory, every video-extension file under it (recursive)."""
    if path.is_file():
        return [path]
    vids = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return vids


def main():
    """Parse args, probe each video, print reports, optionally dump JSON."""
    ap = argparse.ArgumentParser(description="Read-only video metadata inspector (ffprobe).")
    ap.add_argument("path", type=Path, help="a video file OR a folder of videos")
    ap.add_argument("--frame-types", action="store_true",
                    help="also count I/P/B frames (decodes the stream — slower)")
    ap.add_argument("--compare-jpg", type=Path, default=None,
                    help="a current extracted jpg to compare resolution against")
    ap.add_argument("--json-out", action="store_true",
                    help="write <video>.videoinfo.json next to each video")
    args = ap.parse_args()

    check_ffprobe()

    if not args.path.exists():
        print(f"ERROR: path not found: {args.path}")
        sys.exit(1)

    videos = collect_videos(args.path)
    if not videos:
        print(f"No video files found under: {args.path}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to inspect.\n")
    all_info = []
    for vid in videos:
        info = inspect_one(vid, do_frame_types=args.frame_types, compare_jpg=args.compare_jpg)
        print_report(info)
        all_info.append(info)
        if args.json_out:
            out = vid.with_suffix(vid.suffix + ".videoinfo.json")
            out.write_text(json.dumps(info, indent=2))
            print(f"  wrote {out}")
        print()


if __name__ == "__main__":
    main()
