# Building COLMAP from Source with CUDA Support

This guide walks you through building COLMAP with CUDA SIFT support on Ubuntu (inside Docker or natively). The Ubuntu apt package (`apt install colmap`) is CPU-only — for GPU acceleration (5–8× faster SIFT) you need to build from source.

Designed to work end-to-end without errors on a fresh container. Every dependency we hit during our build (May 2026, COLMAP 4.1.0.dev0 on Ubuntu 24.04 + CUDA 13.1) is pre-installed in Step 2 below, so you should not run into the same trial-and-error we did.

**Tested on:** Ubuntu 24.04 inside Docker on WSL2, RTX 5070 Ti (Blackwell, sm_120), CUDA 13.1, NVIDIA driver 591.86. Should also work on Ubuntu 22.04 with CUDA 11.8+ on Ada/Ampere GPUs.

**Total time:** ~20–30 minutes (mostly the compile step).

---

## Prerequisites

Before you start, verify that **CUDA is fully set up inside your environment** (Docker container or host). Run these three checks:

```bash
nvidia-smi
```
Expected output: a table showing your GPU + driver + CUDA version. If this errors with "command not found" or "no devices found", install/fix your NVIDIA driver first (and for Docker, make sure the container was launched with `--gpus all`).

```bash
nvcc --version
```
Expected output: something like `Cuda compilation tools, release 13.1, V13.1.115`. If this errors, **you only have the driver, not the CUDA toolkit** — install it with `apt install -y cuda-toolkit-13-1` (or your version) before continuing.

```bash
ls /usr/local/cuda
```
Expected output: a list of folders (`bin`, `include`, `lib64`, etc). This is the CUDA install directory.

**If all three commands succeed**, you're ready. Move on.

---

## Step 1 — Identify your GPU's compute capability

CUDA needs to know which GPU architecture to compile for. Find yours below:

| GPU family | Example GPUs | Compute capability | Flag value |
|---|---|---|---|
| Turing | RTX 2060/2070/2080, T4 | 7.5 | `75` |
| Ampere | RTX 3060/3070/3080/3090, A100 | 8.6 | `86` |
| Ada Lovelace | RTX 4060/4070/4080/4090 | 8.9 | `89` |
| **Blackwell (consumer)** | **RTX 5070/5080/5090** | **12.0** | **`120`** |
| Hopper (datacenter) | H100 | 9.0 | `90` |

If your GPU isn't listed, check the table at https://developer.nvidia.com/cuda-gpus.

You'll plug this number into `-DCMAKE_CUDA_ARCHITECTURES=<value>` in Step 4. **Use the right value for your GPU** — using a wrong/lower value still compiles but runs slower; using a value higher than your GPU supports produces binaries that fail to load.

If you want broader compatibility across multiple GPU generations, list multiple values: `-DCMAKE_CUDA_ARCHITECTURES="89;120"` produces binaries that run natively on both Ada and Blackwell.

---

## Step 2 — Remove any existing apt COLMAP, then install all build dependencies

If you previously installed COLMAP via apt (CPU-only), uninstall it first so the new build doesn't conflict:

```bash
apt remove -y colmap
```

Now install **every** build dependency in one shot. We learned the hard way that COLMAP's cmake fails one error at a time — installing them all upfront avoids that frustrating loop:

```bash
apt update && apt install -y \
  git cmake build-essential \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libboost-filesystem-dev libboost-test-dev \
  libeigen3-dev libfreeimage-dev libmetis-dev \
  libgoogle-glog-dev libgflags-dev libsqlite3-dev \
  libglew-dev qtbase5-dev libqt5opengl5-dev libqt5svg5-dev libcgal-dev libceres-dev \
  libopenimageio-dev openimageio-tools libopencv-dev
```

**What each package does** (useful if one fails to install on your distro):

| Package | Used for |
|---|---|
| `git`, `cmake`, `build-essential` | Standard build tools (gcc, make, etc.) |
| `libboost-*-dev` | Boost C++ libraries (CLI args, graph algorithms, filesystem) |
| `libeigen3-dev` | Linear algebra (camera math, point math) |
| `libfreeimage-dev` | Legacy image I/O (kept for backward compat) |
| `libmetis-dev` | Graph partitioning for SfM scene clustering |
| `libgoogle-glog-dev`, `libgflags-dev` | Logging + command-line flags (Google libs) |
| `libsqlite3-dev` | SQLite — COLMAP's feature database backend |
| `libglew-dev`, `qtbase5-dev`, `libqt5opengl5-dev`, `libqt5svg5-dev` | OpenGL + Qt5 — required for the GUI viewer |
| `libcgal-dev` | Computational geometry (Delaunay triangulation for MVS) |
| `libceres-dev` | Nonlinear least squares — the core of bundle adjustment |
| `libopenimageio-dev`, `openimageio-tools` | Modern image I/O — replaced FreeImage in recent COLMAP |
| `libopencv-dev` | OpenCV headers — referenced by OpenImageIO's CMake config |

If `apt` complains that a package "has no installation candidate", your Ubuntu version may name it differently (e.g. `libqt5-svg5-dev` vs `libqt5svg5-dev`). Run `apt search <prefix>` to find the actual package name.

---

## Step 3 — Clone the COLMAP source

```bash
cd /tmp
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
```

This downloads ~80 MB of source code. Use `/tmp` (or any scratch directory) — after `make install` the source can be deleted.

If you want a specific release rather than the dev tip, add `--branch <tag>` to the clone, e.g. `--branch 3.10`.

---

## Step 4 — Configure the build with CMake

This is where you plug in your GPU's compute capability from Step 1. For RTX 50-series (Blackwell, sm_120):

```bash
cmake .. -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=120
```

Replace `120` with your value (e.g. `89` for RTX 4090, `86` for RTX 3090, `"89;120"` for both).

**Expected output:** dozens of `-- Found <something>` lines, ending in:

```
-- Configuring done
-- Generating done
-- Build files have been written to: /tmp/colmap/build
```

**You'll likely see this warning if you have a Blackwell GPU — it is OK:**
```
CMake Warning at src/colmap/mvs/CMakeLists.txt:205 (message):
  Blackwell GPU (sm_100+) detected. Replacing sm_100+ architectures with
  90-virtual (PTX) for colmap_mvs_cuda to work around an NVCC compiler bug.
```
This is COLMAP's workaround for a known nvcc bug with Blackwell. The MVS kernels will be JIT-compiled from PTX on first run instead of being precompiled native. Adds a few seconds to the first MVS run only. **Doesn't affect SfM (feature extraction + matching), which is what 3DGS pipelines actually use.**

**If cmake fails** with a "Could not find a package configuration file provided by ..." error, see **Troubleshooting** at the bottom of this guide. Each missing package has a one-liner fix.

---

## Step 5 — Build and install

This is the slow step — typically 15–20 minutes on 8 CPU cores:

```bash
make -j8 && make install && ldconfig
```

- `make -j8` runs 8 parallel jobs. Use `-j$(nproc)` to use all cores, or lower (`-j4`) if you have limited RAM.
- `make install` copies the binary + libraries to `/usr/local/bin/colmap` and `/usr/local/lib/`.
- `ldconfig` refreshes the system's library cache so the dynamic linker finds the new shared libraries.

Lots of warnings during compilation are normal (mostly `unused parameter`, `comparison of integers of different signs`). As long as the final commands are `Built target ...` lines and there's no `error:` — you're good.

If the build fails partway through, you can resume with `make -j8` again (it picks up where it left off). If `make install` fails with a permission error, prefix with `sudo`.

---

## Step 6 — Verify the install

```bash
colmap -h | head -5
```

Expected output (the **with CUDA** at the end is the key thing to look for):
```
COLMAP 4.1.0.dev0 (Commit <hash> on <date> with CUDA)
Usage:
  colmap [command] [options]
Documentation:
  https://colmap.github.io/
```

```bash
which colmap
```
Expected: `/usr/local/bin/colmap`

```bash
colmap feature_extractor -h 2>&1 | grep -i gpu
```
Expected:
```
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.gpu_index arg (=-1)
```

If all three checks pass, you have a working CUDA-enabled COLMAP.

**Quick functional test** (runs feature extraction on a tiny set of images):
```bash
mkdir -p /tmp/colmap_smoke/input && cp <some_image.jpg> /tmp/colmap_smoke/input/
colmap feature_extractor \
  --database_path /tmp/colmap_smoke/db \
  --image_path /tmp/colmap_smoke/input \
  --FeatureExtraction.use_gpu 1
```
You should see `Creating SIFT GPU feature extractor` in the output. If you instead see `SIFT CPU feature extractor`, GPU SIFT was silently disabled — go back and re-check the build.

---

## Troubleshooting

Errors below all happened during our build — each fix is a single `apt install -y <package>` followed by re-running `cmake ..`.

### `Could not find a package configuration file provided by "OpenImageIO"`
Missing OpenImageIO dev headers:
```bash
apt install -y libopenimageio-dev
```

### `The imported target "OpenImageIO::iconvert" references the file "/usr/bin/iconvert" but this file does not exist`
The OpenImageIO dev package was installed but the CLI tools (`iconvert`, etc.) weren't:
```bash
apt install -y openimageio-tools
```

### `Could not find a package configuration file provided by "Qt5Svg"`
Missing Qt5 SVG module:
```bash
apt install -y libqt5svg5-dev
```

### `Imported target "OpenImageIO::OpenImageIO" includes non-existent path "/usr/include/opencv4"`
OpenImageIO's CMake config references OpenCV headers — install them:
```bash
apt install -y libopencv-dev
```

### `Could not find a package configuration file provided by "Ceres"` (or "Eigen3", "Glog", "gflags")
You're missing one of the math libraries:
```bash
apt install -y libceres-dev libeigen3-dev libgoogle-glog-dev libgflags-dev
```

### `Could not find CUDA` / `CMAKE_CUDA_COMPILER could be found`
CUDA toolkit isn't installed or not on PATH. Re-do the prerequisite checks. Ubuntu install:
```bash
apt install -y cuda-toolkit-13-1   # or whichever version matches your driver
```

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`
Your CUDA version is too old for the architecture you set. Blackwell (sm_120) needs CUDA ≥ 12.8. Either upgrade CUDA or use a lower architecture (`-DCMAKE_CUDA_ARCHITECTURES=89` for Ada). The compiled binary will still run on your GPU just without using Blackwell-specific instructions.

### `error: 'std::filesystem' has not been declared` / similar C++17 errors
Your gcc is too old (need ≥ 8). On Ubuntu:
```bash
apt install -y g++-13
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

### `make: *** [...] Error 137`
Build was killed by the OOM killer — `-j8` used too much RAM. Retry with fewer jobs:
```bash
make -j2
```

---

## Cleaning up

After a successful install, the build artifacts in `/tmp/colmap/` can be deleted (the installed binary in `/usr/local/bin/colmap` stays):
```bash
rm -rf /tmp/colmap
```

To completely uninstall COLMAP later:
```bash
rm -f /usr/local/bin/colmap
rm -f /usr/local/lib/libcolmap*
rm -rf /usr/local/include/colmap
ldconfig
```

---

## Compute-capability quick reference

If you need to look this up again (e.g. building on a different machine):

| Generation | Cards | sm |
|---|---|---|
| Maxwell | GTX 9xx, M-series Tesla | 50/52/53 |
| Pascal | GTX 10xx, P100 | 60/61 |
| Volta | V100 | 70 |
| Turing | RTX 20xx, T4 | 75 |
| Ampere | RTX 30xx, A100, A40, A6000 | 80/86/87 |
| Ada Lovelace | RTX 40xx, L4, L40 | 89 |
| Hopper | H100, H200 | 90 |
| **Blackwell (datacenter)** | B100, B200, GB200 | 100 |
| **Blackwell (consumer)** | RTX 50xx | **120** |

Use the **same number without the dot** in `-DCMAKE_CUDA_ARCHITECTURES`. Multiple values are separated by `;` and listed in quotes: `-DCMAKE_CUDA_ARCHITECTURES="86;89;120"`.
