# Installation

Phone-Wheat3DGS runs in a single Docker image. It ships the compiled CUDA extensions and
a CUDA-enabled COLMAP build. All four stages run inside it, with no environment to switch
between them.

---

## Prerequisites

- Recommended: modern NVIDIA GPU with at least 24 GB VRAM.
- A recent NVIDIA driver. Everything else CUDA-related is inside the image.
- Docker with GPU support. Linux needs the NVIDIA Container Toolkit, Windows needs
  Docker Desktop with WSL2 integration.

On Windows, do this first:

- `wsl --install` in an administrator PowerShell, then reboot.
- Install the NVIDIA driver on Windows, not inside Ubuntu.
- Turn on WSL2 integration in the Docker Desktop settings.

Make sure Docker sees your GPU before building.

---

## Build the image

Clone the repository and build from its root:

```bash
git clone https://github.com/Patrick-Eugster/Bachelor_Thesis.git
cd Bachelor_Thesis
docker build -t phone-wheat3dgs .
```

On Windows clone into your WSL home folder, since I/O on the Windows disk is much slower.

The build takes a while, mostly COLMAP and the gsplat CUDA kernels.

The CUDA code is compiled for `CUDA_ARCHITECTURES=120`, which is sm_120, the RTX
50-series card this was developed and tested on.

## Run the image

Mount the repository into the container so your code, data and results stay on the host:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it --rm \
    -v "$(pwd)":/workspace phone-wheat3dgs
```

You land in a shell at `/workspace` with everything installed.

Both flags matter. `--ipc=host` lifts Docker's 64 MB shared memory limit, `--ulimit
memlock=-1` the page-locked memory cap. Without them PyTorch I/O breaks or crawls.

## Verify the install

Inside the container:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
colmap -h | head -1
python -c "import diff_gaussian_rasterization, flashsplat_rasterization, simple_knn; print('rasterizers ok')"
python -c "import gsplat, sahi, segment_anything, ultralytics; print('stage packages ok')"
python -c "from gsplat.cuda._backend import _C; print('gsplat kernels ok')"
```

The first line should report CUDA as available, and the COLMAP line must end in
`with CUDA`. Without that, Structure-from-Motion falls back to the slow CPU path. The
gsplat line should return instantly, since its CUDA kernels are compiled into the image.
A long compile there means the build failed to bake them in.

---

## Model weights

The detector and segmentation weights are not part of the repository. Download the ones
you need into `src/mask_generation/weights/`.

| File | Needed for | Where to get it |
|---|---|---|
| `wheat_head_detection_model.pt` | `yolo_sam_v1` and `sahi_yolo_sam` | The Global Wheat Challenge 2021 winning YOLOv5 model, from [ksnxr/GWC_solution](https://github.com/ksnxr/GWC_solution). Their README links the trained weights under *Use Our Trained Model*. Download that `best.pt` and rename it. |
| `best_yolo11l_40ep.pt` | `yolo11_sam` | [olzumst/distilled_transformer_spike_volume](https://huggingface.co/spaces/olzumst/distilled_transformer_spike_volume/tree/main) on Hugging Face |
| `yolo-medium-segment.pt` | the direct YOLO11-seg segmenter | [olzumst/distilled_transformer_spike_volume](https://huggingface.co/spaces/olzumst/distilled_transformer_spike_volume/tree/main) on Hugging Face |
| `sam_vit_h_4b8939.pth` | SAM1 | [dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `sam2.1_l.pt` | SAM2 | downloaded automatically by Ultralytics on first use |
| `sam3.pt` | SAM3 | access has to be requested on Hugging Face at `facebook/sam3`, then place the file here yourself |

---

## If the ALIKED front end fails to start

ALIKED runs through ONNX and can abort with a missing `libcublasLt.so.12` when the system
CUDA does not match. Create the CUDA-12 lib folder that `aliked_cuda12_libdir` in
`configs/preprocessing/colmap.yaml` already points at:

```bash
pip install --target /workspace/tools/cuda12libs \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
```

---

## What to do next

The [README](README.md) walks through the four stages and how to run them. Your own
images go under `input_plots/` and every result is written to `results/`.

---

## Manual conda setup for Euler (rough guide)

On the Euler cluster we run the pipeline with conda instead of Docker, so it splits the
pipeline across three conda environments: 3DGS reconstruction and 3D segmentation, mask
generation, and the COLMAP build. Use Docker unless you have the same constraint.

`environment.yml` describes the environment for the 3DGS reconstruction and the 3D
segmentation:

```bash
conda env create -f environment.yml
conda activate wheat3dgs
pip install -e .
pip install gsplat            # the default renderer
pip install viser nerfview    # only for src/viewer/
pip install ./src/gaussians/submodules/diff-gaussian-rasterization
pip install ./src/gaussians/submodules/flashsplat-rasterization
pip install ./src/gaussians/submodules/simple-knn
```

Mask generation needs a newer PyTorch than that environment has, so it gets its own
environment with the `maskgen` extra from `pyproject.toml`:

```bash
conda create -n wheat-maskgen python=3.10
conda activate wheat-maskgen
# install torch >= 2.4 for your CUDA version first, then:
pip install -e ".[maskgen]"
```

Do not install the `maskgen` extra into the first environment. It requires a newer
PyTorch, and upgrading PyTorch there breaks every compiled CUDA extension.

COLMAP still has to be built from source with CUDA for this route, which is what the third
environment is for. The build options that matter are the same ones the `Dockerfile` uses,
so read that file for the exact CMake flags.
