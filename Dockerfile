# phone-wheat3dgs - all 4 stages in one image
# build:  docker build -t phone-wheat3dgs .
# run:    docker run --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v "$(pwd)":/workspace phone-wheat3dgs
#         (--ipc=host: the default 64MB /dev/shm is too small for torch dataloader workers)
# prerequisites + model weights -> INSTALL.md
#
# everything below is taken from the working local container:
# colmap commit + cmake flags from its build cache, apt + pip lists from the install dates

FROM nvcr.io/nvidia/pytorch:26.01-py3

# 120 = sm_120 = RTX 50xx, what we built colmap for
# other card -> --build-arg CUDA_ARCHITECTURES=xx
ARG CUDA_ARCHITECTURES="120"

# pinned, not main. the revision the thesis results were made with
ARG COLMAP_COMMIT=58e626ab36b4085651b4809313d32073aa8c6c66

ENV DEBIAN_FRONTEND=noninteractive

# colmap build deps + ffmpeg for the 360 renders
# pdftoppm (poppler-utils) is used by markers/legacy/test_cct_phase0.py
# opencv-dev, openimageio-tools, pngquant: installed locally, colmap itself does not use them
# util-linux, mount, bsdutils: already in the base image, listed because we have them
RUN apt-get update && apt-get install -y \
        git \
        cmake \
        ninja-build \
        build-essential \
        ffmpeg \
        poppler-utils \
        pngquant \
        libopencv-dev \
        openimageio-tools \
        libopenimageio-dev \
        util-linux \
        mount \
        bsdutils \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-test-dev \
        libboost-thread-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        libcgal-dev \
        libceres-dev \
        libsuitesparse-dev \
        libcurl4-openssl-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libqt5svg5-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# colmap from source, the apt one is CPU only
# CUDA_ENABLED -> gpu sift
# ONNX_ENABLED -> aliked + lightglue, the phone sfm default (sift breaks on wheat)
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap \
    && cd /tmp/colmap \
    && git checkout ${COLMAP_COMMIT} \
    && cmake -B build -S . -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCUDA_ENABLED=ON \
        -DONNX_ENABLED=ON \
        -DGUI_ENABLED=ON \
        -DCGAL_ENABLED=ON \
        -DTESTS_ENABLED=OFF \
    && ninja -C build \
    && ninja -C build install \
    && rm -rf /tmp/colmap

# every pip package we installed on top of the base image, at the version we have.
# torch, torchvision and numpy are NOT here - they come from the base image and the
# rasterizers below are compiled against exactly those.
# if this ever fails on a version conflict, add --no-deps here (the list is complete)
RUN pip install --no-cache-dir \
        GitPython==3.1.46 \
        ImageIO==2.37.3 \
        PyQt5-Qt5==5.15.18 \
        PyQt5==5.15.11 \
        PyQt5_sip==12.18.0 \
        PySocks==1.7.1 \
        antlr4-python3-runtime==4.9.3 \
        click==8.2.1 \
        colorlog==6.10.1 \
        comet_ml==3.57.2 \
        configobj==5.0.9 \
        dulwich==0.25.2 \
        embreex==2.17.7.post7 \
        et_xmlfile==2.0.0 \
        everett==3.1.0 \
        ffmpeg-python==0.2.0 \
        fire==0.7.1 \
        ftfy==6.3.1 \
        future==1.0.0 \
        gdown==5.2.1 \
        gitdb==4.0.12 \
        gsplat==1.5.3 \
        hf-xet==1.2.0 \
        huggingface_hub==1.3.1 \
        hydra-core==1.3.2 \
        jaxtyping==0.3.9 \
        lxml==6.0.3 \
        manifold3d==3.4.1 \
        mapbox_earcut==2.0.0 \
        msgspec==0.21.0 \
        nerfview==0.1.3 \
        omegaconf==2.3.0 \
        opencv-python==4.13.0.92 \
        openpyxl==3.1.5 \
        pillow==12.1.1 \
        plyfile==1.1.3 \
        polars-runtime-32==1.38.1 \
        polars==1.38.1 \
        pycollada==0.9.3 \
        pycolmap==4.0.4 \
        python-box==6.1.0 \
        requests-toolbelt==1.0.0 \
        rtree==1.4.1 \
        sahi==0.12.1 \
        scikit-image==0.26.0 \
        seaborn==0.13.2 \
        semantic-version==2.10.0 \
        sentry-sdk==2.54.0 \
        shapely==2.1.2 \
        simplejson==3.20.2 \
        smmap==5.0.3 \
        splines==0.3.3 \
        svg.path==7.0 \
        termcolor==3.3.0 \
        thop==0.1.1.post2209072238 \
        tifffile==2026.5.15 \
        timm==1.0.27 \
        trimesh==4.11.5 \
        ultralytics-thop==2.0.18 \
        ultralytics==8.4.21 \
        vhacdx==0.0.10 \
        viser==1.0.26 \
        wadler_lindig==0.1.7 \
        wandb==0.25.1 \
        websockets==16.0 \
        wurlitzer==3.1.1 \
        yourdfpy==0.0.60 \
        zstandard==0.25.0

# these two came from git, not pypi. same commits we have.
# clip is the ultralytics fork, not openai's. needed for sam3 text prompts
RUN pip install --no-cache-dir \
        git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf \
        git+https://github.com/ultralytics/CLIP.git@488e81a6711eea7346872b46ea928b367da8889d

# the 3 cuda rasterizers. NOT editable on purpose, so the compiled .so sits in
# site-packages and survives the bind mount over /workspace.
# TORCH_CUDA_ARCH_LIST is not set here, the base image already has it
COPY src/gaussians/submodules /tmp/submodules
RUN pip install --no-cache-dir --no-build-isolation \
        /tmp/submodules/diff-gaussian-rasterization \
        /tmp/submodules/flashsplat-rasterization \
        /tmp/submodules/simple-knn \
    && rm -rf /tmp/submodules

# gsplat ships no compiled .so, it JIT-builds its kernels the first time it runs (10-20 min of
# cpu with the gpu idle). do it here so the cache is baked into the image, otherwise every
# fresh container pays it again
RUN python -c "from gsplat.cuda._backend import _C" && echo "gsplat kernels built"

# ultralytics writes its settings to ~/.config/Ultralytics, but only if that dir already
# exists (os.access says no on a missing path). without this it falls back to /tmp every run
RUN mkdir -p /root/.config/Ultralytics

# apt pulls in ubuntu's libucx0, whose libucs.so.0 is missing the ucs_config_doc_nop
# symbol that torch needs. nvidia ships its own working one in hpcx, so put that first
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}"

# --no-deps: the pins in pyproject.toml are for the conda route, they would
# downgrade numpy and torch here
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir --no-deps -e .

CMD ["/bin/bash"]
