# ============================================================================
# SparkStack — one Dockerfile, four targets. Compose picks one per service:
#
#   target          service(s)                        profile
#   ──────────────  ────────────────────────────────  ──────────
#   llama-runtime   d1-decide, embeddings             utility
#   tools-runtime   tools (one-shot downloads/bench)  tools
#   sd-runtime      image-flux3                       vision
#   vllm-runtime    llm-nemotron                      llm
#
# Compiled for GB10 (sm_121) / CUDA 13.
# Update everything: ./sparkstack.sh update  (pulls fresh bases, rebuilds)
# ============================================================================

# CUDA_ARCH: GB10 is Blackwell sm_121 — use "121a-real" (real-arch SASS, no PTX).
ARG CUDA_ARCH=121a-real
ARG LLAMA_REF=master   # any commit >= llama.cpp PR #29818 (provides /v1/systemone)
ARG SD_REF=master      # any commit >= 2026-10-16 (provides FLUX.3 support)


# ── llama.cpp ───────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS build-base
# The libcuda.so.1 stub symlink lets the linker resolve the driver API without
# a GPU at build time; the nvidia toolkit mounts the real driver at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

FROM build-base AS llama-builder
ARG CUDA_ARCH LLAMA_REF
ADD https://github.com/ggml-org/llama.cpp/archive/${LLAMA_REF}.tar.gz /tmp/llama.tar.gz
RUN mkdir -p /src && tar -xzf /tmp/llama.tar.gz -C /src --strip-components=1 \
 && cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGGML_NATIVE=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build /src/build -j --target llama-server

# CUDA runtime base (not plain ubuntu): libggml-cuda needs libcudart/libcublas/
# libnccl, which only the CUDA runtime images ship.
FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS llama-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=llama-builder /src/build/bin/ /opt/llama/bin/
ENV PATH=/opt/llama/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama/bin
EXPOSE 8080
# no ENTRYPOINT: every compose command names the binary (llama-server …)

FROM llama-runtime AS tools-runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --break-system-packages --no-cache-dir 'huggingface_hub[cli]'
ENTRYPOINT []


# ── stable-diffusion.cpp ────────────────────────────────────────────────────
# git clone (not tarball): SD needs its submodules, which tarballs omit.

FROM build-base AS sd-builder
ARG CUDA_ARCH SD_REF
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp /src \
 && git -C /src checkout "$SD_REF" \
 && git -C /src submodule update --init --recursive \
 && cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSD_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DSD_USE_UPSTREAM_GGML=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build /src/build -j --target sd-server sd-cli

FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS sd-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=sd-builder /src/build/bin/ /opt/sd/bin/
ENV PATH=/opt/sd/bin:$PATH \
    LD_LIBRARY_PATH=/opt/sd/bin
EXPOSE 8080
# no ENTRYPOINT: the compose command names the binary (sd-server …)


# ── vLLM ─────────────────────────────────────────────────────────────────────
# Recipe: huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
# Rolling tag by policy: `update` pulls the current build; if a nightly breaks
# the recipe, pin the tag by digest until the next good one.

# Image digest pinned to the NVIDIA recipe (the rolling cu134-nightly segfaults
# in b12x MoE kernels on first inference). Update the pin when NVIDIA updates
# the recipe: huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
FROM vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989 AS vllm-runtime
# flashinfer invoked directly because vLLM's CLI parser does NVML device
# inference at startup and dies in a GPU-less build. /app must be created.
RUN mkdir -p /app \
 && curl -fsSL -o /app/ultra_v3_reasoning_parser.py \
      https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/resolve/main/ultra_v3_reasoning_parser.py \
 && pip install --no-cache-dir b12x \
 && python3 -m flashinfer download-kernels
ENV VLLM_USE_V2_MODEL_RUNNER=1
