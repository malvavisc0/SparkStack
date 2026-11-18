# WhisperX 3.8.6 on NVIDIA DGX Spark (ARM64 + Blackwell SM_121)
# Based on the Mekopa/whisperx-blackwell recipe, updated for WhisperX 3.8.6,
# pyannote.audio 4 and faster-whisper 1.2.

FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_FORCE_PTX_JIT=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libopenblas-dev \
    libsndfile1 \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Build torchaudio against NVIDIA's torch.
RUN git clone --depth 1 --branch v2.6.0 --recursive https://github.com/pytorch/audio.git /tmp/torchaudio && \
    cd /tmp/torchaudio && \
    pip install --no-cache-dir --no-deps --no-build-isolation . && \
    rm -rf /tmp/torchaudio

RUN pip install --no-cache-dir pybind11

# Build CTranslate2 with CUDA and cuDNN.
RUN git clone --recursive --depth 1 --branch v4.7.1 https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 && \
    cd /tmp/ctranslate2 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DCUDNN_ROOT=/usr \
        -DCUDA_DYNAMIC_LOADING=ON \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON \
        -DOPENMP_RUNTIME=COMP \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp/ctranslate2/python && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/ctranslate2

# Preserve NVIDIA's torch, numpy and custom torchaudio/torio before pip may
# replace them with the generic PyTorch 2.8 wheels required by WhisperX.
RUN cp -r /usr/local/lib/python3.12/dist-packages/torch /tmp/torch_nvidia && \
    cp -r /usr/local/lib/python3.12/dist-packages/torchvision /tmp/torchvision_nvidia && \
    cp -r /usr/local/lib/python3.12/dist-packages/torchaudio /tmp/torchaudio_custom && \
    cp -r /usr/local/lib/python3.12/dist-packages/torio /tmp/torio_custom && \
    cp -r /usr/local/lib/python3.12/dist-packages/numpy /tmp/numpy_nvidia && \
    cp -r /usr/local/lib/python3.12/dist-packages/numpy.libs /tmp/numpy_libs_nvidia 2>/dev/null || true

# Two independent DNS hazards in this build step:
#  1. The NGC base image's pip.conf adds extra-index-url pypi.ngc.nvidia.com,
#     an NVIDIA-internal host that is NXDOMAIN outside their network. pip
#     queries it for every project, gets "Name or service not known", and
#     retries each one — with --retries 20 that is ~20 min of silent backoff
#     per project. PIP_CONFIG_FILE=/dev/null ignores that config; every
#     package below is on public PyPI.
#  2. Parallel DNS *over UDP* through the Docker bridge drops queries on this
#     host (166/200 fail in a burst; single and host-network lookups are fine).
#     RES_OPTIONS=use-vc makes glibc use TCP for DNS, which is 100% reliable.
# Keep retries small so a real outage fails fast instead of backing off.
RUN PIP_CONFIG_FILE=/dev/null RES_OPTIONS=use-vc \
    pip install --no-cache-dir --index-url https://pypi.org/simple \
        --retries 5 --timeout 30 \
    "fastapi==0.136.0" \
    "faster-whisper==1.2.1" \
    "matplotlib==3.10.8" \
    "nltk==3.9.4" \
    "pandas==2.2.3" \
    "pyannote.audio==4.0.4" \
    "python-multipart==0.0.26" \
    # pyannote-metrics 4.x (pulled in by pyannote.audio 4.0.4) requires
    # scipy>=1.15.1; 1.13.1 makes the resolver fail before it downloads
    # anything. 1.15.3 still supports the NVIDIA numpy 1.26.4 restored below.
    "scipy==1.15.3" \
    "semver==3.0.4" \
    "torchcodec==0.11.1" \
    "transformers<5.0.0" \
    "uvicorn[standard]==0.44.0" \
    "whisperx==3.8.6"

# Restore NVIDIA's torch/torchvision/torchaudio/torio/numpy for GB10. The
# base numpy 1.26.4 is the ABI that the NVIDIA torch 2.6 build expects.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/torch && \
    rm -rf /usr/local/lib/python3.12/dist-packages/torchvision && \
    rm -rf /usr/local/lib/python3.12/dist-packages/torchaudio && \
    rm -rf /usr/local/lib/python3.12/dist-packages/torio && \
    rm -rf /usr/local/lib/python3.12/dist-packages/numpy && \
    rm -rf /usr/local/lib/python3.12/dist-packages/numpy.libs && \
    mv /tmp/torch_nvidia /usr/local/lib/python3.12/dist-packages/torch && \
    mv /tmp/torchvision_nvidia /usr/local/lib/python3.12/dist-packages/torchvision && \
    mv /tmp/torchaudio_custom /usr/local/lib/python3.12/dist-packages/torchaudio && \
    mv /tmp/torio_custom /usr/local/lib/python3.12/dist-packages/torio && \
    mv /tmp/numpy_nvidia /usr/local/lib/python3.12/dist-packages/numpy && \
    mv /tmp/numpy_libs_nvidia /usr/local/lib/python3.12/dist-packages/numpy.libs 2>/dev/null || true && \
    echo "Restored NVIDIA CUDA torch, numpy, and custom torchaudio/torio for Blackwell GPU support"

# Blackwell SM_121 compatibility patches. NVRTC does not recognize SM_121,
# but SM_90 code runs on Blackwell.
RUN sed -i 's/def get_device_capability/def _original_get_device_capability/g' \
    /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '# BLACKWELL PATCH: Spoof SM_121 as SM_90 for nvrtc compatibility' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo 'def get_device_capability(device=None):' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '    major, minor = _original_get_device_capability(device)' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '    if major == 12 and minor == 1:' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '        return (9, 0)  # Pretend to be Hopper H100' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py && \
    echo '    return (major, minor)' >> /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py

RUN sed -i 's/compute_121/compute_90/g' /usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py 2>/dev/null || true && \
    sed -i 's/sm_121/sm_90/g' /usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py 2>/dev/null || true

RUN sed -i 's/compute_121/compute_90/g' /usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py 2>/dev/null || true && \
    sed -i 's/sm_121/sm_90/g' /usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py 2>/dev/null || true

RUN sed -i 's/spectrum = torch.fft.rfft(strided_input).abs()/# BLACKWELL PATCH: Avoid jiterator by computing abs manually\n    fft_result = torch.fft.rfft(strided_input)\n    spectrum = torch.sqrt(fft_result.real**2 + fft_result.imag**2)/' \
    /usr/local/lib/python3.12/dist-packages/torchaudio/compliance/kaldi.py && \
    echo "=== TORCHAUDIO FBANK PATCH APPLIED ==="

# Persist PTX JIT results across restarts. The NGC base image sets
# CUDA_CACHE_DISABLE=1; without this, every restart recompiles all kernels
# from PTX (~5 min on the first request). The path lives inside the
# ./data/whisper-data volume mounted by compose.
ENV CUDA_CACHE_DISABLE=0 \
    CUDA_CACHE_PATH=/root/.cache/huggingface/nv-jit-cache \
    CUDA_CACHE_MAXSIZE=4294967296

COPY docker/whisperx/app/ /app/app/

EXPOSE 8003

# 300s start period: uvicorn serves nothing until the lifespan startup loads
# Whisper large-v3; a cold load (plus the first-boot JIT compile) takes minutes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "app.main_gpu:app", "--host", "0.0.0.0", "--port", "8003"]
