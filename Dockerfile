# Optimized Dockerfile for RunPod Whisper Audio Transcription
# Base: CUDA-enabled PyTorch image for fast GPU processing

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-root --no-interaction --no-ansi

# Pre-download Whisper turbo model to avoid runtime delays
# Turbo model provides 8x faster processing with near-identical quality
RUN python -c "import whisper; whisper.load_model('turbo')"

# Copy application code
COPY audio2text/ ./audio2text/
COPY runpod_batch_transcribe.py ./

# Create directories for input/output
RUN mkdir -p /workspace/input /workspace/output

# Set environment variables for optimal GPU performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Default command - can be overridden by RunPod
CMD ["python", "runpod_batch_transcribe.py"]