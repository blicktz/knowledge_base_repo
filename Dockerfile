# Optimized Dockerfile for RunPod FastAPI Whisper Transcription
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

# Install Python dependencies including FastAPI
RUN pip install --no-cache-dir \
    openai-whisper \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles

# Pre-download Whisper turbo model to avoid runtime delays
RUN python -c "import whisper; whisper.load_model('turbo')"

# Copy application code
COPY runpod_fastapi_server.py ./

# Create directories for uploads and outputs
RUN mkdir -p /workspace/uploads /workspace/outputs /workspace/jobs

# Set environment variables for optimal GPU performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set default API key (override with environment variable)
ENV TRANSCRIBE_API_KEY=mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I

# Expose HTTP port
EXPOSE 8080

# Start FastAPI server
CMD ["python", "runpod_fastapi_server.py"]