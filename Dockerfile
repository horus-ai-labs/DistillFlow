FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and git
RUN apt-get update && \
    apt-get install -y \
    git \
    python3.12-dev \
    tmux \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/horus-ai-labs/DistillFlow.git .

RUN git pull

# Install Python dependencies
RUN pip install --no-cache-dir poetry flash-attn

# Configure poetry to not create virtual environment in container
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry lock --no-update && \
    poetry install
