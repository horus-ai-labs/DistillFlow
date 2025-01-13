FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies and git
RUN apt-get update && \
    apt-get install -y \
    git \
    python3.12-dev \
    tmux \
    vim \
    gcc \
    g++ \
    make \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone repository
# RUN git clone https://github.com/horus-ai-labs/DistillFlow.git .

# Install Python dependencies
RUN pip install --no-cache-dir poetry flash-attn

# Configure poetry to not create virtual environment in container
RUN poetry config virtualenvs.create false

# # Install project dependencies
# RUN poetry lock && \
#     poetry install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    git clone https://github.com/horus-ai-labs/DistillFlow.git . && \
    poetry lock && \
    poetry install

# Use an entrypoint script for startup logic
ENTRYPOINT ["/entrypoint.sh"]
