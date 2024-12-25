FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and git
RUN apt-get update && \
    apt-get install -y \
    git \
    python3.12-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/horus-ai-labs/DistillFlow.git .

# Install Python dependencies
RUN pip install --no-cache-dir poetry flash-attn

# Configure poetry to not create virtual environment in container
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry lock --no-update && \
    poetry install

# Command to run the script
CMD ["poetry", "run", "python", "src/anthropic_sft.py"]
# Update package lists
# RUN apt update
# # Install dependencies for DistillFlow
# # RUN apt install -y git build-essential
#
# # Copy the DistillFlow source code from your local machine
# WORKDIR /app
# COPY src/ DistillFlow/
#
# # Install Poetry
# RUN pip install poetry
#
# # Lock dependencies to avoid unexpected changes
# RUN poetry config virtualenvs.prefer-active true
#
# # Install DistillFlow dependencies within a virtual environment
# WORKDIR /app/DistillFlow
# RUN poetry install
#
# RUN pip install flash-attn
# # Execute the DistillFlow script
# CMD ["poetry", "run", "python", "src/anthropic_sft.py"]
#
# ####
# RUN apt update
# # Install the application dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
#
# # Copy in the source code
# COPY src ./src
# EXPOSE 5000
#
# # Setup an app user so the container doesn't run as the root user
# RUN useradd app
# USER app
#
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]