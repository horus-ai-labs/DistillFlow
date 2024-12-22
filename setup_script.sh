#!/bin/bash

# Update package list
apt update

# Install Python development tools
apt install python3.12-dev

# Install Poetry
pip install poetry

# Clone the repository
git clone https://github.com/horus-ai-labs/DistillFlow.git

# Navigate to the project directory
cd DistillFlow

# Activate Poetry shell
#poetry shell

# Install dependencies
poetry install