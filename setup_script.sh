#!/bin/bash

# Update package list
apt update

# Install Python development tools
apt install python3.12-dev

# Install Poetry
pip install poetry

# Activate Poetry shell
#poetry shell

# Install dependencies
poetry install