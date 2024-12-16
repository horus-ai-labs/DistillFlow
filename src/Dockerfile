#!/bin/bash
set -e

# Clone the repository
git clone $GITHUB_REPO app
cd app

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH=$HOME/.local/bin:$PATH

# Install dependencies and run the entry point
poetry install
poetry run python $ENTRY_POINT
