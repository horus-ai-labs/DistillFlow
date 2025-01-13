#!/bin/bash
set -e

# Pull latest changes
git pull

# Re-install dependencies in case poetry.lock changed
poetry install

# Execute the command passed to docker run (or default command)
exec "$@"