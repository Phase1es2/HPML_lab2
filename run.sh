#!/bin/bash

echo "Installing uv into this Python environment..."
python3 -m pip install --user uv

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"
echo "Using PATH = $PATH"

echo "Running lab2..."
exec uv run lab2.py "$@"