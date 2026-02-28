#!/bin/bash

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if uv exists
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    python3 -m pip install --user uv
else
    echo "uv is already installed. Skipping installation."
fi

echo "Using PATH = $PATH"

echo "Running lab2..."
uv run lab2.py "$@"