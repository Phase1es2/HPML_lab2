#!/bin/bash

# use python -m pip to guarantee correct environment
echo "Installing uv into this Python environment..."
python3 -m pip install uv

# Add user local bin to PATH so uv can be found
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "Using PATH = $PATH"

uv run lab2.py "$@"

echo "Running lab2..."
uv run lab2.py "$@"