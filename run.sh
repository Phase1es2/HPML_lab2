#!/bin/bash

echo "Download the uv for the environment setting..."

pip install uv

echo "uv install complete"

uv run lab2.py "$@"