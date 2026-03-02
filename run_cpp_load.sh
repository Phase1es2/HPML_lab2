#!/usr/bin/env bash
set -e  # stop on any error

echo "Installing C++ Distributions of PyTorch"

# --- Always work relative to this script's location (repo root) ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# libtorch will be placed one level above repo root (same as your original cd .. behavior)
PARENT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

# --- Download/prepare libtorch ---
if [ ! -d "$PARENT_DIR/libtorch" ]; then
  mkdir -p "$PARENT_DIR/libtorch_downloads"
  cd "$PARENT_DIR/libtorch_downloads"

  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-*.zip

  mv libtorch "$PARENT_DIR/libtorch"

  cd "$PARENT_DIR"
  rm -rf "$PARENT_DIR/libtorch_downloads"
  echo "Done Downloading libtorch"
else
  echo "libtorch already exists, skip downloading."
fi

# make sure cmake in PATH
export PATH="$HOME/.local/bin:$PATH"

# ensure cmake installed (user)
python3 -m pip install --user cmake
cmake --version

echo "Start building C++ app..."

# --- Build in repo root ---
cd "$ROOT_DIR"
rm -rf build
mkdir build
cd build

cmake ..
make -j

echo "Build done. Running inference with TorchScript model..."

./resnet_18 ../best_model_jit.pt