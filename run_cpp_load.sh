#!/usr/bin/env bash
set -e  # 任何一步錯誤就停

echo "Installing C++ Distributions of PyTorch"

# HPML_folder
cd ..

#  libtorch
if [ ! -d "libtorch" ]; then
  mkdir -p libtorch_downloads
  cd libtorch_downloads

  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-*.zip

  mv libtorch ../libtorch

  cd ..
  rm -rf libtorch_downloads
  echo "Done Downloading libtorch"
else
  echo "libtorch already exists, skip downloading."
fi

# make sure cmake in PATH
export PATH="$HOME/.local/bin:$PATH"

# back_to HPML_lab2
cd HPML_lab2


python3 -m pip install --user cmake

cmake --version

echo "Start building C++ app..."

rm -rf build
mkdir build
cd build

cmake ..
make -j

echo "Build done. Running inference with TorchScript model..."

./my_torch_app ../best_model_jit.pt