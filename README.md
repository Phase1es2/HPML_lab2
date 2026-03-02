# HPML Lab 2 (PyTorch + TorchScript)

This repository contains **Part A: Training & Profiling** and **Part B: TorchScript Model Optimization**.  
I use **`uv`** to manage the Python environment for reproducibility, portability, and ease of use.

---

## Repository Quick Start

### 1) Install `uv`

You can install `uv` using `pip` or `brew`:

```bash
pip install uv
# or
brew install uv
```

### 2) Run scripts with `uv run`

After installation, you can run any Python file in this repo with:

```bash
uv run <python_file_path>
```

### 3) Run via helper scripts (checks/installs uv and runs with args)

- `run_lab2_a.sh`: Part A
- `run_lab2_b.sh`: Part B

---

## Command-line Arguments

Most CLI arguments are parsed/handled by `utils.py` (e.g., `get_parser()`).

### Common arguments (Part A + Part B)

```text
--lr             # learning rate
--opt            # optimizer; choices=["sgd","adam","adagrad","adadelta","adamw","rmsprop"]
--nesterov       # enable Nesterov for SGD
--epochs         # default=5, number of epochs
--batch_size     # default=128, batch size
--weight_decay   # default=5e-4, weight decay
--momentum       # default=0.9, momentum
--cuda           # enable CUDA if available
--data_path      # dataset path
--num_workers    # default=0, number of data loading workers
--best_worker    # call find_best_work function
--no_batch       # train ResNet18 without BatchNorm
```

### Part B only (lab2_torchscript.py / run_lab2_b.sh)

```text
--eval           # evaluate the model
--converting     # default="scripting", choices=["scripting","tracing"]
--latency        # evaluate model latency
```

---

## How to Run

### If you already have `uv`

```bash
uv run lab2.py --opt sgd --lr 0.1 --nesterov --batch_size 128 --momentum 0.9 --cuda --epochs 5 --num_workers 4
```

### If you do not have `uv` (use bash scripts)

```bash
./run_lab2_a.sh --opt sgd --lr 0.1 --nesterov --batch_size 128 --momentum 0.9 --cuda --epochs 5 --num_workers 4
```

> Note: the argument name is `--num_workers` (not `--num_work`).

---

# Part A: Training and Profiling

## C1: Training in PyTorch

- Argument parsing: `utils.py` (calls `get_parser()`)
- Training loop: `train.py`

---

## C2: Time Measurement

### C2.1 Data-loading time for each epoch  
### C2.2 Training time for each epoch  
### C2.3 Total running time for each epoch  

You can test it using:

```bash
# use uv
uv run lab2.py --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# since opt=sgd, lr=0.1, batch_size=128, momentum=0.9 are set as default
# you can also run:
uv run lab2.py --cuda
```

```bash
# use bash
./run_lab2_a.sh --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

---

## C3: I/O Optimization

```bash
# use uv
uv run lab2.py --best_worker

# use bash
./run_lab2_a.sh --best_worker
```

The output of `best_worker` is **4** in my case (it may differ depending on your machine/environment).

---

## C4: Training on GPUs vs CPUs

### On GPUs

```bash
# use uv
uv run lab2.py --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# use bash
./run_lab2_a.sh --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

### On CPUs

```bash
# use uv
uv run lab2.py --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9

# use bash
./run_lab2_a.sh --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9
```

---

## C5: Experimenting with Different Optimizers

### SGD

```bash
# use uv
uv run lab2.py --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# use bash
./run_lab2_a.sh --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

### SGD with Nesterov

```bash
# use uv
uv run lab2.py --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --nesterov --cuda

# use bash
./run_lab2_a.sh --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --nesterov --cuda
```

### Adam

```bash
# use uv
uv run lab2.py --num_workers 4 --opt adam --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# use bash
./run_lab2_a.sh --num_workers 4 --opt adam --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

---

## C6: Experimenting without Batch Norm

In `model.py`, there is a `ResidualBlockNoBN` and `ResNet18NoBN` implementation.

```bash
# use uv
uv run lab2.py --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --no_batch --cuda

# use bash
./run_lab2_a.sh --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --no_batch --cuda
```

---

# Part B: Model Optimization with TorchScript

## C7: Convert to TorchScript

```bash
# use uv
uv run lab2_torchscript.py --converting scripting --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# use bash
./run_lab2_b.sh --converting scripting --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

---

## C8: Print Model Graph

```bash
# use uv
uv run lab2_torchscript.py --converting scripting --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda

# use bash
./run_lab2_b.sh --converting scripting --num_workers 4 --opt sgd --lr 0.1 --batch_size 128 --momentum 0.9 --cuda
```

There is code:

```python
print(scripted.code)
```

which prints the model graph during the scripting process.

---

## C9: Evaluate the TorchScript Model

### Enable GPU

```bash
# use uv
uv run lab2_torchscript.py --eval --cuda

# use bash
./run_lab2_b.sh --eval --cuda
```

### CPU

```bash
# use uv
uv run lab2_torchscript.py --eval

# use bash
./run_lab2_b.sh --eval
```

---

## C10: Latency Comparison

### Enable GPU

```bash
# use uv
uv run lab2_torchscript.py --latency

# use bash
./run_lab2_b.sh --latency
```

### CPU

```bash
# use uv
uv run lab2_torchscript.py --latency

# use bash
./run_lab2_b.sh --latency
```

> Note: in the original content, both GPU/CPU sections use the same `--latency` command; kept as-is.

---

# Extra Credit

Stay in the directory where `run_cpp_load.sh` is located:

```bash
./run_cpp_load.sh
```
