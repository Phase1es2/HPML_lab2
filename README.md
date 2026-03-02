# HPML Lab 2 (PyTorch + TorchScript)

This repository contains:

- **Part A: Training & Profiling** (PyTorch training + timing/profiling experiments)
- **Part B: Model Optimization with TorchScript** (TorchScript conversion + evaluation + latency benchmark)
- **Extra Credit**: Load TorchScript model from C++ (`load_model.cpp`)

I use **`uv`** to manage the Python environment for reproducibility, portability, and ease of use.

---

## Project Structure

```text
.
├── CMakeLists.txt
├── README.md
├── lab2.py
├── lab2_data.py
├── lab2_torchscript.py
├── load_model.cpp
├── model.py
├── model_jit.py
├── pyproject.toml
├── run_cpp_load.sh
├── run_lab2_a.sh
├── run_lab2_b.sh
├── train.py
├── utils.py
└── uv.lock
```

---

## File/Module Overview

### `lab2.py` (Part A entrypoint / main)
**This is the main program for Part A.** It parses CLI arguments via `get_parser()` and then:

- Selects device: CPU vs GPU (`--cuda` + `torch.cuda.is_available()`)
- If `--best_worker` is set, runs `find_best_work(use_cuda)` to search the best dataloader worker count
- Otherwise:
  - Builds CIFAR-10 training dataset and dataloader
  - Builds model: `ResNet18` **or** `ResNet18NoBN` (when `--no_batch`)
  - Creates optimizer via `set_optimizer(...)` (based on `--opt`, `--lr`, `--momentum`, `--weight_decay`, `--nesterov`)
  - Runs epoch loop calling `train_one_epoch(...)`
  - Prints per-epoch metrics:
    - `avg_loss`, `avg_acc`
    - `total_data_time`, `total_train_time`, `total_running_time`

This is the core logic flow (simplified from your `main()`):

```python
def main():
    args = get_parser().parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    if args.best_worker:
        best_work, best_time = find_best_work(use_cuda)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")

        data_set = get_train_dataset()
        train_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if use_cuda else False,
        )

        model = ResNet18NoBN(num_classes=10).to(device) if args.no_batch else ResNet18(num_classes=10).to(device)

        summary(model, input_size=(3, 32, 32), device=str(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = set_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay, args.nesterov)

        print(f"\nBegin Training, Optimizer {optimizer}, Device {device}")
        print(f"{'Epoch':<6} | {'Loss':<8} | {'Acc (%)':<8} | {'Data Time':<10} | {'Train Time':<10} | {'Running Time':<10}")
        print("-" * 60)

        for epoch in range(1, args.epochs + 1):
            loss, acc, d_time, t_time, r_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"{epoch:<6} | " f"{loss:<10.4f} | " f"{acc:<10.2f} | " f"{d_time:<12.4f} | " f"{t_time:<12.4f} | " f"{r_time:<12.4f}")

if __name__ == "__main__":
    main()
```

---

### `lab2_torchscript.py` (Part B entrypoint / main)
**This is the main program for Part B.** It uses:

```python
MODEL_PATH = "best_model_jit.pt"
```

It supports three main modes:

1) **Evaluation (`--eval`)**  
Loads TorchScript model from `MODEL_PATH` and evaluates on CIFAR-10 test set, printing Top-1 accuracy.

2) **Latency benchmark (`--latency`)**  
Benchmarks latency for:
- Original PyTorch model (`ResNet18`)
- TorchScript model loaded from `MODEL_PATH`
Then prints latency and speedup.

3) **Training + TorchScript conversion (default)**  
Trains a TorchScript-compatible model (`ResNet18JIT` / `ResNet18NoBNJIT`) and **whenever a new best accuracy is reached**, converts and saves the model to `MODEL_PATH` using either:
- `--converting scripting`  (default)
- `--converting tracing`

It also prints the TorchScript graph (once) during conversion:
- scripting: `print(scripted.code)`
- tracing: `print(traced.graph)` / `print(traced.inlined_graph)`

---

### `lab2_data.py`
Provides dataset helpers for CIFAR-10:
- `get_train_dataset()` for CIFAR-10 training set
- `get_test_dataset()` for CIFAR-10 test set

---

### `model.py`
Defines PyTorch ResNet models for Part A:
- `ResNet18`
- `ResNet18NoBN` (ResNet18 without BatchNorm)

---

### `model_jit.py`
Defines TorchScript-friendly ResNet models for Part B:
- `ResNet18JIT`
- `ResNet18NoBNJIT`

(These are typically small modifications to make the architecture compatible with `torch.jit.script` / `torch.jit.trace`.)

---

### `train.py`
Implements training utilities such as:
- `train_one_epoch(...)`  
Runs one epoch and reports metrics used by `lab2.py` and `lab2_torchscript.py`:
- `avg_loss`, `avg_acc`
- `total_data_time`, `total_train_time`, `total_running_time`

---

### `utils.py`
Helper functions, including:

- `get_parser()`  
Defines and returns the CLI argument parser.

- `set_optimizer(model, opt, lr, momentum, weight_decay, nesterov)`  
Creates optimizer based on user input (`--opt`).

- `find_best_work(use_cuda: bool)`  
Searches for the best `num_workers` setting for data loading in the current environment.

- `plot_best_work_results(...)`  
Plots the `best_worker` results.

- `benchmark_latency(model, device)`  
Measures inference latency (used by Part B `--latency`).

---

### `load_model.cpp`, `CMakeLists.txt`, `run_cpp_load.sh` (Extra Credit)
- `load_model.cpp`: C++ loader for TorchScript model
- `CMakeLists.txt`: build config
- `run_cpp_load.sh`: build & run helper script

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
