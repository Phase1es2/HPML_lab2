import argparse
import time
from typing import List, Dict

import torch
import multiprocessing

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from lab2_data import get_train_dataset
from model import ResNet18
from train import train_one_epoch

"""
Write the code as device-agnostic, use the ArgumentParser to be able read parameters from input,
such as the use of cuda, the data-path, the number of dataloader workers and the optimizer
"""
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam", "adagrad", "adadelta", "adamw", "rmsprop"], help="optimizer")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum (SGD only)") # for sgd only
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--data_path", type=str , help="data path")
    parser.add_argument("--num_workers", type=int, default=0, help="number of data loading workers")
    parser.add_argument("--best_worker", action="store_true", help="call find best work function")
    parser.add_argument("--no_batch", action="store_true", help="do not use a batch norm during training")
    parser.add_argument("--eval", action="store_true", help="evaluate the model")
    parser.add_argument("--converting", type=str, default="scripting", choices=["scripting", "tracing"], help="converting")
    parser.add_argument("--latency", action="store_true", help="evaluate the model latency")
    return parser


"""
C3.1 Report the total time spent for the Dataloader varying the number of the workers
starting from zero and increment the number of workers by 4 (0, 4, 8, 12, 16) util the I/O time
does not decrease anymore. 
"""
def find_best_work(use_cuda: bool) -> tuple[int, float]:
    max_workers: int = int(multiprocessing.cpu_count())
    data_set = get_train_dataset()
    device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")

    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = set_optimizer(model, "sgd", 0.1, 0.9, 5e-4, False)

    work_time: dict[int, float] = {}
    work_list: List[int] = []
    best_work: int = 0
    best_time: float = float("inf")

    for work in range(0, max_workers + 1, 4):
        work_list.append(work)
        train_loader = DataLoader(
            data_set, batch_size=128, shuffle=True,
            num_workers=work, pin_memory=True if use_cuda else False
        )
        # use warm up
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        loss, acc, d_time, t_time, r_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        work_time[work] = d_time
        if d_time < best_time:
            best_work, best_time = work, d_time
        print(f"num_work={work}, time={d_time}")
    """
    draw the results in a graph to illustrate the performance you are getting as you
    increase the number of workers. Report how many workers are needed for the best runtime
    performance.
    """
    print(f"best_work={best_work}, time={best_time}")
    plot_best_work_results(work_list, work_time)
    return best_work, best_time

def plot_best_work_results(workers_list: List[int], results: Dict[int, float]) -> None:
    times = [results[i] for i in workers_list]
    plt.figure(figsize=(10, 6))
    plt.plot(workers_list, times, marker='o', linestyle='-', color='royalblue', linewidth=2)

    for i, t in enumerate(times):
        plt.text(workers_list[i], times[i], f'{t:.4f}s', ha='center', va='bottom')

    plt.title('DataLoader Performance: Impact of num_workers', fontsize=14)
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Total Data Time per Epoch (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(workers_list)

    plt.savefig('dataloader_benchmark.png')
    print("\nGraph has been saved as 'dataloader_benchmark.png'")
    plt.show()

"""
This part handle optimizer based on user's input
"""
def set_optimizer(
        model: nn.Module,
        opt: str,
        lr: float=0.1,
        momentum: float=0.9,
        weight_decay: float=5e-4,
        nesterov: bool = False,
):
    if opt == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif opt == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif opt == 'adagrad':
        return optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt == "adadelta":
        return optim.Adadelta(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def benchmark_latency(model, device, input_size=(1, 3, 32, 32), num_runs=100):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # warm up
    for _ in range(10):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    #Latency
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency = ((end_time - start_time) / num_runs) * 1000 # in milliseconds

    return avg_latency
