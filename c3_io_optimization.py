import time
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

"""
The dataset we are using is [IFAR-10 Dataset which is labeled subsets of the 80 millions tiny images dataset. 
CIFAR-10 and CIFAR-100 were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. (Sadly, the 80 million tiny images dataset 
has been thrown into the memory hole by its authors. Spotting the doublethink which was used to justify its erasure is left as an exercise for the reader.)
The CIFAR-10 dataset consists of 60000 ($32 \times 32$) color, 
images in 10 classes, with 6000 images per class. and there are 50000 training images and 10000 test images.
"""


def get_train_dataset():
    """
    To pre-processing image data for better accuracy, we used data augmentation, here is the technique that we applied.
    - **Random Cropping** `RandomCrop()`: randomly crop out smaller section of the image, forcing the model to handle parts of objects or varied backgrounds.
    - **Horizontal/Vertical Flipping** `RandomHorizontalFlip()`/: flip an image horizontally with a certain probability. This is techniques the model to be invariant to flips.
    - **ToTensor** `ToTensor()`: convert image or numpy arrays into PyTorch tensors, which makes them usable by PyTorch models. Usually transforms the image from shape (Height, Width, Channel) with values in [0, 255] to shape (Channel, Height, Width) with values in [0, 1]
    - **Normalization** `Normalize(mean=[], std=[])`: Subtract the mean and divide by the standard deviation on each channel, usually based on the overall dataset 'statistics. This makes training more stable and faster, normlized inputs help the network learn more effectively.
    :return:
    """
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def measure_dataloader_time(
        dataset, num_works: int, batches: int = 20, device: torch.device = torch.device("cpu")
) -> Tuple[float, float]:
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_works,
        pin_memory=True if device == torch.device("cuda") else False,
    )

    _ = next(iter(dataloader))

    stat_time = time.perf_counter()
    for i, (images, labels) in enumerate(dataloader):
        if i >= batches:
            break
    end_time = time.perf_counter()
    total_time = end_time - stat_time
    avg_time = total_time / batches
    return total_time, avg_time

def find_best_num_work(
        dataset,
        max_num_workers: int=32,
        step: int=4,
        improvement_tol: float = 0.01
) -> Tuple[Dict[int, float], List[int]]:
    results: Dict[int, float] = {}
    print(f"{'num_workers':<12} | {'total_time(s)':<15} | {'avg_time_per_batch(s)':<22}")
    print("-" * 60)
    work_list = []
    best_time = float("inf")
    num_workers = 0

    while num_workers < max_num_workers:
        work_list.append(num_workers)
        total_time, avg_time = measure_dataloader_time(dataset, num_workers)
        results[num_workers] = total_time
        print(f"{num_workers:<12} | {total_time:<15.4f} | {avg_time:<22.4f}")

        if best_time == float("inf"):
            best_time = total_time
        else:
            improvement = (best_time - total_time) / best_time
            if improvement <= improvement_tol:
                break
            else:
                best_time = total_time
        num_workers += step
    return results, work_list

def plot_results(workers_list: List[int], results: Dict[int, float]) -> None:
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