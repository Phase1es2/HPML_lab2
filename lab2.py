import time
from typing import List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
import multiprocessing

from c3_io_optimization import get_train_dataset, plot_results
from prog import get_parser

# color images has rgb channels 3
# and it is 32 x 32
INPUT_SIZE = (3, 32, 32)

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            strides: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, padding=1, stride=strides, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=strides, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(
            self,
            x: Float[Tensor, "batch in_channels h w"]
    ) -> Float[Tensor, "batch out_channels h2 w2"]:
        identity = self.shortcut(x)
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out, inplace=True)


class ResNet18(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def _make_layer(
            self,
            out_channels: int,
            blocks: int,
            stride: int
    ) -> nn.Sequential:
        layers: list[ResidualBlock] = [ResidualBlock(self.in_channels, out_channels, strides=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(
            self,
            x: Float[Tensor, "batch in_channels h w"]
    ) -> Float[Tensor, "batch num_classes"]:
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = rearrange(x, 'b c 1 1 -> b c')
        x = self.fc(x)
        return x

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    total_data_time, total_train_time = 0.0, 0.0

    start_step_time = time.perf_counter()

    pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
    printed = False
    for inputs, labels in pbar:

        data_time = time.perf_counter() - start_step_time
        total_data_time += data_time

        inputs, labels = inputs.to(device), labels.to(device)

        train_start_time = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if not printed:
            total_grads = sum(
                p.grad.numel() for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            print(f"Gradient elements: {total_grads}")
            printed = True
        optimizer.step()
        """
        if device.type =='cuda':
            torch.cuda.synchronize()
        """
        if device.type == 'mps':
            torch.mps.synchronize()

        train_time = time.perf_counter() - train_start_time
        total_train_time += train_time

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        cur_loss = running_loss / total
        cur_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{cur_loss:.3f}',
            'Acc': f'{cur_acc:.1f}%',
            'Data_t': f'{data_time:.3f}s'
        })

        start_step_time = time.perf_counter()

    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc, total_data_time, total_train_time


def set_optimizer(
        model: ResNet18,
        opt: str,
        lr: float=0.1,
        momentum: float=0.9,
        weight_decay: float=5e-4
) -> optim.Optimizer:
    if opt == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif opt == "adam":
        return optim.Adam(
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
    elif opt == "adamw":
        return optim.AdamW(
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
    elif opt == "adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

"""
C3.1 Report the total time spent for the Dataloader varying the number of the workers
starting from zero and increment the number of workers by 4 (0, 4, 8, 12, 16) util the I/O time
does not decrease anymore. 
"""
def find_best_work(use_cuda):
    max_workers: int = int(multiprocessing.cpu_count())
    data_set = get_train_dataset()
    device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")
    # print(device)
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

    work_time: dict[int, float] = {}
    work_list: List[int] = []
    best_work, best_time = 0, float('inf')
    for work in range(0, max_workers + 1, 4):
        work_list.append(work)
        train_loader = DataLoader(
            data_set, batch_size=128, shuffle=True,
            num_workers=work, pin_memory=True if use_cuda else False
        )
        loss, acc, d_time, t_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
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
    plot_results(work_list, work_time)
    return best_work, best_time

def main():
    args = get_parser().parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    print(args.opt)
    num_workers = args.num_workers
    epochs = args.epochs
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    data_set = get_train_dataset()
    train_loader = DataLoader(
        data_set, batch_size=128, shuffle=True,
        num_workers=8, pin_memory=True if use_cuda else False
    )
    model = ResNet18(num_classes=10).to(device)

    summary(model, input_size=INPUT_SIZE, device=str(device))
    criterion = nn.CrossEntropyLoss()

    # def set_optimizer(model: ResNet18,opt: str,lr: float=0.1,momentum: float=0.9,weight_decay: float=5e-4)
    optimizer = set_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay)

    # best_work, best_time = find_best_work(use_cuda)
    # best_work, 8

    print(f"\nBegin Training, Optimizer {optimizer}, Device {device}")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'Acc (%)':<8} | {'Data Time':<10} | {'Train Time':<10}")
    print("-" * 60)

    for epoch in range(1, 5 + 1):
        loss, acc, d_time, t_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"{epoch:<6} | " f"{loss:<10.4f} | " f"{acc:<10.2f} | " f"{d_time:<12.4f} | " f"{t_time:<12.4f}")

if __name__ == '__main__':
    main()