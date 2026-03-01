import torch
from torch import nn
from torch.utils.data import DataLoader

from lab2_data import get_train_dataset
from model import ResNet18, ResNet18NoBN
from train import train_one_epoch
from utils import get_parser, find_best_work, set_optimizer
from torchsummary import summary


"""
main function for lab2
as the requirement we take the ArgumentParser to determine which code to do

C3: I/O Optimization
    use 'uv run lab2.py --best_worker'

C4: Training on GPUs vs CPUs
    for GPU use 'uv run lab2.py --cuda'
    for CPU use 'uv run lab2.py' 

C5: Experimenting with Different Optimizers
    for this part we can set up the 
    --lr learning rate default=0.1
    --opt optimizer  ["sgd", "adam", "adagrad", "adadelta", "adamw", "rmsprop"]
    --epochs epochs default=5
    --batch_size batch_size default=128
    --weight_decay weight_decay default=5e-4
    --momentum momentum default=0.9c
    --nesterov default=False
    
    the default setting is 
    'uv run lab2.py'
    'uv run lab2.py --opt sgd --lr 0.1 --epochs 5 --batch_size 128 --weight_decay 5e-4 --momentum 0.9'
    
    for sgd use 'uv run lab2.py --lr 0.1 --opt sgd'
    for sgd with nesterov use 'uv run lab2.py --opt sgd --nesterov'
    for adam use 'uv run lab2.py --opt adam'

C6: Experimenting without Batch Norm
    the model without batch normalization is in model, which is ResNet18NoBN
    for experimeting without batch norm use 'uv run lab2.py --no_batch'
    

"""


def main():
    args = get_parser().parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    if args.best_worker:
        best_work, best_time = find_best_work(use_cuda)
    else:
        num_workers = args.num_workers
        epochs = args.epochs
        device = torch.device("cuda" if use_cuda else "cpu")
        data_set = get_train_dataset()
        train_loader = DataLoader(
            data_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True if use_cuda else False
        )

        model = ResNet18NoBN(num_classes=10).to(device) if args.no_batch else ResNet18(num_classes=10).to(device)

        summary(model, input_size=(3, 32, 32), device=str(device))
        criterion = nn.CrossEntropyLoss()
        # def set_optimizer(model: ResNet18,opt: str,lr: float=0.1,momentum: float=0.9,weight_decay: float=5e-4)
        optimizer = set_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay, args.nesterov)

        print(f"\nBegin Training, Optimizer {optimizer}, Device {device}")
        print(f"{'Epoch':<6} | {'Loss':<8} | {'Acc (%)':<8} | {'Data Time':<10} | {'Train Time':<10} | {'Running Time':<10}")
        print("-" * 60)
        # C4: Training on GPU vs CPU
        for epoch in range(1, args.epochs + 1):
            loss, acc, d_time, t_time, r_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"{epoch:<6} | " f"{loss:<10.4f} | " f"{acc:<10.2f} | " f"{d_time:<12.4f} | " f"{t_time:<12.4f} | " f"{r_time:<12.4f}" )


if __name__ == "__main__":
    main()