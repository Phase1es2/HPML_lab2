import torch
from torch import nn
from torch.utils.data import DataLoader

from lab2_data import get_train_dataset, get_test_dataset
from model import ResNet18
from model_jit import ResNet18JIT, ResNet18NoBNJIT
from train import train_one_epoch
from utils import get_parser, set_optimizer, benchmark_latency
from torchsummary import summary

MODEL_PATH = "best_model_jit.pt"

def main():
    args = get_parser().parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.eval:
        try:
            model = torch.jit.load(MODEL_PATH)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error loading TorchScript model from path {MODEL_PATH}: {e}")
            return

        print("=" * 60)
        print(f"✓ Successfully loaded TorchScript model from: {MODEL_PATH}")
        print(f"✓ Evaluation device: {device}")
        print("=" * 60)

        test_data_set = get_test_dataset()
        test_loader = DataLoader(
            test_data_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True if use_cuda else False
        )

        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        top1_acc = 100. * correct / total
        print("Accuracy: {:.2f}%".format(top1_acc))
        return
    if args.latency:
        original_model = ResNet18(num_classes=10).to(device)
        try:
            ts_model = torch.jit.load(MODEL_PATH).to(device)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error loading TorchScript model from path {MODEL_PATH}: {e}")
            return
        latency_org = benchmark_latency(original_model, device)
        latency_ts = benchmark_latency(ts_model, device)
        speedup = latency_org / latency_ts
        print(f"PyTorch latency on {'GPU' if use_cuda else 'CPU'}: {latency_org:.2f} ms")
        print(f"TorchScript latency on {'GPU' if use_cuda else 'CPU'}: {latency_ts:.2f} ms")
        print("Speedup: {:.2f}x".format(speedup))
    else:
        data_set = get_train_dataset()
        train_loader = DataLoader(
            data_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True if use_cuda else False
        )

        model = ResNet18NoBNJIT(num_classes=10).to(device) if args.no_batch else ResNet18JIT(num_classes=10).to(device)


        summary(model, input_size=(3, 32, 32), device=str(device))
        criterion = nn.CrossEntropyLoss()
        # def set_optimizer(model: ResNet18,opt: str,lr: float=0.1,momentum: float=0.9,weight_decay: float=5e-4)
        optimizer = set_optimizer(model, args.opt, args.lr, args.momentum, args.weight_decay, args.nesterov)

        print(f"\nBegin Training, Optimizer {optimizer}, Device {device}")
        print(f"{'Epoch':<6} | {'Loss':<8} | {'Acc (%)':<8} | {'Data Time':<10} | {'Train Time':<10} | {'Running Time':<10}")
        print("-" * 60)
        best_acc = -1.0
        # C4: Training on GPU vs CPU
        printed_graph = False
        for epoch in range(1, args.epochs + 1):
            loss, acc, d_time, t_time, r_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"{epoch:<6} | " f"{loss:<10.4f} | " f"{acc:<10.2f} | " f"{d_time:<12.4f} | " f"{t_time:<12.4f} | " f"{r_time:<12.4f}" )

            if acc > best_acc:
                best_acc = acc
                was_training = model.training
                model.eval()

                if args.converting == "scripting":
                    scripted = torch.jit.script(model)
                    torch.jit.save(scripted, MODEL_PATH)
                    print(f"Using scripting --> New best acc: {best_acc:.2f}%, JIT model saved to {MODEL_PATH}")
                    if not printed_graph:
                        print("==== TorchScript Graph ====")
                        print(scripted.code)

                        # print("==== TorchScript Graph ====")
                        # print(scripted.graph)

                        # print("==== TorchScript Inlined Graph ====")
                        # print(scripted.inlined_graph)
                        printed_graph = True

                elif args.converting == "tracing":
                    example = torch.randn(1, 3, 32, 32).to(device)
                    traced = torch.jit.trace(model, example)
                    torch.jit.save(traced, MODEL_PATH)
                    print(f"Using tracing --> New best acc: {best_acc:.2f}%, JIT model saved to {MODEL_PATH}")
                    if not printed_graph:
                        print("==== TorchScript Graph ====")
                        print(traced.graph)

                        print("==== TorchScript Inlined Graph ====")
                        print(traced.inlined_graph)
                        printed_graph = True

                if was_training:
                    model.train()

if __name__ == "__main__":
    main()
