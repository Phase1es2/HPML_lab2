import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam", "adagrad", "adadelta", "adamw", "rmsprop"], help="optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--data_path", type=str , help="data path")
    parser.add_argument("--num_workers", type=int, default=0, help="number of data loading workers")
    return parser


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    print(args)
    print(args.opt)
    print(args.lr)