import torchvision
import torchvision.transforms as transforms
"""
The dataset we are using is [IFAR-10 Dataset which is labeled subsets of the 80 millions tiny images dataset. 
CIFAR-10 and CIFAR-100 were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. (Sadly, the 80 million tiny images dataset 
has been thrown into the memory hole by its authors. Spotting the doublethink which was used to justify its erasure is left as an exercise for the reader.)
The CIFAR-10 dataset consists of 60000 ($32 \times 32$) color, 
images in 10 classes, with 6000 images per class. and there are 50000 training images and 10000 test images.
"""

# C1: Training in PyTorch
def get_train_dataset():
    """
    To pre-processing image data for better accuracy, we used data augmentation, here is the technique that we applied.
    - **Random Cropping** `RandomCrop()`: randomly crop out smaller section of the image, forcing the model to handle parts of objects or varied backgrounds.
    - **Horizontal/Vertical Flipping** `RandomHorizontalFlip()`/: flip an image horizontally with a certain probability. This is techniques the model to be invariant to flips.
    - **ToTensor** `ToTensor()`: convert image or numpy arrays into PyTorch tensors, which makes them usable by PyTorch models. Usually transforms the image from shape (Height, Width, Channel) with values in [0, 255] to shape (Channel, Height, Width) with values in [0, 1]
    - **Normalization** `Normalize(mean=[], std=[])`: Subtract the mean and divide by the standard deviation on each channel, usually based on the overall dataset 'statistics. This makes training more stable and faster, normlized inputs help the network learn more effectively.
    """
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def get_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


