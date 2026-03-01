# from einops import rearrange
# from jaxtyping import Float
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResidualBlockJIT(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            strides: int = 1,
    ) -> None:
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
            x: Tensor,
    ) -> Tensor:
        identity = self.shortcut(x)
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet18JIT(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)


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
        layers = [ResidualBlockJIT(self.in_channels, out_channels, strides=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                ResidualBlockJIT(out_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = rearrange(x, 'b c 1 1 -> b c')
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class ResidualBlockNoBNJIT(nn.Module):
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

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, stride=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=strides, bias=False
                )
            )

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += identity
        return F.relu(out)


class ResNet18NoBNJIT(nn.Module):
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
        # self.bn1 = nn.BatchNorm2d(64)
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
        layers = [ResidualBlockNoBNJIT(self.in_channels, out_channels, strides=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                ResidualBlockNoBNJIT(out_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = F.relu(self.conv1(x))
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = rearrange(x, 'b c 1 1 -> b c')
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x