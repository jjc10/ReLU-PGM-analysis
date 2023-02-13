import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.resnet.custom_relu import CustomReLU


class BasicBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expansion: int = 1,
            downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = CustomReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

    def forward_get_code(self, x: Tensor):
        activation_codes = []
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out, relu_1_codes = self.relu.forward_get_code(out)
        activation_codes.append(relu_1_codes)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out, relu_2_codes = self.relu.forward_get_code(out)
        activation_codes.append(relu_2_codes)
        return out, activation_codes