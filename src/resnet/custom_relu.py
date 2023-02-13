import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class CustomReLU(nn.ReLU):

    # returns a tuple of (output, binary code)
    def forward_get_code(self, input: Tensor):
        activation_codes = input > 0
        output = F.relu(input, inplace=self.inplace)
        return output, activation_codes
