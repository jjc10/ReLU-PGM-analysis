import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, depht, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.depht = depht
        self.first_layer = nn.Linear(self.input_size, hidden_size)
        self.list_layers = []
        for _ in range(self.depht-1):
            self.list_layers.append(nn.Linear(hidden_size, hidden_size))

        self.last_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):  # usual forward pass
        x = x.view(-1, self.input_size)
        x = F.relu(self.first_layer(x))
        for layer in self.list_layers:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        log_out = F.log_softmax(x, dim=1)
        return log_out

    def forward_get_code(self, x):  # forward pass with collecting data on relu activation
        layer_codes = []
        x = x.view(-1, self.input_size)
        x = self.first_layer(x)
        layer_codes.append(x > 0)
        for layer in self.list_layers:
            x = layer(x)
            layer_codes.append(x > 0)
            x = F.relu(x)

        x = self.last_layer(x)
        log_out = F.log_softmax(x, dim=1)
        return log_out, layer_codes


def build_model(input_size, config_dict):
    network = Net(input_size=input_size,
                  depht=config_dict['depht'], hidden_size=config_dict['hidden_size'])
    return network
