import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, depth, hidden_size, output_size = 10):
        super(Net, self).__init__()
        self.input_size = input_size
        self.depth = depth
        self.first_layer = nn.Linear(self.input_size, hidden_size)
        self.list_layers = nn.ModuleList()
        for _ in range(self.depth-1):
            self.list_layers.append(nn.Linear(hidden_size, hidden_size))

        self.last_layer = nn.Linear(hidden_size, output_size)

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
        # check which neurons are positive because the negative ones will be turned off by RELU (set to 0)
        layer_codes.append(x > 0)
        for layer in self.list_layers:
            x = layer(x)
            layer_codes.append(x > 0)
            x = F.relu(x)

        x = self.last_layer(x)
        log_out = F.log_softmax(x, dim=1)
        return log_out, layer_codes

    def get_layer_widths(self):
        widths = [self.first_layer.out_features]
        for hidden_layer in self.list_layers:
            widths.append(hidden_layer.out_features)
        return widths



def build_model(config_dict):
    output_size = len(config_dict['classes']) if 'classes' in config_dict else 10
    network = Net(input_size=config_dict['input_size'],
                  depth=config_dict['depth'], hidden_size=config_dict['hidden_size'], output_size = output_size)
    return network

