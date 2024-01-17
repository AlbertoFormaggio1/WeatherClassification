import torch
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, in_feature, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=in_feature, out_features=hidden_size)
        self.layer_2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layer_1(x)
        x = nn.functional.relu(x)
        x = self.layer_2(x)
        return x
