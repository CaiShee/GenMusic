import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: "list[int]", output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.hidLayer = nn.Sequential()
        i = 0
        while i < len(hidden_sizes)-1:
            self.hidLayer.append(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidLayer.append(nn.ReLU())
            i += 1
        self.fc2 = nn.Linear(hidden_sizes[len(hidden_sizes)-1], output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.hidLayer(out)
        out = self.fc2(out)
        return out
