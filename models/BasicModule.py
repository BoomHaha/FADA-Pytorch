import torch
import torch.nn.modules as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, s: str):
        self.load_state_dict(torch.load(s))

    def save(self, s: str):
        torch.save(self.state_dict(), s)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
