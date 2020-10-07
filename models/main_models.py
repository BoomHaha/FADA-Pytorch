import torch.nn.modules as nn
from models.BasicModule import BasicModule, Flatten


class DCD(BasicModule):
    def __init__(self, input_dim=128, hidden_dim=64, out_dim=4):
        super(DCD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(BasicModule):
    def __init__(self, input_dim=84, out_dim=10):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
