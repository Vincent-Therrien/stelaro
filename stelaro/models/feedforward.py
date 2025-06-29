"""
    Simple feedforward neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from torch.nn import Module, Conv1d, ReLU, Sequential, Flatten, Linear


class MLP_1(Module):
    def __init__(self, N, M):
        super(MLP_1, self).__init__()
        self.conv = Sequential(
            Conv1d(4, 1, kernel_size=3, padding=1),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(N, int(N / 2)),
            ReLU(),
            Linear(int(N / 2), int(N / 4)),
            ReLU(),
            Linear(int(N / 4), M)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.to(float)



class CNN_1(Module):
    def __init__(self, N, M):
        super(CNN_1, self).__init__()
        self.conv = Sequential(
            Conv1d(4, 32, kernel_size=7, padding=3),
            ReLU(),
            Conv1d(32, 64, kernel_size=5, padding=2),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(N * 64, 128),
            ReLU(),
            Linear(128, M)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.to(float)


class CNN_2(Module):
    def __init__(self, N, M):
        super(CNN_2, self).__init__()
        self.conv = Sequential(
            Conv1d(4, 32, kernel_size=7, padding=3),
            ReLU(),
            Conv1d(32, 64, kernel_size=5, padding=2),
            ReLU(),
            Conv1d(64, 128, kernel_size=3, padding=1),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(N * 128, int(N / 2)),
            ReLU(),
            Linear(int(N / 2), 128),
            ReLU(),
            Linear(128, M)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.to(float)
