"""
    Feedforward neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from torch import nn

class CNN_1(nn.Module):
    def __init__(self, N, M):
        super(CNN_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * 64, 128),
            nn.ReLU(),
            nn.Linear(128, M)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
