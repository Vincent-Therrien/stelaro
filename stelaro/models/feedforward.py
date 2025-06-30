"""
    Simple feedforward neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

import numpy as np
from torch import argmax, float32
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, Dropout, MaxPool1d)
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import evaluate, BaseClassifier


class Classifier(BaseClassifier):
    """A DNA read classification dataset."""
    def __init__(self, length, mapping, device, model):
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        predictions = argmax(self.model(x_batch), dim=1).to("cpu")
        return predictions

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = []
        average_f_scores = []
        best_f1 = 0.0
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.type(float32).to(self.device)
                # Swap channels and sequence.
                x_batch = x_batch.permute(0, 2, 1)
                y_batch = y_batch.type(float32).to(self.device).to(int)
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
                loss.backward()
                optimizer.step()
                losses[-1] += loss.item()
            f1 = evaluate(self, validate_loader, self.device, self.mapping)
            f1 = [float(f) for f in f1]
            average_f_scores.append(f1)
            if f1[-1] > best_f1:
                best_f1 = f1[-1]
            if f1[-1] < best_f1:
                patience -= 1
            loss_msg = [f"{f:.5}" for f in f1]
            print(
                f"{epoch+1}/{max_n_epochs}",
                f"Loss: {losses[-1]:.2f}.",
                f"F1: {loss_msg}.",
                f"Patience: {patience}"
            )
            if patience <= 0:
                print("The model is overfitting; stopping early.")
                break
        average_f_scores = list(np.array(average_f_scores).T)
        return losses, average_f_scores


class MLP_1(Module):
    def __init__(self, N, M):
        super(MLP_1, self).__init__()
        self.fc = Sequential(
            Flatten(),
            Linear(N * 4, int(N / 2)),
            ReLU(),
            Linear(int(N / 2), int(N / 4)),
            ReLU(),
            Linear(int(N / 4), M)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.to(float)


class MLP_1_dropout(Module):
    def __init__(self, N, M):
        super(MLP_1_dropout, self).__init__()
        self.fc = Sequential(
            Flatten(),
            Dropout(0.2),
            Linear(N * 4, int(N / 2)),
            ReLU(),
            Dropout(0.2),
            Linear(int(N / 2), int(N / 4)),
            ReLU(),
            Dropout(0.2),
            Linear(int(N / 4), M)
        )

    def forward(self, x):
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


class CNN_1_dropout(Module):
    def __init__(self, N, M):
        super(CNN_1_dropout, self).__init__()
        self.conv = Sequential(
            Conv1d(4, 32, kernel_size=7, padding=3),
            ReLU(),
            # MaxPool1d(2),
            Conv1d(32, 64, kernel_size=5, padding=2),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Dropout(0.2),
            Linear(N * 64, 128),
            ReLU(),
            Dropout(0.2),
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


class CNN_2_dropout(Module):
    def __init__(self, N, M):
        super(CNN_2_dropout, self).__init__()
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
            Dropout(0.2),
            Linear(N * 128, int(N / 2)),
            ReLU(),
            Dropout(0.2),
            Linear(int(N / 2), 128),
            ReLU(),
            Dropout(0.2),
            Linear(128, M)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.to(float)
