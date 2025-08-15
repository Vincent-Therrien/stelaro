"""
    Simple feedforward neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

import numpy as np
from torch import argmax, float32, bincount, cat
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
            permute: bool = True,
            ):
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = []
        average_f_scores = []
        best_f1 = 0.0
        # all_labels = cat([labels for _, labels in train_loader]).long()
        # class_counts = bincount(all_labels, minlength=max(all_labels) + 1)
        # class_weights = 1.0 / class_counts.float()
        # class_weights = class_weights.float()
        # criterion = CrossEntropyLoss(weight=class_weights.to(self.device)).float()
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            n_processed = 0
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.to(self.device)
                # Swap channels and sequence.
                if permute:
                    x_batch = x_batch.permute(0, 2, 1)
                y_batch = y_batch.long().to(self.device)
                #class_counts = bincount(y_batch, minlength=31)
                #print(class_counts)
                #raise RuntimeError
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[-1] += loss.item()
                #n_processed += len(y_batch)
                #if n_processed > 1_000_000:
                #    break
            f1 = evaluate(self, validate_loader, self.device, self.mapping, permute=permute)
            f1 = [float(f) for f in f1]
            average_f_scores.append(f1)
            if f1[-1] > best_f1:
                best_f1 = f1[-1]
            if f1[-1] < best_f1:
                patience -= 1
            loss_msg = [float(f"{f:.5}") for f in f1]
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

    def train_large_dataset(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            evaluation_interval: int,
            n_max_reads: int,
            patience: int,
            ):
        """Same as `train`, but this method assumes that the training set is
        infinite.
        """
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = [0]
        validation_losses = []
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.type(float32).to(self.device)
            # Swap channels and sequence.
            y_batch = y_batch.long().to(self.device)
            output = self.model(x_batch)
            loss = criterion(output, y_batch)
            # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[-1] += loss.item() / len(y_batch)
            n_reads_processed += len(y_batch)
            evaluation_countdown -= len(y_batch)
            if evaluation_countdown <= 0:
                evaluation_countdown = evaluation_interval
                f1 = evaluate(self, validate_loader, self.device, self.mapping, permute=False)
                f1 = [float(f) for f in f1]
                average_f_scores.append(f1)
                if f1[-1] > best_f1:
                    best_f1 = f1[-1]
                if f1[-1] < best_f1:
                    patience -= 1
                loss_msg = [float(f"{f:.5}") for f in f1]
                print(
                    f"N Reads: {n_reads_processed:_}",
                    f"Loss: {losses[-1]:.2f}.",
                    f"F1: {loss_msg}.",
                    f"Patience: {patience}"
                )
                losses.append(0)
                validation_losses.append(0)
                for x_batch, y_batch in validate_loader:
                    x_batch = x_batch.type(float32).to(self.device)
                    y_batch = y_batch.long().to(self.device)
                    output = self.model(x_batch)
                    loss = criterion(output, y_batch)
                    validation_losses[-1] += loss.item() / len(y_batch)
                if patience <= 0:
                    print("The model is overfitting; stopping early.")
                    break
            if n_reads_processed > n_max_reads:
                print("Reached the specified maximum number of reads.")
                break
        else:
            print("Exhausted all reads.")

        print(f"Processed {n_reads_processed:_} reads.")
        average_f_scores = list(np.array(average_f_scores).T)
        return losses[:-1], average_f_scores, validation_losses[:-1]


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


class CNN_2_pooling(Module):
    def __init__(self, N, M):
        super(CNN_2_pooling, self).__init__()
        self.conv = Sequential(
            Conv1d(4, 32, kernel_size=7, padding=3),
            ReLU(),
            MaxPool1d(2),
            Conv1d(32, 64, kernel_size=5, padding=2),
            ReLU(),
            MaxPool1d(2),
            Conv1d(64, 128, kernel_size=3, padding=1),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(N * 64 // 2, int(N / 2)),
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


class CNN_2_dropout_token(Module):
    def __init__(self, N, M):
        super(CNN_2_dropout_token, self).__init__()
        self.conv = Sequential(
            Conv1d(1, 32, kernel_size=7, padding=3),
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
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)
        return x.to(float)
