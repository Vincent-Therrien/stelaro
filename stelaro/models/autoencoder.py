"""
    Autoencoder neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

import numpy as np
from torch import argmax, float32, unsqueeze
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, MSELoss, MaxPool1d, ConvTranspose1d,
                      Sigmoid, LeakyReLU)
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import evaluate, BaseClassifier


class Classifier(BaseClassifier):
    """An autoencoder-based DNA read classification dataset."""
    def __init__(self, length, mapping, device, model):
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        logits, _ = self.model(x_batch)
        predictions = argmax(logits, dim=1).to("cpu")
        return predictions

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        reconstruction_criterion = MSELoss()
        classification_criterion = CrossEntropyLoss()
        lamda = 0.5  # Classification / reconstruction weight.
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
                classification, reconstruction = self.model(x_batch)
                reconstruction_loss = reconstruction_criterion(
                    reconstruction, x_batch
                )
                classification_loss = classification_criterion(
                    classification, y_batch
                )
                loss = reconstruction_loss + lamda * classification_loss
                optimizer.zero_grad()
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


class Unsqueeze(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AA_1(Module):
    def __init__(self, N, M):
        super(AA_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Flatten(),
            LeakyReLU(),
            Linear(N * 4, N),
            LeakyReLU(),
            Linear(N, N // 2),
            LeakyReLU(),
            Linear(N // 2, LATENT_SPACE_SIZE)
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction


class CAA_1(Module):
    def __init__(self, N, M):
        super(CAA_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Conv1d(4, 32, kernel_size=5, padding=2),
            LeakyReLU(),
            MaxPool1d(kernel_size=2),
            Conv1d(32, 64, kernel_size=3, padding=1),
            LeakyReLU(),
            Flatten(),
            Linear(N * 64 // 2, N // 2),
            LeakyReLU(),
            Linear(N // 2, LATENT_SPACE_SIZE)
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction
