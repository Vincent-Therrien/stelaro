"""
    Transformer neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2025
    - License: MIT
"""

import numpy as np
from torch import (argmax, float32, randn_like, bernoulli, ones_like,
                   rand_like, clamp)
from torch import sum as torch_sum
from torch import exp as torch_exp
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, MSELoss, MaxPool1d, ConvTranspose1d,
                      Sigmoid, LeakyReLU, functional)
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
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
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
