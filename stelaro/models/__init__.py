"""
    Machine learning model module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from sklearn.metrics import f1_score
from torch import tensor, no_grad,argmax
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SyntheticReadDataset(Dataset):
    def __init__(self, directory: str, mapping: str = None):
        self.x = np.load(directory + "x.npy", mmap_mode=mapping)
        self.y = np.load(directory + "y.npy", mmap_mode=mapping)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = ((x[:, None] & (1 << (3 - np.arange(4)))) > 0).astype(float)
        y = self.y[idx].astype(float)
        return tensor(x), tensor(y)


def evaluate(model, loader, device, mapping):
    """Evaluate the F1 score at different taxonomic levels."""
    model.eval()
    labels = [int(m) for m in list(mapping.keys())]
    f = np.zeros((len(labels)))
    n_batches = 0
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = argmax(model(x_batch), dim=1)
            # TODO: Account for taxonomic level.
            f_scores = f1_score(
                y_batch,
                predictions,
                average=None,
                labels=labels
            )
            n_batches += 1
            for i, value in f_scores:
                f[i] += value
    f /= n_batches
    return f


def train(
        model: Module,
        train_loader: DataLoader,
        validate_loader: DataLoader,
        criterion,
        optimizer,
        max_n_epochs: int,
        patience: int,
        device: str,
        mapping: dict,
        ):
    """Train a neural network.

    Args:
        model: Neural network
        train_loader: Training data loader
        validate_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        max_n_epochs: Maximum number of training epochs.
        patience: Number of epochs during which the validation F1 score is
            allowed to decrease before early stop. Set to `None` to avoid
            early stop.
        device: Specify CPU or CUDA.
        mapping: Maps an index to a taxonomic description.
    """
    losses = []
    average_f_scores = []
    for epoch in range(max_n_epochs):
        model.train()
        losses.append(0)
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            losses[-1] += loss.item()
        f = np.mean(evaluate(model, validate_loader))  # TODO: Taxa
        print(f"{epoch+1}/{max_n_epochs} Loss: {losses[-1]:.2f}. F1: {f:.2f}")
        average_f_scores.append(f)
    return losses, average_f_scores
