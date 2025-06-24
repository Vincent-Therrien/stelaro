"""
    Machine learning model module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from sklearn.metrics import f1_score
from torch import tensor, no_grad,argmax, half, float32, Tensor
from torch.nn import functional
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class SyntheticReadDataset(Dataset):
    """Dataset containing one-hot encoded synthetic reads."""
    def __init__(self, directory: str, mapping: str = None):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
        """
        self.x = np.load(directory + "x.npy", mmap_mode=mapping)
        self.y = np.load(directory + "y.npy", mmap_mode=mapping)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = ((x[:, None] & (1 << (3 - np.arange(4)))) > 0).astype(float)
        y = self.y[idx]
        return tensor(x), tensor(y)


def obtain_rank_based_mappings(mapping: dict) -> list[dict]:
    """Convert the flat-level indices to higher taxonomic rank indices."""
    keys = []
    mappings = []
    for i in range(len(mapping['0']) - 1, -1, -1):
        keys.append(set())
        mappings.append({})
        for j, labels in mapping.items():
            label = labels[i]
            if label not in keys[-1]:
                keys[-1].add(label)
            mappings[-1][int(j)] = len(keys[-1]) - 1
        if i == len(mapping['0']) - 1:
            assert len(keys[-1]) == len(mapping), "Duplicate names."
    return mappings


def rank_based_f1_score(
        mappings: dict, target: list[int], predictions: list[int]
        ) -> list[np.ndarray]:
    """Evaluate F1 score at multiple taxonomic ranks."""
    f = []
    for mapping in mappings:
        labels = list(set(mapping.values()))
        normalized_target = [mapping[int(v)] for v in list(target)]
        normalized_pred = [mapping[int(v)] for v in list(predictions)]
        f.append(
            f1_score(
                normalized_target,
                normalized_pred,
                average=None,
                labels=labels,
                zero_division=0.0
            )
        )
    return f


def penalized_cross_entropy(logits, targets, penalty_matrix):
    log_probs = functional.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    batch_size = logits.size(0)
    penalties = penalty_matrix[targets]
    weighted_loss = (probs * penalties).sum(dim=1)
    return weighted_loss.mean()


def evaluate(model, loader, device, mapping):
    """Evaluate the F1 score at different taxonomic levels."""
    model.eval()
    mappings = obtain_rank_based_mappings(mapping)
    ranks = []
    n_batches = 0
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.type(float32).to(device)
            x_batch = x_batch.permute(0, 2, 1)  # Swap channels and sequence.
            y_batch = y_batch.to("cpu")
            predictions = argmax(model(x_batch), dim=1).to("cpu")
            ranks.append(rank_based_f1_score(mappings, y_batch, predictions))
            n_batches += 1
    collapsed_ranks = [np.zeros(len(r)) for r in ranks[0]]
    for rank in ranks:
        for i, result in enumerate(rank):
            collapsed_ranks[i] += result
    for i in range(len(collapsed_ranks)):
        collapsed_ranks[i] = np.mean(collapsed_ranks[i])
        collapsed_ranks[i] /= n_batches
    return collapsed_ranks


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
    best_f1 = 0.0
    for epoch in range(max_n_epochs):
        model.train()
        losses.append(0)
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.type(float32).to(device)
            x_batch = x_batch.permute(0, 2, 1)  # Swap channels and sequence.
            y_batch = y_batch.type(float32).to(device).to(int)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            losses[-1] += loss.item()
        f1 = evaluate(model, validate_loader, device, mapping)
        f1 = [float(f) for f in f1]
        average_f_scores.append(f1)
        if f1[0] > best_f1:
            best_f1 = f1[0]
        if f1[0] < best_f1:
            patience -= 1
        print(
            f"{epoch+1}/{max_n_epochs}",
            f"Loss: {losses[-1]:.2f}.",
            "F1: ", [f"{f:.5}" for f in f1], " .",
            f"Patience: {patience}"
        )
        if patience < 0:
            print("Stopping early.")
            break
    average_f_scores = list(np.array(average_f_scores).T)
    return losses, average_f_scores
