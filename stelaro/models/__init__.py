"""
    Machine learning model module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from random import randint
import numpy as np
from sklearn.metrics import f1_score
from torch import tensor, no_grad, float32, Tensor, zeros
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
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


class BaseClassifier:
    """A generic DNA read classifier."""
    def __init__(self):
        pass

    def get_parameters(self):
        raise NotImplementedError("Can't get parameters.")

    def predict(self, x_batch):
        """Classify DNA reads."""
        raise NotImplementedError("Can't predict.")

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        """Train a neural network.

        Args:
            model: Neural network
            train_loader: Training data loader
            validate_loader: Validation data loader
            optimizer: Optimizer
            max_n_epochs: Maximum number of training epochs.
            patience: Number of epochs during which the validation F1 score is
                allowed to decrease before early stop. Set to `None` to avoid
                early stop.
            device: Specify CPU or CUDA.
            mapping: Maps an index to a taxonomic description.
        """
        raise NotImplementedError("Can't train.")


class RandomClassifier(BaseClassifier):
    def __init__(self):
        pass

    def get_parameters(self):
        return None

    def predict(self, x_batch):
        n_reads = x_batch.shape[0]
        return [randint(0, self.n_classes - 1) for _ in range(n_reads)]

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        n_classes = 0
        for _, y_batch in tqdm(train_loader):
            n = max(y_batch)
            if n > n_classes:
                n_classes = n
        self.n_classes = n_classes
        return [], []


class MajorityClassifier(BaseClassifier):
    def __init__(self):
        pass

    def get_parameters(self):
        return None

    def predict(self, x_batch):
        n_reads = x_batch.shape[0]
        return [self.majority_class for _ in range(n_reads)]

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        counts = {}
        for _, y_batch in tqdm(train_loader):
            for y in y_batch:
                y = int(y)
                if y in counts:
                    counts[y] += 1
                else:
                    counts[y] = 1
        print(f"Class counts: {counts}")
        best_class, n = 0, 0
        for c, count in counts.items():
            if count > n:
                best_class = c
                n = count
        self.majority_class = best_class
        print(f"Majority class: {self.majority_class}")
        return [], []


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


def create_penalty_matrix(mapping) -> Tensor:
    """Create a penalty matrix that assigns a higher loss to more taxonomically
    erroneous predictions."""
    d = len(mapping)
    t = zeros((d, d))
    n_ranks = len(mapping['0'])
    for i in mapping:
        for j in range(d):
            j = str(j)
            union_length = 0
            for a, b in zip(mapping[i], mapping[j]):
                if a == b:
                    union_length += 1
                else:
                    break
            penalty = (n_ranks - union_length) / n_ranks
            t[int(i), int(j)] = penalty
    return t


def penalized_cross_entropy(logits, targets, penalty_matrix):
    """Penalize some class predictions more than others.

    June 2025: This does not improve performance with the `version_1` dataset.
    """
    log_probs = functional.log_softmax(logits, dim=1)
    penalties = penalty_matrix[targets]
    weighted_loss = (log_probs.exp() * penalties).sum(dim=1)
    return weighted_loss.mean()


def evaluate(classifier, loader, device, mapping):
    """Evaluate the F1 score at different taxonomic levels."""
    mappings = obtain_rank_based_mappings(mapping)
    ranks = []
    n_batches = 0
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.type(float32).to(device)
            x_batch = x_batch.permute(0, 2, 1)  # Swap channels and sequence.
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            ranks.append(rank_based_f1_score(mappings, y_batch, predictions))
            n_batches += 1
    collapsed_ranks = [np.zeros(len(r)) for r in ranks[0]]
    for rank in ranks:
        for i, result in enumerate(rank):
            collapsed_ranks[i] += result
    for i in range(len(collapsed_ranks)):
        collapsed_ranks[i] = np.mean(collapsed_ranks[i])
        collapsed_ranks[i] /= n_batches
    collapsed_ranks = [r for r in reversed(collapsed_ranks)]
    for i in range(len(collapsed_ranks) - 1):
        assert collapsed_ranks[i] > collapsed_ranks[i + 1]
    return collapsed_ranks
