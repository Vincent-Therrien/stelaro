"""
    Machine learning model module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

from random import randint, shuffle
from typing import Callable
import numpy as np
from sklearn.metrics import f1_score, precision_score
import torch
from torch import argmax, tensor, no_grad, float32, Tensor, zeros, cat
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time


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


class SyntheticTetramerDataset(Dataset):
    """Dataset containing one-hot encoded synthetic reads."""
    def __init__(self, directory: str, mapping: str = None, balance=False):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
        """
        if balance:
            x = np.load(directory + "x.npy", mmap_mode=mapping)
            y = np.load(directory + "y.npy", mmap_mode=mapping)
            unique_values, counts = np.unique(y, return_counts = True)
            n = min(counts)
            N = n * len(unique_values)
            self.x = np.zeros((N, 375), dtype=np.uint8)
            self.y = np.zeros(N, dtype=np.uint16)
            indices = list(range(len(y)))
            shuffle(indices)
            amounts = {}
            for v in unique_values:
                amounts[v] = 0
            i = 0
            for index in indices:
                label = y[index]
                if amounts[label] < n:
                    self.x[i] = x[index]
                    self.y[i] = label
                    i += 1
                    amounts[label] += 1
                    if i >= N:
                        break
        else:
            self.x = np.load(directory + "x.npy", mmap_mode=mapping)
            self.y = np.load(directory + "y.npy", mmap_mode=mapping)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x), tensor(y)


class SyntheticMultiLevelTetramerDataset(Dataset):
    """Dataset containing one-hot encoded synthetic reads."""
    def __init__(
            self,
            directory: str,
            mapping: dict,
            selection: tuple[str],
            resolution: int,
            memory_mapping: str = None,
            balance: bool = False,
            other_factor: float = 2.0,
        ):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Dictionary mapping indices to taxa.
            selection: Taxonomic group to select within the mapping.
            resolution: Number of ranks to discriminate.
            memory_mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
            balance: If True, randomly eliminate data from the majority class
                until it comprises at most twice the number of data points from
                the second biggest class.
            other_factor: Adjusts balancing by determining the maximum number
                of data points in the other class as a factor of the number
                of data points in the second biggest class.
        """
        self.selection = selection
        self.conversion_table = tensor([i for i in range(len(mapping))])
        n_levels = resolution if resolution else len(mapping[str(0)])
        # Select a taxon.
        if selection:
            assert type(selection) is tuple, "Unexpected type."
            n_taxa = 0
            for i in range(len(mapping)):
                observation = mapping[str(i)]
                for level in range(len(selection)):
                    if level >= len(selection):
                        pass
                    elif selection[level] != observation[level]:
                        self.conversion_table[i] = -1
                        break
                else:
                    self.conversion_table[i] = n_taxa
                    n_taxa += 1
        # Adjust the resolution.
        taxa = {}
        if resolution:
            for i in range(len(mapping)):
                if self.conversion_table[i] != -1:
                    observation = mapping[str(i)]
                    level = tuple(observation)[:len(selection) + n_levels]
                    if level not in taxa:
                        taxa[level] = len(taxa)
                    self.conversion_table[i] = taxa[level]
        self.target_mapping = taxa
        if other_factor:
            self.target_mapping[("other", )] = len(taxa)
        self.mapping = {str(v): k for k, v in self.target_mapping.items()}
        self.n_classes = len(taxa)
        # Replace wildcards.
        for i in range(len(mapping)):
            if self.conversion_table[i] == -1:
                self.conversion_table[i] = n_taxa
        if balance:
            self.balance(directory, memory_mapping, other_factor)
        else:
            self.x = np.load(directory + "x.npy", mmap_mode=memory_mapping)
            self.y = np.load(directory + "y.npy", mmap_mode=memory_mapping)
            self.y = self.conversion_table[self.y]

    def balance(
            self,
            directory: str,
            memory_mapping: str,
            other_factor: float
            ):
        x = np.load(directory + "x.npy", mmap_mode=memory_mapping)
        y = np.load(directory + "y.npy", mmap_mode=memory_mapping)
        y = self.conversion_table[y]
        unique_values, counts = np.unique(y, return_counts = True)
        if other_factor:
            second_biggest = sorted(counts, reverse=True)[1]
            N = int(second_biggest * other_factor)
        else:
            n = min(counts)
            N = n * 4
        total_points = 0
        for count in counts:
            if count < N:
                total_points += count
            else:
                total_points += N
        L = len(x[0])
        self.x = np.zeros((total_points, L), dtype=np.uint8)
        self.y = np.zeros(total_points, dtype=np.uint16)
        indices = list(range(len(y)))
        shuffle(indices)
        amounts = {}
        for v in unique_values:
            amounts[v] = 0
        i = 0
        for index in indices:
            label = int(y[index])
            if amounts[label] < N:
                self.x[i] = x[index]
                self.y[i] = label
                i += 1
                amounts[label] += 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # x = self.x[idx]
        # y = self.y[idx]
        return self.x[idx], self.y[idx]


class BasicReadDataset(Dataset):
    """Dataset containing one-hot encoded synthetic reads."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

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
            n = max(y_batch.long())
            if n > n_classes:
                n_classes = n
        self.n_classes = n_classes
        return [], [], []

    def train_large_dataset(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            evaluation_interval: int,
            n_max_reads: int,
            patience: int,
            ):
        n_classes = 0
        n_total_reads = 0
        for _, y_batch in tqdm(train_loader):
            n = max(y_batch)
            if n > n_classes:
                n_classes = n
            n_total_reads += len(y_batch)
            if n_total_reads > n_max_reads:
                break
        self.n_classes = n_classes
        return [], [], []


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


def penalized_cross_entropy(y_pred, y_true, penalty):
    """
    y_pred: [B, M] (raw logits)
    y_true: [B] (class indices)
    penalty: [M, M] (penalty matrix, 0 = good, 1 = bad)
    """
    probs = functional.softmax(y_pred, dim=-1)  # [B, M]
    penalty_rows = penalty[y_true]  # [B, M]
    loss = -torch.sum((1 - penalty_rows) * torch.log(probs + 1e-12), dim=-1)
    return loss.sum()


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


def rank_based_precision(
        mappings: dict, target: list[int], predictions: list[int]
        ) -> list[np.ndarray]:
    """Evaluate F1 score at multiple taxonomic ranks."""
    p = []
    for mapping in mappings:
        labels = list(set(mapping.values()))
        normalized_target = [mapping[int(v)] for v in list(target)]
        normalized_pred = [mapping[int(v)] for v in list(predictions)]
        p.append(
            precision_score(
                normalized_target,
                normalized_pred,
                average=None,
                labels=labels,
                zero_division=0.0
            )
        )
    return p


def evaluate(classifier, loader, device, mapping, time_limit: float = None):
    """Evaluate the F1 score at different taxonomic levels."""
    mappings = obtain_rank_based_mappings(mapping)
    predicted_y = []
    real_y = []
    from torch import no_grad
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            predicted_y += predictions
            real_y += y_batch
    ranks =  rank_based_f1_score(mappings, real_y, predicted_y)
    collapsed = []
    for rank in ranks[::-1]:
        collapsed.append(np.mean(rank))
    return collapsed


def evaluate_precision(classifier, loader, device, mapping):
    """Evaluate the precision score at different taxonomic levels."""
    mappings = obtain_rank_based_mappings(mapping)
    predicted_y = []
    real_y = []
    from torch import no_grad
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            predicted_y += predictions
            real_y += y_batch
    ranks =  rank_based_precision(mappings, real_y, predicted_y)
    collapsed = []
    for rank in ranks[::-1]:
        collapsed.append(np.mean(rank))
    return collapsed


def estimate_precision(classifier, loader, device, mapping, time_limit = 30):
    """Evaluate the precision score at different taxonomic levels."""
    mappings = obtain_rank_based_mappings(mapping)
    ranks = []
    all_pred = []
    all_y = []
    a = time()
    n = 0
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.long().to(device)
            predictions = classifier.predict(x_batch)
            all_pred.append(predictions)
            all_y.append(y_batch)
            n += len(y_batch)
            if time() - a > time_limit:
                print(f"Halting evaluation after {n} data points.")
                break
    all_y_cpu = cat(all_y, dim=0).to("cpu")
    all_pred_cpu = cat(all_pred, dim=0).to("cpu")
    ranks.append(rank_based_precision(mappings, all_y_cpu, all_pred_cpu))
    collapsed_ranks = [np.zeros(len(r)) for r in ranks[0]]
    for rank in ranks:
        for i, result in enumerate(rank):
            collapsed_ranks[i] += result
    for i in range(len(collapsed_ranks)):
        collapsed_ranks[i] = np.mean(collapsed_ranks[i])
    collapsed_ranks = [r for r in reversed(collapsed_ranks)]
    for i in range(len(collapsed_ranks) - 1):
        assert collapsed_ranks[i] > collapsed_ranks[i + 1]
    return collapsed_ranks


def confusion_matrix(classifier, loader, device, mapping) -> np.ndarray:
    """Returns: A confusion matrix with rows corresponding to true labels."""
    matrix = np.zeros((len(mapping), len(mapping)))
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            for y, p in zip(y_batch, predictions):
                matrix[y][p] += 1
    return matrix


def get_f1_by_category(confusion) -> np.ndarray:
    f1 = np.zeros(len(confusion))
    for i in range(len(confusion)):
        TP = confusion[i][i]
        FN = sum(confusion[i]) - TP
        FP = np.sum(confusion, axis=0)[i] - TP
        d = 2 * TP + FP + FN
        if d:
            f1[i] = 2 * TP / d
    return f1


class Classifier(BaseClassifier):
    """A DNA read classification dataset."""
    def __init__(
            self,
            length: int,
            mapping: dict,
            device: str,
            model: any,
            formatter: Callable,
            clip: bool
            ):
        """
        Args:
            length: Length of the sequences processed by the model.
            mapping: Dictionary mapping a taxon to an index.
            device: Device onto which run computations.
            model: Neural network (e.g. PyTorch Module).
            formatter: Function that converts the default input (tetramers)
                into another format. If `None`, tetramers are used as input.
            clip: If `True`, clip gradients to 1.0.
        """
        self.length = length
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping
        self.formatter = formatter
        self.clip = clip

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        if self.formatter:
            x_batch = self.formatter(x_batch)
        output = self.model(x_batch)
        if type(output) is tuple:
            output = output[0]
        predictions = argmax(output, dim=1)
        return predictions

    def _compute_loss(
            self,
            x_batch,
            y_batch,
            loss_function_type,
            loss_function
            ) -> float:
        x_batch = x_batch.long().to(self.device)
        if self.formatter:
            x_batch = self.formatter(x_batch)
        y_batch = y_batch.long().to(self.device)
        output = self.model(x_batch)
        if loss_function_type == "supervised":
            loss = loss_function(output, y_batch)
        elif loss_function_type =="unsupervised":
            loss = loss_function(output, x_batch)
        elif loss_function_type =="semi-supervised":
            loss = loss_function(output, x_batch, y_batch)
        return loss

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 2000
            ):
        TRAINING_TYPES = ("supervised", "unsupervised", "semi-supervised")
        assert loss_function_type in TRAINING_TYPES, "Invalid training type."
        losses = []
        validation_losses = []
        average_f_scores = []
        best_f1 = 0.0
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            progress = 0
            for x_batch, y_batch in tqdm(train_loader):
                loss = self._compute_loss(
                    x_batch, y_batch, loss_function_type, loss_function
                )
                optimizer.zero_grad()
                loss.backward()
                if self.clip:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses[-1] += loss.item()
                if progress and progress % evaluation_interval == 0:
                    ps = estimate_precision(self, validate_loader, self.device, self.mapping)
                    ps = [float(p) for p in ps]
                    p_msg = [float(f"{p:.5}") for p in ps]
                    print(f"P: {p_msg}")
                progress += 1
            losses[-1] /= len(train_loader.dataset)
            losses[-1] *= self.length
            validation_losses.append(0)
            for x_batch, y_batch in validate_loader:
                loss = self._compute_loss(
                    x_batch, y_batch, loss_function_type, loss_function
                )
                validation_losses[-1] += loss.item()
            validation_losses[-1] /= len(validate_loader.dataset)
            validation_losses[-1] *= self.length
            f1 = evaluate(self, validate_loader, self.device, self.mapping, 60)
            f1 = [float(f) for f in f1]
            f1_msg = [float(f"{f:.5}") for f in f1]
            average_f_scores.append(f1)
            if f1[-1] > best_f1:
                best_f1 = f1[-1]
            if f1[-1] < best_f1:
                patience -= 1
            ps = estimate_precision(self, validate_loader, self.device, self.mapping)
            ps = [float(p) for p in ps]
            p_msg = [float(f"{p:.5}") for p in ps]
            print(
                f"{epoch+1}/{max_n_epochs}",
                f"T loss: {losses[-1]:.5f}.",
                f"V loss: {validation_losses[-1]:.5f}.",
                f"F1: {f1_msg}.",
                f"P: {p_msg}",
                f"Patience: {patience}"
            )
            if patience <= 0:
                print("The model is overfitting; stopping early.")
                break
        average_f_scores = list(np.array(average_f_scores).T)
        return losses, average_f_scores, validation_losses

    def train_fast(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            ):
        TRAINING_TYPES = ("supervised", "unsupervised", "semi-supervised")
        assert loss_function_type in TRAINING_TYPES, "Invalid training type."
        losses = []
        validation_losses = []
        average_f_scores = []
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            progress = 0
            for x_batch, y_batch in tqdm(train_loader):
                loss = self._compute_loss(
                    x_batch, y_batch, loss_function_type, loss_function
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[-1] += loss.item()
                if progress and progress % 1000 == 0:
                    ps = estimate_precision(self, validate_loader, self.device, self.mapping)
                    ps = [float(p) for p in ps]
                    p_msg = [float(f"{p:.5}") for p in ps]
                    print(f"P: {p_msg}")
                progress += 1
            losses[-1] /= len(train_loader.dataset)
            losses[-1] *= self.length
            validation_losses.append(0)
        average_f_scores = list(np.array(average_f_scores).T)
        return losses, average_f_scores, validation_losses
