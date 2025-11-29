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
from torch import no_grad
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    def __init__(
            self,
            directory: str,
            mapping: str = None,
            balance: bool = False,
            labels: bool = True):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
            balance: If `True`, balance the dataset by eliminating frequent
                taxa.
            labels: If `True`, use classification labels. Incompatible with
                `balance` argument.
        """
        self.use_labels = labels
        if balance:
            x = np.load(directory + "x.npy", mmap_mode=mapping)
            y = np.load(directory + "y.npy", mmap_mode=mapping)
            unique_values, counts = np.unique(y, return_counts = True)
            n = min(counts)
            N = n * len(unique_values)
            self.x = np.zeros((N, 375), dtype=np.uint8)
            if labels:
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
                    if labels:
                        self.y[i] = label
                    i += 1
                    amounts[label] += 1
                    if i >= N:
                        break
        else:
            self.x = np.load(directory + "x.npy", mmap_mode=mapping)
            if labels:
                self.y = np.load(directory + "y.npy", mmap_mode=mapping)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.use_labels:
            y = self.y[idx]
            return tensor(x), tensor(y)
        else:
            return tensor(x)


class SyntheticMultiLevelTetramerDataset(Dataset):
    """Dataset containing tetramer-encoded synthetic reads."""
    def __init__(
            self,
            directory: str,
            mapping: dict,
            selection: tuple[str],
            resolution: int,
            memory_mapping: str = None,
            other_factor: float = None,
        ):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Dictionary mapping indices to taxa.
            selection: Taxonomic group to select within the mapping. For
                example, `('Archaea', )` selects all taxa within the Archaea
                domain.
            resolution: Number of ranks to discriminate. For example, if the
                `selection` specifies a phylum and `resolution` is set to `1`,
                taxa will be grouped by phylum.
            memory_mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
            other_factor: If set, keep taxa that do not belong to the
                selection at most `other_factor` times the number of
                occurrences of the second most frequent class.
        """
        self.selection = selection
        self.conversion_table = tensor([i for i in range(len(mapping))])
        n_levels = resolution if resolution > 0 else len(mapping[str(0)])
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
                self.conversion_table[i] = len(taxa) - 1
        if other_factor:
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
        non_other_counts = counts[:-1]
        second_biggest = sorted(non_other_counts, reverse=True)[0]
        N = int(second_biggest * other_factor)
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
        return tensor(self.x[idx]), tensor(self.y[idx])


class CustomTetramerDataset(Dataset):
    """Dataset containing tetramer-encoded synthetic reads."""
    def __init__(
            self,
            directory: str,
            mapping: dict,
            indices: list[int],
            memory_mapping: str = None,
            other_factor: float = None,
        ):
        """Args:
            directory: Path of the directory that contains the x and y files.
            mapping: Dictionary mapping indices to taxa.
            indices: List of indices in the original classes to preserve.
            memory_mapping: Either `None` (all in main memory) or `"r"`
                (memory-mapped, useful for very large datasets).
            other_factor: If set, keep taxa that do not belong to the
                selection at most `other_factor` times the number of
                occurrences of the second most frequent class.
        """
        self.conversion_table = tensor([i for i in range(len(mapping))])
        floor = 0
        self.mapping = {}
        for index in range(len(mapping)):
            if index in indices:
                taxon = tuple(mapping[str(index)])
                new_index = floor
                floor += 1
            else:
                taxon = "other"
                new_index = len(indices)
            self.mapping[str(new_index)] = taxon
            self.conversion_table[index] = new_index
        if other_factor:
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
        non_other_counts = counts[:-1]
        second_biggest = sorted(non_other_counts, reverse=True)[0]
        N = int(second_biggest * other_factor)
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
        return tensor(self.x[idx]), tensor(self.y[idx])


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
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            ):
        """Train a neural network.

        Args:
            model: Neural network.
            train_loader: Training data loader.
            validate_loader: Validation data loader.
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
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            ):
        n_classes = 0
        n_steps = 0
        for _, y_batch in tqdm(train_loader):
            n = max(y_batch.long())
            if n > n_classes:
                n_classes = n
            n_steps += 1
            if n_steps > 10_000:
                break
        self.n_classes = n_classes
        return None, None, None, None

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
            if labels in (["unknown", ], ("unknown", ), "other"):
                label = "unknown"
                assert label not in keys, "Unexpected unknown taxon."
                keys[-1].add(label)
                mappings[-1][int(j)] = len(keys[-1]) - 1
            else:
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
                average="macro",
                labels=labels,
                zero_division=0.0
            )
        )
    return f


def evaluate(classifier, loader, device, mapping, time_limit: float = None):
    """Evaluate the F1 score at different taxonomic levels."""
    mappings = obtain_rank_based_mappings(mapping)
    predicted_y = []
    real_y = []
    a = time()
    n = 0
    with no_grad():
        for x_batch, y_batch in loader:
            n += len(y_batch)
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            predicted_y += predictions
            real_y += y_batch
            if time_limit and time() - a > time_limit:
                print(f"Halting evaluation after {n} data points.")
                break

    ranks =  rank_based_f1_score(mappings, real_y, predicted_y)
    collapsed = []
    for rank in ranks[::-1]:
        collapsed.append(np.mean(rank))
    return collapsed


def benchmark_classifier(classifier, loader, device, mapping, time_limit: float = None):
    mappings = obtain_rank_based_mappings(mapping)
    predicted_y = []
    real_y = []
    a = time()
    n = 0
    with no_grad():
        for x_batch, y_batch in tqdm(loader):
            n += len(y_batch)
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.to("cpu")
            predictions = classifier.predict(x_batch)
            predicted_y += predictions
            real_y += y_batch
            if time_limit and time() - a > time_limit:
                print(f"Halting evaluation after {n} data points.")
                break

    def collapse(v):
        collapsed = []
        for rank in v[::-1]:
            collapsed.append(np.mean(rank))
        return collapsed

    f1 = collapse(rank_based_f1_score(mappings, real_y, predicted_y))
    print(f"F1 score: {f1}")
    macro = collapse(rank_based_precision(mappings, real_y, predicted_y, "macro"))
    print(f"Macro precision score: {macro}")
    weighted = collapse(rank_based_precision(mappings, real_y, predicted_y, "weighted"))
    print(f"Weighted precision score: {weighted}")
    matrix = np.zeros((len(mapping), len(mapping)))
    for y, p in zip(real_y, predicted_y):
        matrix[y][p] += 1
    plt.matshow(matrix)
    plt.show()

def rank_based_precision(
        mappings: dict,
        target: list[int],
        predictions: list[int],
        average: str="macro"
        ) -> list[np.ndarray]:
    """Evaluate precision at multiple taxonomic ranks."""
    p = []
    for mapping in mappings:
        labels = list(set(mapping.values()))
        normalized_target = [mapping[int(v)] for v in list(target)]
        normalized_pred = [mapping[int(v)] for v in list(predictions)]
        p.append(
            precision_score(
                normalized_target,
                normalized_pred,
                average=average,
                labels=labels,
                zero_division=0.0
            )
        )
    return p


def evaluate_precision(
        classifier,
        loader,
        device,
        mapping,
        average: str="macro"
        ) -> list[float]:
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
    ranks =  rank_based_precision(mappings, real_y, predicted_y, average)
    collapsed = []
    for rank in ranks[::-1]:
        collapsed.append(np.mean(rank))
    return collapsed


def estimate_precision(classifier, loader, device, mapping, time_limit = 30):
    """Evaluate the precision score at different taxonomic levels."""
    a = time()
    n = 0
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
            n += len(y_batch)
            if time() - a > time_limit:
                print(f"Halting evaluation after {n} data points.")
                break
    ranks =  rank_based_precision(mappings, real_y, predicted_y)
    collapsed = []
    for rank in ranks[::-1]:
        collapsed.append(np.mean(rank))
    return collapsed


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


def span_mask(x, mask_prob=0.15, span_len=5):
    B, L = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    for b in range(B):
        num_spans = int(L * mask_prob / span_len)
        for _ in range(num_spans):
            start = torch.randint(0, L - span_len + 1, (1,)).item()
            mask[b, start:start + span_len] = True
    return mask


def mask_tokens(
        x,
        mask_id,
        vocab_size,
        mlm_probability: float = 0.15,
        mask_fraction: float = 0.8,
        order: str = "mlm"
        ):
    """Prepare masked input and MLM labels for MLM pretraining.

    Args:
        x: [B, L] input tensor.
        mask_id: Token used for masking.
        vocab_size: Vocabulary size excluding the masking token.
        mlm_probability: Fraction of input tokens to be manipulated for MLM.
        mask_fraction: Fraction of tokens manipulated by the `mlm_prabability`
            parameter that will be masked. Non-masked tokens will be randomly
            modified (50 %) or left unchanged (50 %).
        order: Way to mask the tokens. If `mlm`, will select random
            tokens as in BERT training. If `causal`, will select the last
            tokens of the sequence.

    Returns:
      masked_x: [B, L] (masked input)
      mlm_labels: [B, L] (original ids, or -1 for unmasked positions)
    """
    MASKING_LABEL = -1
    labels = x.clone()
    if order == "mlm":
        probability_matrix = torch.full(labels.shape, mlm_probability, device=x.device)
        mask = torch.bernoulli(probability_matrix).bool()
        mlm_labels = labels.clone()
        mlm_labels[~mask] = MASKING_LABEL
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, mask_fraction, device=x.device)
        ).bool() & mask
        modified_x = x.clone()
        modified_x[indices_replaced] = mask_id
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=x.device)
        ).bool() & mask & ~indices_replaced
        random_tokens = torch.randint(
            0, vocab_size, labels.shape, dtype=torch.long, device=x.device
        )
        modified_x[indices_random] = random_tokens[indices_random]
    elif order == "causal":
        mask = torch.zeros(labels.shape, dtype=torch.bool, device=x.device)
        L = labels.shape[1]
        n_last = int(L * mlm_probability)
        mask[:, -n_last:] = True
        mlm_labels = labels.clone()
        mlm_labels[~mask] = MASKING_LABEL
        indices_replaced = torch.ones(labels.shape, device=x.device).bool() & mask
        modified_x = x.clone()
        modified_x[indices_replaced] = mask_id
    elif order == "anti-causal":
        mask = torch.zeros(labels.shape, dtype=torch.bool, device=x.device)
        L = labels.shape[1]
        n_last = int(L * mlm_probability)
        mask[:, :n_last] = True
        mlm_labels = labels.clone()
        mlm_labels[~mask] = MASKING_LABEL
        indices_replaced = torch.ones(labels.shape, device=x.device).bool() & mask
        modified_x = x.clone()
        modified_x[indices_replaced] = mask_id
    return modified_x, mlm_labels


class Classifier(BaseClassifier):
    """A neural network-based DNA read classification model."""
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
        self.model = model.to(device)
        self.device = device
        self.mapping = mapping
        self.formatter = formatter
        self.clip = clip

    def get_parameters(self):
        """Get the parameters of the neural network."""
        return self.model.parameters()

    def predict(self, x_batch, probabilities: bool = False) -> list[int]:
        """Predict classes based on reads.

        Args:
            x_batch: Sequence batch of dimension [B, L].
            probabilities: If False, return the most likely class.

        Returns: B predicted classes or a vector of [B, C] probabilities.
        """
        self.model.eval()
        if self.formatter:
            x_batch = self.formatter(x_batch)
        output = self.model(x_batch)
        if type(output) is tuple:
            output = output[0]
        if not probabilities:
            return argmax(output, dim=1)
        else:
            return output

    def _compute_loss(
            self,
            x_batch,
            y_batch,
            loss_function_type,
            loss_function
            ) -> float:
        x_batch = x_batch.to(self.device).long()
        if self.formatter:
            x_batch = self.formatter(x_batch)
        if loss_function_type == "supervised":
            output = self.model(x_batch)
            y_batch = y_batch.to(self.device).long()
            loss = loss_function(output, y_batch)
        elif loss_function_type =="unsupervised":
            output = self.model(x_batch)
            loss = loss_function(output, x_batch)
        elif loss_function_type =="semi-supervised":
            output = self.model(x_batch)
            y_batch = y_batch.to(self.device).long()
            loss = loss_function(output, x_batch, y_batch)
        return loss

    def pretrain(
            self,
            train_loader: DataLoader,
            optimizer,
            max_batches: int,
            vocab_size: int,
            evaluation_interval: int = 500,
            patience: int = 2,
            probability: float = 0.15,
            pretraining_type: str = "mlm",
            ) -> None:
        """Pretrain a neural network with masked language modelling (MLM).

        Args:
            train_loader: Training data.
            optimizer: Model optimizer.
            max_n_batches: Maximum number of optimizing steps.
            vocab_size: Vocabulary size EXCLUDING the masking token.
            evaluation interval: Number of steps between evaluations.
            patience: Maximum increasing loss number before early stop.
            probability: Fraction of masked tokens.
            pretraining_type: Type of pretraining (mlm or clm)
        """
        num_batches, evaluation_n_batches = 0, 0
        lowest_loss = float("inf")
        total_loss = 0.0
        n_identicals, denominator = 0, 0
        entropies = 0.0
        avg_losses = []
        for epoch in range(1000):
            self.model.train()
            for x_batch in tqdm(train_loader):
                x_batch = x_batch.long().to(self.device)
                x_batch = self.formatter(x_batch)
                masked_x, mlm_labels = mask_tokens(
                    x_batch.clone(),
                    vocab_size,
                    vocab_size=256,
                    mlm_probability=probability,
                    order=pretraining_type
                )
                logits_mlm = self.model(masked_x, mlm_labels)
                loss = torch.nn.functional.cross_entropy(
                    logits_mlm.view(-1, vocab_size + 1),
                    mlm_labels.view(-1),
                    ignore_index=-1,
                )
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches += 1
                evaluation_n_batches += 1
                with torch.no_grad():
                    probs = torch.softmax(logits_mlm, dim=-1)
                    entropies += -torch.sum(probs * probs.log(), dim=-1).mean()
                    predictions = argmax(logits_mlm, dim=-1)
                    num_identical = (predictions == mlm_labels).sum().item()
                    n_identicals += num_identical
                    denominator += (mlm_labels != -1).sum().item()
                if num_batches % evaluation_interval == 0 or num_batches == 1:
                    avg_loss = total_loss / evaluation_n_batches
                    if lowest_loss < avg_loss:
                        patience -= 1
                    else:
                        lowest_loss = avg_loss
                    avg_losses.append(avg_loss)
                    entropy = entropies.item() / evaluation_n_batches
                    n = n_identicals
                    d = denominator
                    print(f"Step: {num_batches}. Epoch: {epoch+1}. "
                        + f"Loss: {avg_loss:.4f}. Entropy: {entropy:.5}. "
                        + f"Patience: {patience}. Correct pred.: {100.0 * n / d:.5} %")
                    n_identicals, denominator, entropies = 0, 0, 0
                    if patience <= 0:
                        print("Overfitting; stopping early.")
                        return avg_losses
                    total_loss = 0.0
                    evaluation_n_batches = 0
                if num_batches > max_batches:
                    print("Performed enough steps.")
                    return avg_losses
        return avg_losses

    def _get_validation_loss(
            self,
            validation_data,
            loss_function_type,
            loss_function,
            time_limit: float = 20,
            ) -> float:
        validation_loss = 0
        i = 0
        a = time()
        n = 0
        with no_grad():
            for x_batch, y_batch in validation_data:
                n += len(y_batch)
                loss = self._compute_loss(
                    x_batch, y_batch, loss_function_type, loss_function
                )
                validation_loss += loss.item()
                i += 1
                if time() - a > time_limit:
                    print(f"Halting evaluation after {n} data points.")
                    break
        validation_loss /= i
        validation_loss *= self.length
        return validation_loss

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            ) -> None:
        """Train a model.

        Args:
            train_loader: Training data.
            validate_loader: Validation data.
            optimizer: Model optimizer.
            max_n_epochs: Maximum number of epochs.
            patience: Maximum amount of times that the validation loss is
                allowed to increase before early stop.
            loss_function: Loss function.
            loss_function_type: Type of training (supervised, unsupervised,
                semi-supervised).
            evaluation_interval: Number of steps between evaluations.
            evaluation_maximum_duration: Maximum number of seconds to perform
                an evaluation.
            patience_interval: Number of steps to modify the patience.
        """
        TRAINING_TYPES = ("supervised", "unsupervised", "semi-supervised")
        assert loss_function_type in TRAINING_TYPES, "Invalid training type."
        training_losses = []
        validation_losses = []
        average_f_scores = []
        average_p_scores = []
        n_steps = 0
        current_loss = 0
        for epoch in range(max_n_epochs):
            self.model.train()
            for x_batch, y_batch in tqdm(train_loader):
                loss = self._compute_loss(
                    x_batch, y_batch, loss_function_type, loss_function
                )
                optimizer.zero_grad()
                current_loss += loss.item()
                loss.backward()
                if self.clip:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                # Infrequent evaluations; check real performance.
                msg = ""
                if n_steps and (
                        n_steps % evaluation_interval == 0
                        or n_steps >= n_max_steps
                    ):
                    f1 = evaluate(
                        self,
                        validate_loader,
                        self.device,
                        self.mapping,
                        time_limit=evaluation_maximum_duration
                    )
                    f1 = [float(f) for f in f1]
                    average_f_scores.append(f1)
                    f1_msg = [float(f"{f:.5}") for f in f1]
                    ps = estimate_precision(
                        self,
                        validate_loader,
                        self.device,
                        self.mapping,
                        time_limit=evaluation_maximum_duration
                    )
                    ps = [float(p) for p in ps]
                    average_p_scores.append(ps)
                    p_msg = [float(f"{p:.5}") for p in ps]
                    msg += f"{epoch+1}/{max_n_epochs} "
                    msg += f"F1: {f1_msg}. "
                    msg += f"Precision: {p_msg} "
                    self.model.train()
                # Frequent evaluations; report losses.
                if n_steps and n_steps % patience_interval == 0:
                    training_losses.append(current_loss * self.length / patience_interval)
                    current_loss = 0
                    validation_losses.append(
                        self._get_validation_loss(
                            validate_loader,
                            loss_function_type,
                            loss_function,
                            time_limit=evaluation_maximum_duration
                        )
                    )
                    if len(validation_losses) > 1:
                        best_loss = min(validation_losses[:-1])
                        if validation_losses[-1] > best_loss:
                            patience -= 1
                    if msg:
                        msg += "\n"
                    msg += f"Training loss: {training_losses[-1]:.5f}. "
                    msg += f"Validation loss: {validation_losses[-1]:.5f}. "
                    msg += f"Patience: {patience}"
                    if patience <= 0 or n_steps >= n_max_steps:
                        print(msg)
                        if n_steps >= n_max_steps:
                            print("Reached the maximum number of steps.")
                        else:
                            print("Stopping early.")
                        f = list(np.array(average_f_scores).T)
                        p = list(np.array(average_p_scores).T)
                        return training_losses, validation_losses, f, p
                if msg:
                    print(msg)
                n_steps += 1
        print(
            f"Training loss: {training_losses[-1]:.5f}. "
            f"Validation loss: {validation_losses[-1]:.5f}. "
            f"Patience: {patience}"
        )
        print("Maximum number of epochs reached; stopping early.")
        f = list(np.array(average_f_scores).T)
        p = list(np.array(average_p_scores).T)
        return training_losses, validation_losses, f, p
