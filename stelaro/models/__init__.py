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
        mask_fraction: float = 1.0
        ):
    """Prepare masked input and MLM labels for MLM pretraining.

    Args:
        x: [B, L] input tensor.
        mask_id: Token used for masking.
        vocab_size: Vocabulary size excluding the masking token.
        mlm_probability: Fraction of input tokens to be manipulated for MLM.
        mask_fraction: Fraction of tokens manipultated by the `mlm_prabability`
            parameter that will be masked. Non-masked tokens will be randomly
            modified (50 %) or left unchanged (50 %).

    Returns:
      masked_x: [B, L] (masked input)
      mlm_labels: [B, L] (original ids, or -1 for unmasked positions)
    """
    labels = x.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=x.device)
    mask = torch.bernoulli(probability_matrix).bool()
    mlm_labels = labels.clone()
    MASKING_LABEL = -1
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
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping
        self.formatter = formatter
        self.clip = clip

    def get_parameters(self):
        """Get the parameters of the neural network."""
        return self.model.parameters()

    def predict(self, x_batch) -> list[int]:
        """Predict classes based on reads.

        Args:
            x_batch: Sequence batch of dimension [B, L].

        Returns: B predicted classes.
        """
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
        if loss_function_type == "supervised":
            output = self.model(x_batch)
            y_batch = y_batch.long().to(self.device)
            loss = loss_function(output, y_batch)
        elif loss_function_type =="unsupervised":
            output = self.model(x_batch)
            loss = loss_function(output, x_batch)
        elif loss_function_type =="semi-supervised":
            output = self.model(x_batch)
            y_batch = y_batch.long().to(self.device)
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
            mlm_probability: float = 0.15
            ) -> None:
        """Pretrain a neural network with masked language modelling (MLM).

        Args:
            train_loader: Training data.
            optimizer: Model optimizer.
            max_n_batches: Maximum number of optimizing steps.
            vocab_size: Vocabulary size EXCLUDING the masking token.
            evaluation interval: Number of steps between evaluations.
            patience: Maximum increasing loss number before early stop.
            mlm_probability: Fraction of masked tokens.
        """
        num_batches, evaluation_n_batches = 0, 0
        lowest_loss = float("inf")
        total_loss = 0.0
        n_identicals, denominator = 0, 0
        entropies = 0.0
        for epoch in range(1000):
            self.model.train()
            for x_batch in tqdm(train_loader):
                x_batch = x_batch.long().to(self.device)
                x_batch = self.formatter(x_batch)
                masked_x, mlm_labels = mask_tokens(
                    x_batch.clone(),
                    vocab_size,
                    vocab_size=256,
                    mlm_probability=mlm_probability
                )
                logits_mlm = self.model(masked_x, mlm_labels)
                loss = torch.nn.functional.cross_entropy(
                    logits_mlm.view(-1, vocab_size + 1),
                    mlm_labels.view(-1),
                    ignore_index=-1,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                evaluation_n_batches += 1
                with torch.no_grad():
                    probs = torch.softmax(logits_mlm, dim=-1)
                    entropies += -torch.sum(probs * probs.log(), dim=-1).mean()
                    predictions = argmax(logits_mlm, dim=-1)
                    num_identical = (predictions == mlm_labels).sum().item()
                    n_identicals += num_identical
                    denominator += (mlm_labels != -1).sum().item()
                if num_batches % evaluation_interval == 0:
                    avg_loss = total_loss / evaluation_n_batches
                    print(f"Step: {num_batches}. Epoch: {epoch+1}. MLM loss: {avg_loss:.4f}. Patience: {patience}")
                    entropy = entropies.item() / evaluation_n_batches
                    print(f"Average entropy: {entropy:.5}. ", end="")
                    n = n_identicals
                    d = denominator
                    print(f"Correct predictions: {n} / {d} ({100.0 * n / d:.5} %).")
                    n_identicals, denominator, entropies = 0, 0, 0
                    if lowest_loss < avg_loss:
                        patience -= 1
                        if patience <= 0:
                            print("Overfitting; stopping early.")
                            return
                    else:
                        lowest_loss = avg_loss
                    total_loss = 0.0
                    evaluation_n_batches = 0
                if num_batches > max_batches:
                    print("Performed enough steps.")
                    return

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 2000,
            evaluation_maximum_duration: float = 30,
            ) -> None:
        """Train a model.

        Args:
            train_loader: Training data.
            validate_loader: Validation data.
            optimizer: Model optimizer.
            max_n_epochs: Maximum number of epochs.
            patience: Maximum increasing loss number before early stop.
            loss_function: Loss function.
            loss_function_type: Type of training (supervised, unsupervised,
                semi-supervised).
            evaluation_interval: Number of steps between evaluations.
            evaluation_maximum_duration: Maximum number of seconds to perform
                an evaluation.
        """
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
                    ps = estimate_precision(
                        self,
                        validate_loader,
                        self.device,
                        self.mapping
                    )
                    ps = [float(p) for p in ps]
                    p_msg = [float(f"{p:.5}") for p in ps]
                    print(f"P: {p_msg}")
                    self.model.train()
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
            f1 = evaluate(
                self,
                validate_loader,
                self.device,
                self.mapping,
                time_limit=evaluation_maximum_duration
            )
            f1 = [float(f) for f in f1]
            f1_msg = [float(f"{f:.5}") for f in f1]
            average_f_scores.append(f1)
            if f1[-1] > best_f1:
                best_f1 = f1[-1]
            if f1[-1] < best_f1:
                patience -= 1
            ps = estimate_precision(
                self,
                validate_loader,
                self.device,
                self.mapping,
                time_limit=evaluation_maximum_duration
            )
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
