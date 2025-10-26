"""Classical classification models.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2025
    - License: MIT
"""

from typing import Callable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

from . import BaseClassifier


class Classifier(BaseClassifier):
    """A classical DNA read classification model."""
    def __init__(
            self,
            length: int,
            mapping: dict,
            model: any,
            formatter: Callable,
            ):
        """
        Args:
            length: Length of the sequences processed by the model.
            mapping: Dictionary mapping a taxon to an index.
            model: Classiciation model supporting `fit` and `predict` methods.
            formatter: Function that converts the default input (tetramers)
                into another format. If `None`, tetramers are used as input.
        """
        self.length = length
        self.model = model
        self.mapping = mapping
        self.formatter = formatter

    def get_parameters(self):
        return None

    def predict(self, x_batch) -> list[int]:
        """Predict classes based on reads.

        Args:
            x_batch: Sequence batch of dimension [B, L].

        Returns: B predicted classes.
        """
        if self.formatter:
            x_batch = self.formatter(x_batch)
        return self.model.predict(x_batch)

    def _sample(data: DataLoader, limit: int) -> tuple:
        """Sample data into Numpy arrays."""
        x, y = [], []
        i = 0
        for x_batch, y_batch in tqdm(data):
            for xi, yi in zip(x_batch, y_batch):
                x.append(xi)
                y.append(yi)
                i += 1
                if i >= limit:
                    break
            if i >= limit:
                break
        x = np.array(x)
        y = np.array(y)
        return x, y

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            patience: int,
            n_samples_per_step: int = 1000,
            n_max_steps: int = 100_000,
            ) -> None:
        """Train a model.

        Args:
            train_loader: Training data.
            validate_loader: Validation data.
            patience: Maximum amount of times that the validation loss is
                allowed to increase before early stop.
            n_samples_per_step: Number of samples per steps.
            n_max_steps: Maximum number of steps.
        """
        raise NotImplementedError
