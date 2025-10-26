"""Emsemble-based classification models.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2025
    - License: MIT
"""

from typing import Callable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

from . import BaseClassifier, Classifier, SyntheticMultiLevelTetramerDataset


class HierarchicalClassifier(BaseClassifier):
    """A collection of neural networks classifying reads."""
    def __init__(
            self,
            length: int,
            mapping: dict,
            device: str,
            model_builder: any,
            formatter: Callable,
            clip: bool = True
            ):
        """
        Args:
            length: Length of the sequences processed by the model.
            mapping: Dictionary mapping a taxon to an index.
            device: Device onto which run computations.
            model_builder: Neural network (e.g. PyTorch Module).
            formatter: Function that converts the default input (tetramers)
                into another format. If `None`, tetramers are used as input.
            clip: If `True`, clip gradients to 1.0.
        """
        self.length = length
        self.model_builder = model_builder
        self.device = device
        self.mapping = mapping
        self.formatter = formatter
        self.n_levels = len(self.mapping['0'])
        self.clip = clip

    def train(self,
            train_data_path: str,
            validate_data_path: str,
            max_n_epochs: int,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            ) -> None:
        """Train a hierarchical ensemble of models.

        Args:
            train_data_path: Training data.
            validate_data_path: Validation data.
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
        self.model_tree = {"root": {}}
        for m in self.mapping.values():
            path = ["root", ]
            for taxon in m:
                path.append(taxon)
                selection = self.model_tree
                for p in path:
                    if p not in selection:
                        if p == m[-1]:
                            selection[p] = None
                        else:
                            selection[p] = {}
                    selection = selection[p]


            # train_loader = SyntheticMultiLevelTetramerDataset(
            #     train_data_path,
            #     self.mapping,
            #     selection = (),
            #     resolution = 0,
            #     other_factor = 2.0
            # )
