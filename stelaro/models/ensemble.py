"""Emsemble-based classification models.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2025
    - License: MIT
"""

import json
from typing import Callable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from torch import save, load

from . import BaseClassifier, Classifier, SyntheticMultiLevelTetramerDataset


class HierarchicalClassifier(BaseClassifier):
    """A collection of neural networks classifying reads."""
    def __init__(
            self,
            length: int,
            mapping: dict,
            device: str,
            neural_network: any,
            formatter: Callable,
            ):
        """
        Args:
            length: Length of the sequences processed by the model.
            mapping: Dictionary mapping a taxon to an index.
            device: Device onto which run computations.
            model_builder: Neural network (e.g. PyTorch Module).
            formatter: Function that converts the default input (tetramers)
                into another format. If `None`, tetramers are used as input.
        """
        self.length = length
        self.neural_network = neural_network
        self.device = device
        self.mapping = mapping
        self.formatter = formatter
        self.n_levels = len(self.mapping['0'])
        self.models = {}

    def train(self,
            train_data_path: str,
            validate_data_path: str,
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
            patience: Maximum amount of times that the validation loss is
                allowed to increase before early stop.
            loss_function: Loss function.
            loss_function_type: Type of training (supervised, unsupervised,
                semi-supervised).
            evaluation_interval: Number of steps between evaluations.
            evaluation_maximum_duration: Maximum number of seconds to perform
                an evaluation.
            patience_interval: Number of steps to modify the patience.
            n_max_steps: Maximum number of steps for each model.
        """
        self.tree = {}
        n_models = 1
        for m in self.mapping.values():
            path = []
            for taxon in m:
                path.append(taxon)
                selection = self.tree
                for p in path:
                    if p not in selection:
                        if p == m[-1]:
                            selection[p] = None
                        else:
                            selection[p] = {}
                            n_models += 1
                    selection = selection[p]
        print(f"Training {n_models} models.")

        selections = set()

        def recurse(subtree: dict, selection: list[str]) -> None:
            for taxon in subtree:
                selections.add(tuple(selection))
                if subtree[taxon]:
                    sub_selection = selection.copy()
                    sub_selection.append(taxon)
                    recurse(subtree[taxon], sub_selection)

        recurse(self.tree, [])
        for x in sorted(selections, key=lambda x: len(x)):
            self._add_model(
                x,
                train_data_path,
                validate_data_path,
                patience,
                loss_function,
                loss_function_type,
                evaluation_interval,
                evaluation_maximum_duration,
                patience_interval,
                n_max_steps
            )

    def _add_model(
            self,
            selection: tuple[str],
            train_data_path: str,
            validate_data_path: str,
            patience: int,
            loss_function: Callable,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            ):
        """
        """
        print(f"Training: {selection}")
        train_loader = SyntheticMultiLevelTetramerDataset(
            train_data_path,
            self.mapping,
            selection = selection,
            resolution = 1,
            other_factor = 2.0
        )
        validate_loader = SyntheticMultiLevelTetramerDataset(
            validate_data_path,
            self.mapping,
            selection = selection,
            resolution = 1,
            other_factor = 2.0
        )
        self.models[selection] = Classifier(
            self.length,
            validate_loader.mapping,
            self.device,
            self.neural_network,
            self.formatter,
            True
        )
        self.models[selection].train(
            train_loader,
            validate_loader,
            Adam(self.models[selection].get_parameters(), lr=0.001),
            max_n_epochs=10,
            patience=patience,
            loss_function=loss_function,
            loss_function_type=loss_function_type,
            evaluation_interval=evaluation_interval,
            evaluation_maximum_duration=evaluation_maximum_duration,
            patience_interval=patience_interval,
            n_max_steps=n_max_steps
        )

    def save(self, dir: str) -> None:
        mapping = {}
        for i, model in enumerate(self.models):
            mapping[str(i)] = list(model)
            save(self.models[model].state_dict(), f"{dir}{i}.pt2")
        with open(f"{dir}mapping.json", "w") as file:
            json.dump(mapping, file)

    def load(self, dir: str) -> None:
        with open(f"{dir}mapping.json", "r") as file:
            mapping = json.load(file)
        for i in mapping:
            with open(f"{dir}{i}.pt2", "r") as file:
                self.models[tuple(mapping[i])] = load(file)
