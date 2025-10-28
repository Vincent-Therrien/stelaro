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
from torch import save, load, argmax, tensor
from torch.nn import CrossEntropyLoss

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
        self.sub_mappings = {}

    def predict(self, x_batch) -> list[int]:
        predictions = {}
        for selection in self.models:
            predictions[selection] = self.models[selection].predict(x_batch.long().to(self.device), True)
        hierarchical_predictions = {}
        for taxon_id, lineage in self.mapping.items():
            hierarchical_predictions[taxon_id] = []
            for level in range(len(lineage)):
                taxonomic_level = tuple(lineage[:level])
                key = tuple(lineage[:level + 1])
                index = self.sub_mappings[taxonomic_level]
                for i in range(len(index)):
                    if index[str(i)] == key:
                        taxon_index = i
                value = predictions[taxonomic_level][:, taxon_index]
                hierarchical_predictions[taxon_id].append(value)
        logits = []
        for h in hierarchical_predictions:
            logits.append(sum(hierarchical_predictions[h]))
        return argmax(tensor(logits))

    def train(self,
            train_data_path: str,
            validate_data_path: str,
            patience: int,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            batch_size: int = 128,
            ) -> None:
        """Train a hierarchical ensemble of models.

        Args:
            train_data_path: Training data.
            validate_data_path: Validation data.
            optimizer: Model optimizer.
            patience: Maximum amount of times that the validation loss is
                allowed to increase before early stop.
            loss_function_type: Type of training (supervised, unsupervised,
                semi-supervised).
            evaluation_interval: Number of steps between evaluations.
            evaluation_maximum_duration: Maximum number of seconds to perform
                an evaluation.
            patience_interval: Number of steps to modify the patience.
            n_max_steps: Maximum number of steps for each model.
        """
        self.tree = {}
        self.n_models = 1
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
                            self.n_models += 1
                    selection = selection[p]
        print(f"Training {self.n_models} models.")

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
                    loss_function_type,
                    evaluation_interval,
                    evaluation_maximum_duration,
                    patience_interval,
                    n_max_steps,
                    batch_size
                )

    def _add_model(
            self,
            selection: tuple[str],
            train_data_path: str,
            validate_data_path: str,
            patience: int,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            batch_size: int = 128
            ):
        """
        """
        train_loader = DataLoader(
            SyntheticMultiLevelTetramerDataset(
                train_data_path,
                self.mapping,
                selection = selection,
                resolution = 1,
                other_factor = 2.0
            ),
            batch_size=batch_size,
            shuffle=True
        )
        validate_loader = DataLoader(
            SyntheticMultiLevelTetramerDataset(
                validate_data_path,
                self.mapping,
                selection = selection,
                resolution = 1,
                other_factor = 2.0
            ),
            batch_size=batch_size,
            shuffle=True
        )
        new_mapping = validate_loader.dataset.mapping
        self.sub_mappings[selection] = new_mapping
        print(f"{len(self.models) + 1}/{self.n_models} Training at level {selection} for classes {new_mapping}")
        self.models[selection] = Classifier(
            self.length,
            new_mapping,
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
            loss_function=CrossEntropyLoss(),
            loss_function_type=loss_function_type,
            evaluation_interval=evaluation_interval,
            evaluation_maximum_duration=evaluation_maximum_duration,
            patience_interval=patience_interval,
            n_max_steps=n_max_steps
        )

    def save(self, dir: str) -> None:
        mapping = {}
        for i, model_name in enumerate(self.models):
            mapping[str(i)] = {
                "model_name": list(model_name),
                "target_taxa": self.sub_mappings[model_name]
            }
            save(self.models[model_name].model.state_dict(), f"{dir}{i}.pt2")
        with open(f"{dir}mapping.json", "w") as file:
            json.dump(mapping, file, indent=4)

    def load(self, dir: str) -> None:
        with open(f"{dir}mapping.json", "r") as file:
            mapping = json.load(file)
        for i in mapping:
            selection = tuple(mapping[i]["model_name"])
            sub_mappings = {i: tuple(t) for i, t in mapping[i]["target_taxa"].items()}
            self.sub_mappings[selection] = sub_mappings
            self.models[selection] = Classifier(
                self.length,
                sub_mappings,
                self.device,
                self.neural_network,
                self.formatter,
                True
            )
            self.models[selection].model.load_state_dict(load(f"{dir}{i}.pt2"))
