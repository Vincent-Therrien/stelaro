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
from torch import save, load, argmax, tensor, ones, zeros
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

import matplotlib.pyplot as plt

from . import (
    BaseClassifier,
    Classifier,
    SyntheticTetramerDataset,
    SyntheticMultiLevelTetramerDataset,
    CustomTetramerDataset,
    confusion_matrix,
    evaluate_precision
)


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
            predictions[selection] = softmax(self.models[selection].predict(x_batch.long().to(self.device), True), dim=1)
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
            logits.append(sum(hierarchical_predictions[h]).detach().cpu().numpy())
        logits = tensor(np.array(logits))
        return argmax(logits, dim=0)

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
            epochs: int = 10
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
                    batch_size,
                    epochs
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
            batch_size: int = 128,
            epochs: int = 10
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
            max_n_epochs=epochs,
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


class BottomUpClassifier(BaseClassifier):
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
        self.models = {}

    def get_parameters(self):
        return None

    def predict(self, x_batch) -> list[int]:
        predictions = []
        for indices, model in self.models.items():
            prediction = softmax(model.predict(x_batch.long().to(self.device), True), dim=1)
            if indices:
                normalized_prediction = predictions[0].detach().clone()
                j = 0
                for i in indices:
                    normalized_prediction[:, i] = prediction[:, j]
                    j += 1
                predictions.append(normalized_prediction)
            else:
                predictions.append(prediction)
        logits = predictions[0].to("cpu")
        for p in predictions[1:]:
            logits += p.to("cpu")
        return argmax(logits, dim=1)

    def train(self,
            train_data_path: str,
            validate_data_path: str,
            patience: int = 3,
            max_n_epochs: int = 10,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            batch_size: int = 128,
            n_max_models: int = 3
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
        self.models[()] = self._add_model(
            list(range(len(self.mapping))),
            train_data_path,
            validate_data_path,
            patience,
            "supervised",
            evaluation_interval,
            evaluation_maximum_duration,
            patience_interval,
            n_max_steps,
            batch_size,
            max_n_epochs
        )
        for _ in range(n_max_models):
            indices = self._plan_new_model(validate_data_path, batch_size)
            self.models[tuple(indices)] = self._add_model(
                indices,
                train_data_path,
                validate_data_path,
                patience,
                "supervised",
                evaluation_interval,
                evaluation_maximum_duration,
                patience_interval,
                n_max_steps,
                batch_size,
                max_n_epochs
            )

    def _plan_new_model(self, validate_data_path, batch_size) -> list[int]:
        validate_loader = DataLoader(
            SyntheticTetramerDataset(
                validate_data_path,
            ),
            batch_size=batch_size,
            shuffle=True
        )
        confusion = confusion_matrix(self, validate_loader, self.device, self.mapping)
        true_class_counts = confusion.sum(axis=1)
        normalized = confusion / true_class_counts[:, np.newaxis]
        for i in range(len(normalized)):
            normalized[i][i] = 0
        plt.matshow(normalized)
        plt.show()
        print(evaluate_precision(self, validate_loader, self.device, self.mapping))
        # TMP
        phyla = {}
        for index, name in self.mapping.items():
            phylum = tuple(name[:-1])
            if phylum in phyla:
                phyla[phylum].append(int(index))
            else:
                phyla[phylum] = [int(index)]
        if () in phyla:
            del phyla[()]

        def get_indices(target_taxon) -> tuple[int]:
            indices = []
            for i, m in self.mapping.items():
                phylum = tuple(m[:-1])
                if phylum == target_taxon:
                    indices.append(int(i))
            return tuple(indices)

        worst = float("-inf")
        for name, phylum_indices in phyla.items():
            phylum_loss = normalized[phylum_indices].sum() / len(phylum_indices)
            indices = get_indices(name)
            if phylum_loss > worst and indices not in self.models:
                target_taxon = name
                worst = phylum_loss
                target_indices = indices

        print(f"Phylum {target_taxon}. Planned {target_indices}")
        return target_indices

    def _add_model(
            self,
            indices: list[int],
            train_data_path: str,
            validate_data_path: str,
            patience: int,
            loss_function_type: str,
            evaluation_interval: int = 5000,
            evaluation_maximum_duration: float = 30,
            patience_interval: int = 1000,
            n_max_steps: int = 100_000,
            batch_size: int = 128,
            epochs: int = 10
            ):
        if len(indices) == len(self.mapping):
            train_loader = DataLoader(
                SyntheticTetramerDataset(train_data_path),
                batch_size=batch_size,
                shuffle=True
            )
            validate_loader = DataLoader(
                SyntheticTetramerDataset(validate_data_path),
                batch_size=batch_size,
                shuffle=True
            )
            new_mapping = self.mapping
        else:
            train_loader = DataLoader(
                CustomTetramerDataset(
                    train_data_path,
                    self.mapping,
                    indices,
                    other_factor = 2.0
                ),
                batch_size=batch_size,
                shuffle=True
            )
            validate_loader = DataLoader(
                CustomTetramerDataset(
                    validate_data_path,
                    self.mapping,
                    indices,
                    other_factor = 2.0
                ),
                batch_size=batch_size,
                shuffle=True
            )
            new_mapping = validate_loader.dataset.mapping
        new_model = Classifier(
            self.length,
            new_mapping,
            self.device,
            self.neural_network,
            self.formatter,
            True
        )
        n_expected_steps = len(train_loader)
        if n_expected_steps < 2000:
            patience = 1
        new_model.train(
            train_loader,
            validate_loader,
            Adam(new_model.get_parameters(), lr=0.001),
            max_n_epochs=epochs,
            patience=patience,
            loss_function=CrossEntropyLoss(),
            loss_function_type=loss_function_type,
            evaluation_interval=evaluation_interval,
            evaluation_maximum_duration=evaluation_maximum_duration,
            patience_interval=patience_interval,
            n_max_steps=n_max_steps
        )
        return new_model

    def save(self, dir: str) -> None:
        model_index = {}
        i = 0
        for indices, model in self.models.items():
            model_index[str(i)] = {
                "indices": indices,
                "mapping": {str(v): k for k, v in model.mapping.items()}
            }
            save(model.model.state_dict(), f"{dir}{i}.pt2")
            i += 1
        with open(f"{dir}mapping.json", "w") as file:
            json.dump(model_index, file, indent=4)

    def load(self, dir: str) -> None:
        with open(f"{dir}mapping.json", "r") as file:
            model_index = json.load(file)
        self.models = {}
        for i, model_information in model_index.items():
            indices = tuple(model_information["indices"])
            mapping = model_information["mapping"]
            mapping = {v: k for k, v in mapping.items()}
            self.models[indices] = Classifier(
                self.length,
                mapping,
                self.device,
                self.neural_network,
                self.formatter,
                True
            )
            self.models[indices].model.load_state_dict(load(f"{dir}{i}.pt2"))
