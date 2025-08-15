"""
    Transformer neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2025
    - License: MIT
"""

import numpy as np
from torch import (argmax, float32, randn_like, bernoulli, ones_like,
                   rand_like, clamp)
from torch import sum as torch_sum
from torch import exp as torch_exp, randn
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, MSELoss, MaxPool1d, ConvTranspose1d,
                      Sigmoid, LeakyReLU, functional, Parameter,
                      TransformerEncoderLayer, TransformerEncoder, Dropout)
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import evaluate, get_f1_by_category, confusion_matrix, BaseClassifier


class Classifier(BaseClassifier):
    """A DNA read classification dataset."""
    def __init__(self, length, mapping, device, model, use_tokens=False):
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping
        self.use_tokens = use_tokens

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        if self.use_tokens:
            x_batch = x_batch.long()
        predictions = argmax(self.model(x_batch), dim=1).to("cpu")
        return predictions

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            permute=False,
            ):
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = []
        average_f_scores = []
        best_f1 = 0.0
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            n_processed = 0
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.long().to(self.device)
                if permute:
                    x_batch = x_batch.permute(0, 2, 1)  # Swap channels and sequence.
                y_batch = y_batch.long().to(self.device)
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[-1] += loss.item()
                #n_processed += len(y_batch)
                #if n_processed > 50_000:
                #    break
            f1 = evaluate(self, validate_loader, self.device, self.mapping, permute=permute)
            f1 = [float(f) for f in f1]
            average_f_scores.append(f1)
            if f1[-1] > best_f1:
                best_f1 = f1[-1]
            if f1[-1] < best_f1:
                patience -= 1
            loss_msg = [float(f"{f:.5}") for f in f1]
            print(
                f"{epoch+1}/{max_n_epochs}",
                f"Loss: {losses[-1]:.2f}.",
                f"F1: {loss_msg}.",
                f"Patience: {patience}"
            )
            if patience <= 0:
                print("The model is overfitting; stopping early.")
                break
        average_f_scores = list(np.array(average_f_scores).T)
        return losses, average_f_scores

    def train_large_dataset(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            evaluation_interval: int,
            n_max_reads: int,
            patience: int,
            ):
        """Same as `train`, but this method assumes that the training set is
        infinite.
        """
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = [0]
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.long().to(self.device)
            # Swap channels and sequence.
            y_batch = y_batch.long().to(self.device)
            output = self.model(x_batch)
            loss = criterion(output, y_batch)
            # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[-1] += loss.item()
            n_reads_processed += len(y_batch)
            evaluation_countdown -= len(y_batch)
            if evaluation_countdown <= 0:
                evaluation_countdown = evaluation_interval
                f1 = evaluate(self, validate_loader, self.device, self.mapping, permute=False)
                f1 = [float(f) for f in f1]
                average_f_scores.append(f1)
                if f1[-1] > best_f1:
                    best_f1 = f1[-1]
                if f1[-1] < best_f1:
                    patience -= 1
                loss_msg = [float(f"{f:.5}") for f in f1]
                print(
                    f"N Reads: {n_reads_processed:_}",
                    f"Loss: {losses[-1]:.2f}.",
                    f"F1: {loss_msg}.",
                    f"Patience: {patience}"
                )
                losses.append(0)
                if patience <= 0:
                    print("The model is overfitting; stopping early.")
                    break
            if n_reads_processed > n_max_reads:
                print("Reached the specified maximum number of reads.")
                break
        else:
            print("Exhausted all reads.")

        print(f"Processed {n_reads_processed:_} reads.")
        average_f_scores = list(np.array(average_f_scores).T)
        return losses[:-1], average_f_scores

    def train_large_dataset_adaptive(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            evaluation_interval: int,
            n_max_reads: int,
            patience: int,
            ):
        """Same as `train`, but this method assumes that the training set is
        infinite.
        """
        criterion = CrossEntropyLoss()
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = [0]
        validation_losses = []
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        f1_scores = [1 for _ in range(len(self.mapping))]
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.long().to(self.device)
            # Swap channels and sequence.
            y_batch = y_batch.long().to(self.device)
            output = self.model(x_batch)
            loss = criterion(output, y_batch)
            # loss *= penalized_cross_entropy(output, y_batch, penalty_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[-1] += loss.item() / len(y_batch)
            n_reads_processed += len(y_batch)
            evaluation_countdown -= len(y_batch)
            if evaluation_countdown <= 0:
                evaluation_countdown = evaluation_interval
                f1 = evaluate(self, validate_loader, self.device, self.mapping, permute=False)
                f1 = [float(f) for f in f1]
                average_f_scores.append(f1)
                if f1[-1] > best_f1:
                    best_f1 = f1[-1]
                if f1[-1] < best_f1:
                    patience -= 1
                loss_msg = [float(f"{f:.5}") for f in f1]
                print(
                    f"N Reads: {n_reads_processed:_}",
                    f"Loss: {losses[-1]:.2f}.",
                    f"F1: {loss_msg}.",
                    f"Patience: {patience}"
                )
                losses.append(0)
                for x_batch, y_batch in validate_loader:
                    x_batch = x_batch.type(float32).to(self.device)
                    y_batch = y_batch.long().to(self.device)
                    output = self.model(x_batch)
                    loss = criterion(output, y_batch)
                    validation_losses[-1] += loss.item() / len(y_batch)
                if patience <= 0:
                    print("The model is overfitting; stopping early.")
                    break
                # Adjust data generation
                matrix = confusion_matrix(self, validate_loader, "cuda", self.mapping, False)
                f1_scores = get_f1_by_category(matrix)
                print(f"F1 scores: {f1_scores}")
                f1_scores = (np.ones(f1_scores.shape) / 2) + ((1 - f1_scores) / 2)
                frequencies = np.array(f1_scores).astype(float)
                frequencies /= frequencies.mean()
                frequencies /= len(frequencies)
                print(f"Setting distributions: {frequencies}")
                train_loader.dataset.set_distribution(frequencies)
            if n_reads_processed > n_max_reads:
                print("Reached the specified maximum number of reads.")
                break
        else:
            print("Exhausted all reads.")

        print(f"Processed {n_reads_processed:_} reads.")
        average_f_scores = list(np.array(average_f_scores).T)
        return losses[:-1], average_f_scores, validation_losses


class T_1(Module):
    def __init__(self, N, M):
        super().__init__()
        d_model = 128
        self.input_proj = Linear(4, d_model)
        self.pos_embedding = Parameter(randn(1, N, d_model))
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = Sequential(
            Linear(d_model, d_model),
            ReLU(),
            Dropout(0.1),
            Linear(d_model, M)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Swap channels and sequence.
        x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
