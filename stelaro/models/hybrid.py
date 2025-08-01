"""
    Transformer/Autoencoder hybrid neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2025
    - License: MIT
"""

import numpy as np
from torch import (argmax, float32, randn_like, bernoulli, ones_like,
                   rand_like, clamp, tensor, arange)
from torch import sum as torch_sum
from torch import exp as torch_exp
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, MSELoss, MaxPool1d, ConvTranspose1d,
                      Sigmoid, LeakyReLU, Module, Linear, Embedding,
                      TransformerEncoderLayer, TransformerEncoder)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import evaluate, get_f1_by_category, confusion_matrix, BaseClassifier


class Classifier(BaseClassifier):
    """A non-variational autoencoder-based DNA read classifier."""
    def __init__(self, length, mapping, device, model):
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        x_batch = x_batch.long()
        logits, _ = self.model(x_batch)
        predictions = argmax(logits, dim=1).to("cpu")
        return predictions

    def train(
            self,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer,
            max_n_epochs: int,
            patience: int,
            ):
        reconstruction_criterion = MSELoss()
        classification_criterion = CrossEntropyLoss()
        classification_weight = 0.25
        losses = []
        average_f_scores = []
        best_f1 = 0.0
        for epoch in range(max_n_epochs):
            self.model.train()
            losses.append(0)
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.type(float32).to(self.device)
                # Swap channels and sequence.
                x_batch = x_batch.permute(0, 2, 1)
                y_batch = y_batch.long().to(self.device)
                classification, reconstruction = self.model(x_batch)
                reconstruction_loss = reconstruction_criterion(
                    reconstruction, x_batch
                )
                classification_loss = classification_criterion(
                    classification, y_batch
                )
                loss = (
                    classification_weight * classification_loss
                    + (1 - classification_weight) * reconstruction_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[-1] += loss.item()
            f1 = evaluate(self, validate_loader, self.device, self.mapping)
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
        reconstruction_criterion = MSELoss()
        classification_criterion = CrossEntropyLoss()
        classification_weight = 0.25
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = [0]
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.long().to(self.device)

            B, N = x_batch.shape
            shifts = tensor([6, 4, 2, 0], device=x_batch.device).view(1, 1, 4)
            x_expanded = x_batch.unsqueeze(-1)
            tokens = (x_expanded >> shifts) & 0b11
            tokens = tokens.view(B, N * 4)
            one_hot = F.one_hot(tokens, num_classes=4).float()
            x_batch_one_hot = one_hot.permute(0, 2, 1)

            # Swap channels and sequence.
            y_batch = y_batch.long().to(self.device)
            classification, reconstruction = self.model(x_batch)
            reconstruction_loss = reconstruction_criterion(
                reconstruction, x_batch_one_hot
            )
            classification_loss = classification_criterion(
                classification, y_batch
            )
            loss = (
                classification_weight * classification_loss
                + (1 - classification_weight) * reconstruction_loss
            )
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
        reconstruction_criterion = MSELoss()
        classification_criterion = CrossEntropyLoss()
        classification_weight = 0.25
        # penalty_matrix = create_penalty_matrix(mapping).to(device)
        losses = [0]
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        f1_scores = [1 for _ in range(len(self.mapping))]
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.long().to(self.device)

            # Swap channels and sequence.
            B, N = x_batch.shape
            shifts = tensor([6, 4, 2, 0], device=x_batch.device).view(1, 1, 4)
            x_expanded = x_batch.unsqueeze(-1)
            tokens = (x_expanded >> shifts) & 0b11
            tokens = tokens.view(B, N * 4)
            one_hot = F.one_hot(tokens, num_classes=4).float()
            x_batch_one_hot = one_hot.permute(0, 2, 1)

            # Swap channels and sequence.
            y_batch = y_batch.long().to(self.device)
            classification, reconstruction = self.model(x_batch)
            reconstruction_loss = reconstruction_criterion(
                reconstruction, x_batch_one_hot
            )
            classification_loss = classification_criterion(
                classification, y_batch
            )
            loss = (
                classification_weight * classification_loss
                + (1 - classification_weight) * reconstruction_loss
            )

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
        return losses[:-1], average_f_scores


class Unsqueeze(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def add_gaussian_noise(x, std=0.1):
    noise = randn_like(x) * std
    return x + noise


class T_DAE_hybrid_1(Module):
    def __init__(self, N, M):
        super(T_DAE_hybrid_1, self).__init__()
        embed_dim = 128
        self.token_embedding = Embedding(256, embed_dim)
        self.position_embedding = Embedding(N, embed_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True  # makes input/output shape [B, L, D]
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = Linear(embed_dim, M)
        self.decoder = Sequential(
            Linear(embed_dim, N),
            LeakyReLU(),
            Linear(N, N * 4),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)  # [B, L, D]
        x = self.transformer_encoder(x)  # [B, L, D]
        x = x.mean(dim=1)  # [B, D] - average pooling
        classification = self.classifier(x)
        if self.training:
            x = add_gaussian_noise(x, 0.1)
        reconstruction = self.decoder(x)
        return classification, reconstruction


class T_DAE_hybrid_2(Module):
    def __init__(self, N, M):
        super(T_DAE_hybrid_2, self).__init__()
        embed_dim = 256
        self.token_embedding = Embedding(256, embed_dim)
        self.position_embedding = Embedding(N, embed_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True  # makes input/output shape [B, L, D]
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = Linear(embed_dim, M)
        self.decoder = Sequential(
            Linear(embed_dim, N),
            LeakyReLU(),
            Linear(N, N * 4),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)  # [B, L, D]
        x = self.transformer_encoder(x)  # [B, L, D]
        x = x.mean(dim=1)  # [B, D] - average pooling
        classification = self.classifier(x)
        if self.training:
            x = add_gaussian_noise(x, 0.1)
        reconstruction = self.decoder(x)
        return classification, reconstruction
