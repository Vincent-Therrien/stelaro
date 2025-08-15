"""
    Autoencoder neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

import numpy as np
from torch import (argmax, float32, randn_like, bernoulli, ones_like,
                   rand_like, clamp, tensor)
from torch import sum as torch_sum
from torch import exp as torch_exp
from torch.nn import (Module, Conv1d, ReLU, Sequential, Flatten, Linear,
                      CrossEntropyLoss, MSELoss, MaxPool1d, ConvTranspose1d,
                      Sigmoid, LeakyReLU)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import evaluate, BaseClassifier


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
            permute: bool=False,
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
                if permute:
                    x_batch = x_batch.permute(0, 2, 1)
                y_batch = y_batch.long().to(self.device)
                classification, reconstruction = self.model(x_batch)

                B, N = x_batch.shape
                shifts = tensor([6, 4, 2, 0], device=x_batch.device).view(1, 1, 4)
                x_expanded = x_batch.unsqueeze(-1).to(int)
                tokens = (x_expanded >> shifts) & 0b11
                tokens = tokens.view(B, N * 4)
                one_hot = F.one_hot(tokens, num_classes=4).float()
                onehot_batch = one_hot.permute(0, 2, 1)

                reconstruction_loss = reconstruction_criterion(
                    reconstruction, onehot_batch
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
            f1 = evaluate(self, validate_loader, self.device, self.mapping, permute)
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
        validation_losses = []
        average_f_scores = []
        best_f1 = 0.0
        n_reads_processed = 0
        evaluation_countdown = evaluation_interval
        for x_batch, y_batch in tqdm(train_loader):
            self.model.train()
            x_batch = x_batch.type(float32).to(self.device)

            B, N = x_batch.shape
            shifts = tensor([6, 4, 2, 0], device=x_batch.device).view(1, 1, 4)
            x_expanded = x_batch.unsqueeze(-1).to(int)
            tokens = (x_expanded >> shifts) & 0b11
            tokens = tokens.view(B, N * 4)
            one_hot = F.one_hot(tokens, num_classes=4).float()
            x_batch = one_hot.permute(0, 2, 1)

            # Swap channels and sequence.
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
                validation_losses.append(0)
                for x_batch, y_batch in validate_loader:
                    x_batch = x_batch.type(float32).to(self.device)
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
                    validation_losses[-1] += loss.item() / len(y_batch)
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
        return losses[:-1], average_f_scores, validation_losses


class Unsqueeze(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AE_1(Module):
    """Autoencoder 1."""
    def __init__(self, N, M):
        super(AE_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Flatten(),
            LeakyReLU(),
            Linear(N * 4, N),
            LeakyReLU(),
            Linear(N, N // 2),
            LeakyReLU(),
            Linear(N // 2, LATENT_SPACE_SIZE)
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction


class CAE_1(Module):
    """Convolutional autoencoder 1."""
    def __init__(self, N, M):
        super(CAE_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Conv1d(4, 32, kernel_size=5, padding=2),
            LeakyReLU(),
            MaxPool1d(kernel_size=2),
            Conv1d(32, 64, kernel_size=3, padding=1),
            LeakyReLU(),
            Flatten(),
            Linear(N * 64 // 2, N // 2),
            LeakyReLU(),
            Linear(N // 2, LATENT_SPACE_SIZE),
            LeakyReLU(),
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction


class CAE_2(Module):
    """Convolutional autoencoder 2."""
    def __init__(self, N, M):
        super(CAE_2, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Conv1d(4, 32, kernel_size=7, padding=3),
            ReLU(),
            MaxPool1d(kernel_size=2),
            Conv1d(32, 64, kernel_size=5, padding=2),
            ReLU(),
            Conv1d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Flatten(),
            Linear(N * 128 // 2, N // 2),
            ReLU(),
            Linear(N // 2, N // 4),
            ReLU(),
            Linear(N // 4, LATENT_SPACE_SIZE),
            ReLU(),
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 4),
            ReLU(),
            Linear(N // 4, N // 2),
            ReLU(),
            Linear(N // 2, N),
            ReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction


def add_gaussian_noise(x, std=0.1):
    noise = randn_like(x) * std
    return x + noise


def add_masking_noise(x, dropout_prob=0.1):
    mask = bernoulli((1 - dropout_prob) * ones_like(x))
    return x * mask


def add_salt_and_pepper_noise(x, prob=0.1):
    rand = rand_like(x)
    x_noisy = x.clone()
    x_noisy[rand < (prob / 2)] = 0
    x_noisy[(rand >= (prob / 2)) & (rand < prob)] = 1
    return x_noisy


class DAE_1(Module):
    """Denoising autoencoder 1."""
    def __init__(self, N, M):
        super(DAE_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = Sequential(
            Conv1d(4, 32, kernel_size=5, padding=2),
            LeakyReLU(),
            MaxPool1d(kernel_size=2),
            Conv1d(32, 64, kernel_size=3, padding=1),
            LeakyReLU(),
            Flatten(),
            Linear(N * 64 // 2, N // 2),
            LeakyReLU(),
            Linear(N // 2, LATENT_SPACE_SIZE),
            LeakyReLU(),
        )
        self.decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )
        self.classifier = Linear(LATENT_SPACE_SIZE, M)

    def forward(self, x):
        if self.training:
            x_noisy = add_masking_noise(x)
            x_noisy = clamp(x_noisy, 0.0, 1.0)
            latent = self.encoder(x_noisy)
        else:
            latent = self.encoder(x)
        classification = self.classifier(latent)
        reconstruction = self.decoder(latent)
        return classification, reconstruction


class VE_1(Module):
    """Variational encoder 1."""
    def __init__(self, N, LATENT_SPACE_SIZE):
        super(VE_1, self).__init__()
        self.internal_encoder = Sequential(
            Conv1d(4, 32, kernel_size=5, padding=2),
            LeakyReLU(),
            MaxPool1d(kernel_size=2),
            Conv1d(32, 64, kernel_size=3, padding=1),
            LeakyReLU(),
            Flatten(),
            Linear(N * 64 // 2, N // 2),
            LeakyReLU(),
        )
        self.function_logits = Linear(N // 2, LATENT_SPACE_SIZE)
        self.function_mean = Linear(N // 2, LATENT_SPACE_SIZE)
        self.function_var = Linear(N // 2, LATENT_SPACE_SIZE)

    def forward(self, x):
        h = self.internal_encoder(x)
        logits = self.function_logits(h)
        mean = self.function_mean(h)
        log_var = self.function_var(h)
        return logits, mean, log_var


class VD_1(Module):
    """Variational decoder 1."""
    def __init__(self, N, LATENT_SPACE_SIZE):
        super(VD_1, self).__init__()
        self.internal_decoder = Sequential(
            Linear(LATENT_SPACE_SIZE, N // 2),
            LeakyReLU(),
            Linear(N // 2, N),
            LeakyReLU(),
            Unsqueeze(1),
            ConvTranspose1d(1, 4, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            Sigmoid(),
        )

    def forward(self, x):
        return self.internal_decoder(x)


class VAE_1(Module):
    """Variational autoencoder 1."""
    def __init__(self, N, M):
        super(VAE_1, self).__init__()
        LATENT_SPACE_SIZE = 128
        self.encoder = VE_1(N, LATENT_SPACE_SIZE)
        self.decoder = VD_1(N, LATENT_SPACE_SIZE)
        self.classifier = Linear(LATENT_SPACE_SIZE, M)
        self.device = "cuda"  # TODO: Parametrize
        self.n_classes = M

    def reparameterization(self, mean, var):
        epsilon = randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        logits, mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch_exp(0.5 * log_var))
        reconstruction = self.decoder(z)
        classification = self.classifier(logits)
        return classification, reconstruction, mean, log_var


def vae_loss_function(x, reconstructed_x, mean, log_var):
    fn = MSELoss()
    reproduction_loss = fn(reconstructed_x, x)
    KLD = -0.5 * torch_sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


class VariationalClassifier(BaseClassifier):
    """A variational autoencoder DNA read classifier."""
    def __init__(self, length, mapping, device, model):
        self.model = model(length, len(mapping)).to(device)
        self.device = device
        self.mapping = mapping

    def get_parameters(self):
        return self.model.parameters()

    def predict(self, x_batch):
        self.model.eval()
        logits, _, _, _ = self.model(x_batch)
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
        reconstruction_criterion = vae_loss_function
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
                y_batch = y_batch.type(float32).to(self.device).to(int)
                classification, x_hat, mean, log_var = self.model(x_batch)
                optimizer.zero_grad()
                reconstruction_loss = reconstruction_criterion(
                    x_batch, x_hat, mean, log_var
                )
                classification_loss = classification_criterion(
                    classification, y_batch
                )
                loss = (
                    classification_weight * classification_loss
                    + (1 - classification_weight) * reconstruction_loss
                )
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
