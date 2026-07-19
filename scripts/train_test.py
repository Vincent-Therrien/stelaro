"""Train an test neural networks to classify

    Input:
      - `train_x`: filepath to a Numpy array file of dimension [N, L / 4]
        containing 4-mer tokenized training sequences, where N is the number of
        sequences and L is the number of nucleotides in each sequence.
      - `train_y`: filepath to a Numpy array file of dimension [N, ] containing
        integer labels assigning a taxon to each corresponding sequence in the
        `train_x` array.
      - `test_x`: Same as `train_x` but for testing.
      - `test_y`: Same as `train_y` but for testing.
      - `validate_x`: Same as `train_x` but for validation.
      - `validate_y`: Same as `train_y` but for validation.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2026
    - License: MIT
"""

import argparse
from datetime import datetime, timezone
from time import time
import numpy as np
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import (CrossEntropyLoss, Module, Embedding, Linear, Parameter,
                      TransformerEncoderLayer, TransformerEncoder,
                      LayerNorm, ModuleList, Dropout, Identity, BatchNorm1d,
                      Conv1d, ReLU, Sequential)
from torch import (argmax, save, load, tensor, no_grad, arange, cat, randn,
                   Tensor, full, bernoulli, randint)
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from math import sqrt
from tokenizers import Tokenizer


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat()} {msg}")


# Models ######################################################################
class BERTax(Module):
    def __init__(
            self,
            N: int,
            num_classes: int,
            embed_dim: int = 250,
            vocab_size: int = 64,
            n_head: int = 5,
            dropout: float = 0.05,
            n_layers: int = 6,
            ):
        super(BERTax, self).__init__()
        self.embed_dim = embed_dim
        self.token_embedding = Embedding(vocab_size + 1, embed_dim)
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # makes input/output shape [B, L, D]
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = Linear(embed_dim, vocab_size + 1)
        self.classifier = Linear(embed_dim, num_classes)
        self.cls_token = Parameter(randn(1, 1, embed_dim))
        self.position_embedding = Embedding(N + 1, embed_dim)

    def forward(self, x, mlm=None):
        batch_size, seq_len = x.size()
        token_embeddings = self.token_embedding(x) * sqrt(self.embed_dim)
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1) * sqrt(self.embed_dim)
        x = cat([cls_tokens, token_embeddings], dim=1)  # Shape: [B, L + 1, D]
        positions = arange(0, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        x = self.dropout(self.emb_layer_norm(x))
        h = self.transformer_encoder(x)  # [B, L+1, D]
        if mlm is not None:
            h = h[:, 1:, :]
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            h = h[:, 0, :]  # [B, D]
            logits = self.classifier(h)  # [B, num_classes]
            return logits


class MambaBlock(Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaClassifier(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 64,
        d_model: int = 250,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(MambaClassifier, self).__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)
        self.mlm_head = Linear(d_model, vocab_size + 1)

    def forward(self, x) -> Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class MambaClassifierBPE(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 8196,
        d_model: int = 250,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(MambaClassifierBPE, self).__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x) -> Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class MambaClassifierAttention(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 64,
        d_model: int = 250,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(MambaClassifier, self).__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.attn_projection = Linear(d_model, 1, bias=False)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x) -> Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # [B, L, d_model]
        h = self.norm(h)
        attn_scores = self.attn_projection(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(h * attn_weights, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class Mamba2Block(Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=64
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Mamba2Classifier(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 64,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(Mamba2Classifier, self).__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            Mamba2Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)
        self.mlm_head = Linear(d_model, vocab_size + 1)

    def forward(self, x) -> Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class MambaClassifierLarge(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 64,
        d_model: int = 250,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(MambaClassifierLarge, self).__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x) -> Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class DownsamplingCNN(Module):
    """Converts [B, 4, L] into [B, L_reduced, d_model]."""
    def __init__(
        self,
        in_channels=4,
        d_model=250,
        n_layers=2,
        kernel_size=9,
        downsample_factor=4,
    ):
        super().__init__()
        layers = []
        c = in_channels
        current_factor = 1
        for i in range(n_layers):
            stride = 2 if current_factor < downsample_factor else 1
            current_factor *= stride
            if type(kernel_size) is int:
                layers += [
                    Conv1d(c, d_model, kernel_size,
                           stride=stride, padding=kernel_size // 2),
                    ReLU(),
                    BatchNorm1d(d_model)
                ]
            else:
                layers += [
                    Conv1d(c, d_model, kernel_size[i],
                           stride=stride, padding=kernel_size[i] // 2),
                    ReLU(),
                    BatchNorm1d(d_model)
                ]
            c = d_model
        self.net = Sequential(*layers)

    def forward(self, x):
        h = self.net(x)  # [B, d_model, L_reduced]
        return h.transpose(1, 2)  # [B, L_reduced, d_model]


class MambaClassifierCNN(Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 100,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
    ):
        super(MambaClassifierCNN, self).__init__()
        self.embedding = DownsamplingCNN(d_model=d_model, kernel_size=9)
        self.layers = ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x) -> Tensor:
        batch_size, _ = x.size()
        n1 = x // 16
        n2 = (x // 4) % 4
        n3 = x % 4
        nucleotides = torch.stack([n1, n2, n3], dim=2).view(batch_size, -1)
        one_hot = F.one_hot(nucleotides, num_classes=4).float()
        x_onehot = one_hot.transpose(1, 2)  # [B, 4, 3 * L]
        h = self.embedding(x_onehot)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class AdjustableMambaBlock(Module):
    def __init__(self, d_model, d_state, d_conv, expand, m, M):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_min = m,
            dt_max=M,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class DualTimescaleMamba(Module):
    def __init__(self, d_model, d_state, fast_dt=(0.00001, 0.01), slow_dt=(0.001, 0.2)):
        super().__init__()
        self.in_proj = Linear(d_model, d_model * 2)
        self.ssm_fast = AdjustableMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            m=fast_dt[0],
            M=fast_dt[1],
        )
        self.ssm_slow = AdjustableMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            m=slow_dt[0],
            M=slow_dt[1],
        )
        self.gate = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, x):
        u, v = self.in_proj(x).chunk(2, dim=-1)
        y_fast = self.ssm_fast(u)
        y_slow = self.ssm_slow(u)
        g = torch.sigmoid(self.gate(v))
        y = g * y_fast + (1 - g) * y_slow
        return self.out_proj(y)


class MambaClassifierDual(Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 64,
        d_model: int = 250,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        fast_dt=(0.00001, 0.01),
        slow_dt=(0.001, 0.2),
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size + 1, d_model)
        self.layers = ModuleList([
            DualTimescaleMamba(d_model=d_model, d_state=d_state, fast_dt=fast_dt, slow_dt=slow_dt)
        ])
        for _ in range(n_layers - 2):
            self.layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.classifier = Linear(d_model, num_classes)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        pooled = h.mean(dim=1)  # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


# Training ####################################################################
def decode_trimer(encoded_sequence: list[int]) -> str:
    BITS_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    decoded_chars = []

    for val in encoded_sequence:
        if not (0 <= val <= 63):
            raise ValueError("Encoded trimer integers must be between 0 and 63.")

        # Extract the three nucleotides from right to left
        base3 = BITS_TO_BASE[val & 3]          # Get lowest 2 bits
        base2 = BITS_TO_BASE[(val >> 2) & 3]   # Shift 2 bits, get next 2 bits
        base1 = BITS_TO_BASE[(val >> 4) & 3]   # Shift 4 bits, get highest 2 bits

        decoded_chars.extend([base1, base2, base3])

    return "".join(decoded_chars)


class LabelledDataset(Dataset):
    def __init__(self, directory: str, formatting: str = None, bpe_path: str = None):
        assert directory.endswith('/')
        self.x = np.load(directory + "x.npy")
        self.y = np.load(directory + "y.npy")
        self.formatting = formatting
        if bpe_path:
            self.tokenizer = Tokenizer.from_file(bpe_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.formatting == "1mer":
            pass
        elif self.formatting == "bpe":
            x = decode_trimer(x)
            x = self.tokenizer.encode(x)
        y = self.y[idx]
        return tensor(x), tensor(y)


class PretrainingDataset(Dataset):
    def __init__(self, filepath: str):
        self.x = np.load(filepath)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        return tensor(x)


def load_training_data(train, test, validate, batch_size, formatting: str = None, bpe_path: str = None):
    train_data = DataLoader(LabelledDataset(train, formatting, bpe_path), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(LabelledDataset(test, formatting, bpe_path), batch_size=batch_size, shuffle=True)
    validate_data = DataLoader(LabelledDataset(validate, formatting, bpe_path), batch_size=batch_size, shuffle=True)
    log(f"Number of training batches: {len(train_data)}")
    log(f"Number of test batches: {len(test_data)}")
    log(f"Number of validation batches: {len(validate_data)}")
    return train_data, test_data, validate_data


def pretrain(classifier, pretrain_filepath, batch_size, device):
    optimizer = Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(PretrainingDataset(pretrain_filepath), batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader)
    log(f"Number of pretraining batches: {total_steps}")
    criterion = CrossEntropyLoss(ignore_index=-100)
    vocab_size = 64
    mask_token_id = vocab_size
    classifier.train()

    total_loss = 0.0
    running_entropy = 0.0
    entropy_element_count = 0

    for batch_idx, x in enumerate(dataloader):
        a = time()
        x = x.to(device).long()
        targets = x.clone()

        # 1. Generate Mask Matrix (15%)
        probability_matrix = full(x.shape, 0.15, device=device)
        masked_indices = bernoulli(probability_matrix).bool()
        targets[~masked_indices] = -100

        # 2. Apply 80/10/10 Rule
        indices_replaced = bernoulli(full(x.shape, 0.8, device=device)).bool() & masked_indices
        x[indices_replaced] = mask_token_id

        indices_random = bernoulli(full(x.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
        random_tokens = randint(0, vocab_size, x.shape, device=device)
        x[indices_random] = random_tokens[indices_random]

        # 3. Forward Pass
        optimizer.zero_grad()
        logits = classifier(x, mlm=True) # Shape: [B, L, vocab_size + 1]

        # 4. Calculate MLM Loss
        loss = criterion(logits.view(-1, vocab_size + 1), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 5. Track Prediction Entropy on Masked Tokens Only
        with no_grad():
            # Get probabilities for all items: [B * L, V]
            probs = F.softmax(logits.view(-1, vocab_size + 1), dim=-1)

            # Calculate element-wise entropy: -p * log(p)
            # Add small eps (1e-9) to avoid log(0)
            token_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

            # Filter out unmasked locations using the flattened targets matrix
            flat_targets = targets.view(-1)
            masked_token_mask = flat_targets != -100

            # Accumulate entropy calculations
            running_entropy += token_entropy[masked_token_mask].sum().item()
            entropy_element_count += masked_token_mask.sum().item()

        b = time()

        # 6. Periodic Reporting (Every 1000 steps)
        step = batch_idx + 1
        if step == 50:
            duration = b - a
            total = duration * total_steps
            log(f"Step duration: {duration:.3} s. Estimated total duration: {int(total):_} s.")
        if step % 1000 == 0:
            avg_loss = total_loss / step
            avg_entropy = running_entropy / max(1, entropy_element_count)
            log(f"Step {step}. Avg Loss: {avg_loss:.4f} | Masked Token Entropy: {avg_entropy:.4f}")


def train_for_one_epoch(model, train_data, optimizer, loss_function, device):
    model.train()
    total_steps = len(train_data)
    i = 0
    for x_batch, y_batch in train_data:
        start = time()
        x_batch = x_batch.to(device).long()
        y_batch = y_batch.to(device).long()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        end = time()
        # Tracking
        if i == 50:  # Wait until duration stabilizes.
            duration = end - start
            total = duration * total_steps
            log(f"Step duration: {duration:.3} s. Estimated total duration: {int(total):_} s.")
        elif i and i % 1000 == 0:
            log(f"Training step {i}.")
        i += 1


def train(
        model, directory: str, n_epochs: int, train_data, validation_data, n_classes, device, start_epoch
    ) -> None:
    parameters = model.parameters()
    total_params = sum(param.numel() for param in parameters)
    log(f"Number of parameters: {total_params:_}")
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()
    for epoch in range(n_epochs):
        actual_epoch = start_epoch + epoch
        log(f"Epoch {actual_epoch + 1}")
        train_for_one_epoch(model, train_data, optimizer, loss_function, device)
        log(f"Saving the model at the end of epoch {actual_epoch + 1}.")
        save(model.state_dict(), f"{directory}weights_{actual_epoch + 1}_epoch.pt2")
        log("Validating the model.")
        validate(
            model, validation_data, n_classes, device, loss_function
        )


# Evaluation ##################################################################
def validate(model, validation_data, n_classes, device, loss_function=None):
    model.eval()
    n = 0
    validation_loss = 0
    real_y, predicted_y = [], []
    i = 0
    total_steps = len(validation_data)
    with no_grad():
        for x_batch, y_batch in validation_data:
            start = time()
            n += len(y_batch)
            x_batch = x_batch.to(device).long()
            y_batch = y_batch.to(device).long()
            output = model(x_batch)
            if loss_function:
                loss = loss_function(output, y_batch)
                validation_loss += loss.item()
            predictions = argmax(output, dim=1)
            predicted_y += predictions.cpu()
            real_y += y_batch.cpu()
            end = time()
            # Tracking
            if i == 50:  # Wait until duration stabilizes.
                duration = end - start
                total = duration * total_steps
                log(f"Step duration: {duration:.3} s. Estimated total duration: {int(total):_} s.")
            elif i and i % 1000 == 0:
                log(f"Evaluation step {i}.")
            i += 1
    if loss_function:
        validation_loss /= n
        log(f"Average loss: {validation_loss:.5}")
    f1 = f1_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    log(f"F1 score: {f1:.5}")
    precision = precision_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    log(f"Precision: {precision:.5}")


# Arguments ###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model benchmarking."
    )
    parser.add_argument("--train", type=str, help="Filepath to training set")
    parser.add_argument("--test", type=str, help="Filepath to testing set")
    parser.add_argument("--validate", type=str, help="Filepath to validation set")
    parser.add_argument("--n_classes", type=int, help="Number of classes")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--directory", type=str, help="output directory (weights and results)")
    parser.add_argument("--device", type=str, help="Either `cpu` or `cuda`.")
    parser.add_argument("--model", type=str, help="Name of the model to train and test.")
    parser.add_argument(
        "--pretraining",
        type=str,
        default=None,
        help="Optional filepath to pretraining weights/data (default: None)"
    )
    parser.add_argument(
        "--start_weight_epoch",
        type=int,
        default=None,
        help="Optional epoch from which to lead weights (default: None)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Optional format. Default: 3mer. Options: bpe"
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="Optional filepath to the BPE token file, if using BPE tokenization."
    )
    args = parser.parse_args()
    assert args.directory.endswith("/"), "`directory` must be a directory."

    log(f"Creating the model.")
    n_classes = int(args.n_classes)
    if args.model.lower() == "bertax":
        model = BERTax(500, n_classes)
    elif args.model.lower() == "mamba":
        model = MambaClassifier(n_classes)
    elif args.model.lower() == "mamba-cnn":
        model = MambaClassifierCNN(n_classes)
    elif args.model.lower() == "mamba-dual":
        model = MambaClassifierDual(n_classes)
    elif args.model.lower() == "mamba-large":
        model = MambaClassifierLarge(n_classes)
    elif args.model.lower() == "mamba-2":
        model = Mamba2Classifier(n_classes)
    elif args.model.lower() == "mamba-attention":
        model = MambaClassifierAttention(n_classes)
    elif args.model.lower() == "mamba-bpe":
        model = MambaClassifierBPE(n_classes)

    model = model.to(args.device)
    if args.start_weight_epoch:
        start_epoch = args.start_weight_epoch
        log(f"Loading the model at epoch {start_epoch}.")
        model.load_state_dict(load(f"{args.directory}weights_{start_epoch}_epoch.pt2"))
    else:
        start_epoch = 0

    if args.pretraining:
        log("Pretrain the model.")
        pretrain(model, args.pretraining, args.batch_size, args.device)
        log(f"Saving the model at the end of pretraining.")
        save(model.state_dict(), f"{args.directory}weights_pretraining_epoch.pt2")

    log("Load training data.")
    train_data, test_data, validate_data = load_training_data(args.train, args.test, args.validate, args.batch_size, args.format, args.bpe_path)
    if args.n_epochs > 0:
        log("Train the model.")
        train(model, args.directory, args.n_epochs, train_data, validate_data, args.n_classes, args.device, start_epoch)

    log("Test the trained model.")
    validate(model, test_data, args.n_classes, args.device)
