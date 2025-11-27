"""
    SSM-based models.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2025
    - License: MIT
"""

import torch
from mamba_ssm import Mamba
from torch import nn


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaSequenceClassifier(nn.Module):
    """A sequence classifier network that supports MLM."""
    def __init__(
        self,
        N: int,
        num_classes: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        pooling = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)
        self.pooling = pooling
        self.mlm_head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: torch.LongTensor, mlm=None) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)
        if mlm is not None:
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            pooled = h.mean(dim=1)  # [B, d_model]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)  # [B, num_classes]
            return logits


class GlobalMemory(nn.Module):
    """Summarize h vectors into global representation."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )

    def forward(self, short_term, memory):
        """
        short_term: [B, L, d_model]
        global: [B, d_model]
        """
        seq_summary = short_term.mean(dim=1)  # [B, d_model]
        combined = torch.cat([seq_summary, memory], dim=-1)
        g = self.gate(combined)
        u = self.update(combined)
        new_memory = memory * (1 - g) + u * g
        return new_memory


class MambaMemorySequenceClassifier(nn.Module):
    def __init__(
        self,
        N: int,
        num_classes: int,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)
        self.memory_blocks = nn.ModuleList([
            GlobalMemory(d_model) for _ in range(n_layers)
        ])
        self.memory_init = nn.Parameter(torch.zeros(1, d_model))
        self.mlm_head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: torch.LongTensor, mlm=None) -> torch.Tensor:
        B = x.size(0)
        h = self.embedding(x)
        memory = self.memory_init.expand(B, -1)

        for i, block in enumerate(self.layers):
            h = block(h)  # each Mamba block returns [B, L, d_model]
            memory = self.memory_blocks[i](h, memory)
            h = h + memory.unsqueeze(1)
        h = self.norm(h)  # [B, L, d_model]

        if mlm is not None:
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            pooled = h.mean(dim=1)  # [B, d_model]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)  # [B, num_classes]
            return logits


class DownsamplingCNNTokenizer(nn.Module):
    """
    CNN front-end that converts one-hot nucleotide sequences [B, L, 4]
    into a downsampled token sequence [B, L_reduced, d_model].
    """
    def __init__(
        self,
        in_channels=4,
        d_model=128,
        n_layers=2,
        kernel_size=9,
        downsample_factor=4,  # total reduction ratio
    ):
        super().__init__()
        layers = []
        c = in_channels
        current_factor = 1
        for i in range(n_layers):
            stride = 2 if current_factor < downsample_factor else 1
            current_factor *= stride
            layers += [
                nn.Conv1d(c, d_model, kernel_size,
                          stride=stride, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(d_model)
            ]
            c = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)             # [B, d_model, L_reduced]
        return h.transpose(1, 2)    # [B, L_reduced, d_model]


class MambaSequenceClassifierCNN(nn.Module):
    def __init__(
        self,
        N: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = DownsamplingCNNTokenizer()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        h = self.tokenizer(x)

        for block in self.layers:
            h = block(h)

        h = self.norm(h)
        pooled = h.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class MambaSequenceClassifierMemoryCNN(nn.Module):
    def __init__(
        self,
        N: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = DownsamplingCNNTokenizer()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)
        self.memory_blocks = nn.ModuleList([
            GlobalMemory(d_model) for _ in range(n_layers)
        ])
        self.memory_init = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        B = x.size(0)
        h = self.tokenizer(x)
        memory = self.memory_init.expand(B, -1)
        for i, block in enumerate(self.layers):
            h = block(h)  # each Mamba block returns [B, L, d_model]
            memory = self.memory_blocks[i](h, memory)
            h = h + memory.unsqueeze(1)
        h = self.norm(h)  # [B, L, d_model]
        pooled = h.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class MambaSequenceClassifierResidual(nn.Module):
    """A sequence classifier network that supports MLM."""
    def __init__(
        self,
        N: int,
        num_classes: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        pooling = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model + 1, num_classes)
        self.pooling = pooling
        self.mlm_head = nn.Linear(d_model, vocab_size + 1)
        self.combiner = MambaBlock(d_model=d_model + 1, d_state=d_state, d_conv=d_conv, expand=expand)
        self.final_norm = nn.LayerNorm(d_model + 1)

    def forward(self, x: torch.LongTensor, mlm=None) -> torch.Tensor:
        h = self.embedding(x)
        residual = x.unsqueeze(-1)   # [B, L, 1]
        for block in self.layers:
            h = block(h)   # each Mamba block returns [B, L, d_model]
        h = self.norm(h)

        # Residual
        h = torch.cat([h, residual], dim=-1)  # [B, L, d_model + 1]
        h = self.combiner(h)
        h = self.final_norm(h)

        if mlm is not None:
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            pooled = h.mean(dim=1)  # [B, d_model + 1]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)  # [B, num_classes]
            return logits
