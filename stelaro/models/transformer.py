"""
    Transformer neural network module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: July 2025
    - License: MIT
"""

from torch import arange, cat, randn
from torch.nn import (Module, Embedding, Linear, Parameter,
                      TransformerEncoderLayer, TransformerEncoder,
                      Sequential, Conv1d, ReLU, BatchNorm1d)


class TransformerClassifier(Module):
    def __init__(
            self,
            N: int,
            num_classes: int,
            embed_dim: int = 128,
            vocab_size: int = 256,
            n_head: int = 4,
            dropout: float = 0.1,
            n_layers: int = 4,
            classification_type: str = "mean"
            ):
        super(TransformerClassifier, self).__init__()
        self.token_embedding = Embedding(vocab_size + 1, embed_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # makes input/output shape [B, L, D]
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = Linear(embed_dim, vocab_size + 1)
        self.classifier = Linear(embed_dim, num_classes)
        self.classification_type = classification_type
        if classification_type == "mean":
            self.position_embedding = Embedding(N, embed_dim)
        if classification_type == "token":
            self.cls_token = Parameter(randn(1, 1, embed_dim))
            self.position_embedding = Embedding(N + 1, embed_dim)

    def forward(self, x, mlm=None):
        batch_size, seq_len = x.size()
        token_embedding = self.token_embedding(x)

        if self.classification_type == "token":
            cls = self.cls_token.repeat(batch_size, 1, 1) # (B, 1, D)
            x = cat([cls, token_embedding], dim=1) # (B, L+1, D)
            positions = arange(0, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)
        else:
            positions = arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            x = token_embedding + self.position_embedding(positions)  # [B, L, D]
        h = self.transformer_encoder(x)  # [B, L, D]
        if mlm is not None:
            if self.classification_type == "token":
                h = h[:, 1:, :]
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            if self.classification_type == "mean":
                h = h.mean(dim=1)  # [B, D] - average pooling
            elif self.classification_type == "token":
                h = h[:, 0, :]  # [B, D]
            logits = self.classifier(h)  # [B, num_classes]
            return logits


class DownsamplingCNNTokenizer(Module):
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
                Conv1d(c, d_model, kernel_size,
                          stride=stride, padding=kernel_size // 2),
                ReLU(),
                BatchNorm1d(d_model)
            ]
            c = d_model
        self.net = Sequential(*layers)

    def forward(self, x):
        h = self.net(x)             # [B, d_model, L_reduced]
        return h.transpose(1, 2)    # [B, L_reduced, d_model]


class TransformerClassifierCNN(Module):
    def __init__(
            self,
            N: int,
            num_classes: int,
            embed_dim: int = 128,
            vocab_size: int = 256,
            n_head: int = 4,
            dropout: float = 0.1,
            n_layers: int = 4,
            classification_type: str = "mean"
            ):
        super(TransformerClassifierCNN, self).__init__()
        self.tokenizer = DownsamplingCNNTokenizer()
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # makes input/output shape [B, L, D]
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = Linear(embed_dim, vocab_size + 1)
        self.classifier = Linear(embed_dim, num_classes)
        self.classification_type = classification_type
        if classification_type == "mean":
            self.position_embedding = Embedding(N, embed_dim)
        if classification_type == "token":
            self.cls_token = Parameter(randn(1, 1, embed_dim))
            self.position_embedding = Embedding(N + 1, embed_dim)

    def forward(self, x, mlm=None):
        batch_size, _, seq_len = x.size()
        seq_len //= 4
        token_embedding = self.tokenizer(x)

        if self.classification_type == "token":
            cls = self.cls_token.repeat(batch_size, 1, 1) # (B, 1, D)
            x = cat([cls, token_embedding], dim=1) # (B, L+1, D)
            positions = arange(0, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)
        else:
            positions = arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            x = token_embedding + self.position_embedding(positions)  # [B, L, D]
        h = self.transformer_encoder(x)  # [B, L, D]
        if mlm is not None:
            if self.classification_type == "token":
                h = h[:, 1:, :]
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            if self.classification_type == "mean":
                h = h.mean(dim=1)  # [B, D] - average pooling
            elif self.classification_type == "token":
                h = h[:, 0, :]  # [B, D]
            logits = self.classifier(h)  # [B, num_classes]
            return logits

