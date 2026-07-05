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
import numpy as np
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import (CrossEntropyLoss, Module, Embedding, Linear, Parameter,
                      TransformerEncoderLayer, TransformerEncoder,
                      Sequential, Conv1d, ReLU, BatchNorm1d)
from torch import argmax, save, load, tensor, no_grad, arange, cat, randn
from torch.nn.utils import clip_grad_norm_


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat()}: {msg}")


# Models ######################################################################
class BERTax(Module):
    def __init__(
            self,
            N: int,
            num_classes: int,
            embed_dim: int = 250,
            vocab_size: int = 128,
            n_head: int = 4,
            dropout: float = 0.05,
            n_layers: int = 6,
            ):
        super(BERTax, self).__init__()
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
        self.cls_token = Parameter(randn(1, 1, embed_dim))
        self.position_embedding = Embedding(N + 1, embed_dim)

    def forward(self, x, mlm=None):
        batch_size, seq_len = x.size()
        token_embedding = self.token_embedding(x)
        cls = self.cls_token.repeat(batch_size, 1, 1) # (B, 1, D)
        x = cat([cls, token_embedding], dim=1) # (B, L+1, D)
        positions = arange(0, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        h = self.transformer_encoder(x)  # [B, L+1, D]
        if mlm is not None:
            h = h[:, 1:, :]
            logits = self.mlm_head(h)  # [B, L, vocab_size + 1]
            return logits
        else:
            h = h[:, 0, :]  # [B, D]
            logits = self.classifier(h)  # [B, num_classes]
            return logits


# Training ####################################################################

class LabelledDataset(Dataset):
    def __init__(self, directory: str):
        assert directory.endswith('/')
        self.x = np.load(directory + "x.npy")
        self.y = np.load(directory + "y.npy")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x), tensor(y)


class PretrainingDataset(Dataset):
    def __init__(self, directory: str):
        self.x = np.load(directory + "x.npy")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        return tensor(x)


def load_training_data(train, test, validate, batch_size):
    train_data = DataLoader(LabelledDataset(train), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(LabelledDataset(test), batch_size=batch_size, shuffle=True)
    validate_data = DataLoader(LabelledDataset(validate), batch_size=batch_size, shuffle=True)
    log(f"Number of training batches: {len(train_data)}")
    log(f"Number of test batches: {len(test_data)}")
    log(f"Number of validation batches: {len(validate_data)}")
    return train_data, test_data, validate_data


def pretrain(classifier, directory, pretrain_data, optimizer):
    pass  # TODO


def train_for_one_epoch(model, train_data, optimizer, loss_function, device):
    model.train()
    i = 0
    for x_batch, y_batch in train_data:
        x_batch = x_batch.to(device).long()
        y_batch = y_batch.to(device).long()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        i += 1
        log(i)
        if i > 250:
            break


def train(
        model, directory: str, n_epochs: int, train_data, validation_data, n_classes, device
    ) -> None:
    parameters = model.parameters()
    total_params = sum(param.numel() for param in parameters)
    log(f"Number of parameters: {total_params:_}")
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()
    for epoch in range(n_epochs):
        log(f"Epoch {epoch}")
        train_for_one_epoch(model, train_data, optimizer, loss_function, device)
        save(model.state_dict(), f"{directory}weights_{epoch + 1}_epoch.pt2")
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
    with no_grad():
        for x_batch, y_batch in validation_data:
            n += len(y_batch)
            x_batch = x_batch.to(device).long()
            y_batch = y_batch.to(device).long()
            output = model(x_batch)
            if loss_function:
                loss = loss_function(output, y_batch)
                validation_loss += loss.item()
            predictions = argmax(output, dim=1)
            predicted_y += predictions
            real_y += y_batch
            # TMP
            i += 1
            if i > 50:
                break
            # TMP
    if loss_function:
        validation_loss /= n
        log(f"Average loss: {validation_loss}")
    f1 = f1_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    log(f"F1 score: {f1}")
    precision = precision_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    log(f"Precision: {precision}")


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
    args = parser.parse_args()
    log(f"Creating the model.")
    model = BERTax(500, args.n_classes, 128)  # test
    model = model.to(args.device)
    if args.start_weight_epoch:
        log(f"Loading the model")
        model.load_state_dict(load(f"{args.directory}weights_{args.start_weight_epoch}_epoch.pt2"))
    log("Load training data.")
    train_data, test_data, validate_data = load_training_data(args.train, args.test, args.validate, args.batch_size)
    log("Train the model.")
    train(model, args.directory, args.n_epochs, train_data, validate_data, args.n_classes, args.device)
    # test
    log("Test the trained model.")
    validate(model, test_data, args.n_classes, args.device)



# 250
# 2026-07-05T14:08:16.521536+00:00: Average loss: 0.311730075116251
# 2026-07-05T14:08:16.731554+00:00: F1 score: 0.0004253250698748329
# 2026-07-05T14:08:16.760196+00:00: Precision: 0.00021995977878330823
# 2026-07-05T14:08:16.761134+00:00: Test the trained model.
# 2026-07-05T14:08:31.491923+00:00: F1 score: 7.807864080702083e-05
# 2026-07-05T14:08:31.518821+00:00: Precision: 3.927853192559075e-05

# 500
# 2026-07-05T14:24:10.276973+00:00: Average loss: 0.3086799505878897
# 2026-07-05T14:24:10.489036+00:00: F1 score: 0.0006133500882625737
# 2026-07-05T14:24:10.517356+00:00: Precision: 0.00032208396178984414
# 2026-07-05T14:24:10.518005+00:00: Test the trained model.
# 2026-07-05T14:24:24.769317+00:00: F1 score: 3.1345997116168264e-05
# 2026-07-05T14:24:24.796333+00:00: Precision: 1.5711412770236298e-05

# 750