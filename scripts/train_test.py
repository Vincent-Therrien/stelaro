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
from torch.nn import CrossEntropyLoss
from torch import argmax, save, load, tensor, no_grad
from torch.nn.utils import clip_grad_norm_

# Models ######################################################################




# Training ####################################################################

class LabelledDataset(Dataset):
    def __init__(self, directory: str):
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
    return train_data, test_data, validate_data


def pretrain(classifier, directory, pretrain_data, optimizer):
    pass  # TODO


def train_for_one_epoch(model, train_data, optimizer, loss_function, device):
    model.train()
    for x_batch, y_batch in train_data:
        x_batch = x_batch.to(device).long()
        output = model(x_batch)
        y_batch = y_batch.to(device).long()
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def train(
        model, directory: str, n_epochs: int, train_data, validation_data, n_classes
    ) -> None:
    parameters = model.get_parameters()
    total_params = sum(param.numel() for param in parameters)
    print(f"Number of parameters: {total_params:_}")
    optimizer = Adam(model.get_parameters(), lr=0.001)
    loss_function = CrossEntropyLoss()
    for epoch in range(n_epochs):
        print(f"{datetime.now(timezone.utc).isoformat()}: Epoch {epoch}")
        train_for_one_epoch(model, train_data, optimizer, loss_function, "cuda")
        save(model.state_dict(), f"{directory}weights_{epoch}_epoch.pt2")
        validate(
            model, validation_data, n_classes, "cuda", loss_function
        )


# Evaluation ##################################################################
def validate(model, validation_data, n_classes, device, loss_function=None):
    model.eval()
    n = 0
    validation_loss = 0
    real_y, predicted_y = [], []
    with no_grad():
        for x_batch, y_batch in validation_data:
            n += len(y_batch)
            x_batch = x_batch.to(device).long()
            output = model(x_batch)
            if loss_function:
                loss = loss_function(output, y_batch)
                validation_loss += loss.item()
            predictions = argmax(output, dim=1)
            predicted_y += predictions
            real_y += y_batch
    if loss_function:
        validation_loss /= n
        print(f"Average loss: {validation_loss}")
    f1 = f1_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    print(f"F1 score: {f1}")
    precision = precision_score(
        real_y,
        predicted_y,
        average="macro",
        labels=range(n_classes),
        zero_division=0.0
    )
    print(f"Precision: {precision}")


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
    # create
    model = None
    # load
    if args.start_weight_epoch:
        model.load_state_dict(load(f"{args.directory}weights_{args.start_weight_epoch}_epoch.pt2"))
    # train
    train_data, test_data, validate_data = load_training_data(args.train, args.test, args.validate, args.batch_size)
    train(model, args.directory, args.n_epochs, train_data, validation_data, args.n_classes)
    # test
    validate(model, test_data, args.n_classes, "cuda")
