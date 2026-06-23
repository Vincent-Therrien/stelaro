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
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import save, load, tensor

# Models ######################################################################




# Utility functions ###########################################################

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


def train_for_one_epoch(classifier, directory, train_data, validation_data, optimizer):
    pass  # TODO


def train_classifier(
        classifier, directory: str, n_epochs: int, train_data, validation_data
    ) -> None:
    parameters = classifier.get_parameters()
    total_params = sum(param.numel() for param in parameters)
    print(f"Number of parameters: {total_params:_}")
    optimizer = Adam(classifier.get_parameters(), lr=0.001)
    for epoch in range(n_epochs):
        print(f"{datetime.now(timezone.utc).isoformat()}: Epoch {epoch}")
        train_for_one_epoch(
            classifier,
            directory,
            train_data,
            validation_data,
            optimizer,
        )
        save(classifier.model.state_dict(), f"{directory}weights_{epoch}_epoch.pt2")


def test_classifier(classifier, directory):
    pass  # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model benchmarking."
    )
    parser.add_argument("--train", type=str, help="Filepath to training set")
    parser.add_argument("--test", type=str, help="Filepath to testing set")
    parser.add_argument("--validate", type=str, help="Filepath to validation set")
    parser.add_argument("--n_classes", type=int, help="Number of classes")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_max_epochs", type=int, help="Maximum number of epochs")
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

    # load
    if args.start_weight_epoch:
        classifier.model.load_state_dict(load(f"{arg.directory}weights_{args.start_weight_epoch}_epoch.pt2"))
    # train
    train, test, validate = load_training_data(args.train, args.test, args.validate, args.batch_size)
    # test
