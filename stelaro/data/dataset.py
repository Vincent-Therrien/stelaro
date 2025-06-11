"""
    Create PyTorch datasets to train and evaluate neural networks.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025
    - License: MIT
"""

import torch
from torch.utils.data import Dataset


class StaticDataset(Dataset):
    def __init__(self, directory: str, reference_genomes: dict):
        self.samples = []
        with open(fasta_file, "r") as f:
            for line in f:
                if line[0] in list(NUCLEOTIDE_TO_ONEHOT.keys()):
                    encoded = one_hot_encode(line.strip())
                    self.samples.append((encoded, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]


class AdaptiveDataset(Dataset):
    def __init__(self, fasta_file, label):
        self.samples = []
        with open(fasta_file, "r") as f:
            for line in f:
                if line[0] in list(NUCLEOTIDE_TO_ONEHOT.keys()):
                    encoded = one_hot_encode(line.strip())
                    self.samples.append((encoded, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
