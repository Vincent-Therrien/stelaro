"""NCBI data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import os
import numpy as np
import stelaro.stelaro as stelaro_rust


def install_summaries(path: str, force: bool = False) -> None:
    """Download reference assembly summaries.

    Args:
        path: Directory in which to install data.
    """
    stelaro_rust.install("ncbi", "genome_summaries", path, force)


def summarize_assemblies(path: str) -> None:
    """Visualize reference genome counts.

    Args:
        path: Directory in which the NCBI dataset is installed.
    """
    files = os.listdir(path)
    total = 0
    for filename in files:
        name = path + filename
        with open(name, "r", encoding='utf-8') as file:
            n_lines = sum(1 for _ in file)
        n_lines -= 2  # Comment lines at the beginning of the file.
        total += n_lines
        print(f"{filename[:-4]}: {n_lines:_} genomes".replace("_", " "))
    print()
    print(f"Total: {total:_}".replace("_", " "))


def sample_genomes(
        src: str,
        dst: str,
        sampling: str | list = "micro",
        fraction: float = 1.0
        ):
    """Sample genomes from assembly summaries.

    Args:
        src: Location of the assembly summaries.
        dst: Name of the file in which to save the index.
        sampling: Type of sampling. `full` or `micro`.
        fraction: Proportion of genomes to sample from the original dataset.
    """
    stelaro_rust.sample_genomes("ncbi", src,  dst, sampling, fraction)


def install_genomes(
        src: str,
        dst: str
        ):
    """Install genomes listed in a file.

    Args:
        src: Index file containing genomes.
        dst: Directory in which to install the genomes.
    """
    stelaro_rust.install_genomes(src,  dst)


def synthetic_samples(
        genome_index_filepath: str,
        genome_directory: str,
        reads: int,
        length: int,
        encoding: str,
        length_deviation: int=0,
        indels: int=0,
        indels_deviation: int=0,
        ) -> np.ndarray:
    """Sample synthetic metagenome samples from reference genomes.

    Args:
        genome_index_filepath: Location of the file listing all genomes.
        genome_directory: Directory containing the reference genomes.
        reads: Number of reads to sample.
        length: Average length of a read.
        encoding: Encoding format used to transform the sequences into Numpy
            arrays. Supported encodings: `onehot`.
        length_deviation: Average deviation for the sample length.
        indels: Number of indels to add to each sample.
        indels_deviation: Average deviation for the number of indels.

    Returns: A tuple containing (1) the encoded samples and (2) the list
        of genome identifiers from which the genomes were sampled.
    """
    encodings, identifiers = stelaro_rust.synthetic_sample(
        genome_index_filepath,
        genome_directory,
        reads=reads,
        length=length,
        length_deviation=length_deviation,
        indels=indels,
        indels_deviation=indels_deviation,
        encoding=encoding
    )
    return encodings, identifiers
