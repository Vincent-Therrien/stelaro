"""NCBI data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import os
import stelaro.stelaro as stelaro_rust


def install_summaries(path: str, force: bool = False) -> None:
    """Download reference assembly summaries.

    Args:
        path: Directory in which to install data.
    """
    stelaro_rust.download("ncbi", "genome_summaries", path, force)


def assembly_summaries(path: str) -> None:
    """Visualize reference genome counts.

    Args:
        path: Directory in which the NCBI dataset is installed.
    """
    files = os.listdir(path)
    if "ncbi_genome_summaries" not in files:
        print(f"Error: NCBI genome summaries are not installed at `{path}`.")
        print("Execute `ncbi.install_summaries(path)` to install them.")
    files = os.listdir(path + "/ncbi_genome_summaries")
    total = 0
    for filename in files:
        name = path + "/ncbi_genome_summaries/" + filename
        with open(name, "r", encoding='utf-8') as file:
            n_lines = sum(1 for _ in file)
        n_lines -= 2  # Comment lines at the beginning of the file.
        total += n_lines
        print(f"{filename[:-4]}: {n_lines:_} genomes".replace("_", " "))
    print()
    print(f"Total: {total:_}".replace("_", " "))
