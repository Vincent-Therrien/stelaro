"""NCBI data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import os
import numpy as np
import rustworkx as rx
import stelaro.stelaro as stelaro_rust


ASSEMBLY_ACCESSION_NUMBER_COLUMN = 0
ASSEMBLY_TAXID_COLUMN = 5
TAXONOMY_TAXID_COLUMN = 0
TAXONOMY_PARENT_TAXID_COLUMN = 2
TAXONOMY_RANK_COLUMN = 4


def install_summaries(path: str, force: bool = False) -> None:
    """Download reference assembly summaries.

    Args:
        path: Directory in which to install data.
    """
    stelaro_rust.install("ncbi", "genome_summaries", path, force)


def install_taxonomy(path: str, force: bool = False) -> None:
    """Download reference assembly taxonomy.

    Args:
        path: Directory in which to install data.
    """
    stelaro_rust.install("ncbi", "taxonomy", path, force)


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
        length_deviation: int = 0,
        indels: int = 0,
        indels_deviation: int = 0,
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


def get_reference_genome_taxid(file: str) -> set[str]:
    """Obtain the taxonomy identifiers in an NCBI genome summary file.

    Args:
        file: NCBI genome summary file path.

    Returns: Set of taxonomy identifiers.
    """
    taxIDs = set()
    with open(file, "r") as f:
        next(f)  # Skip the first line.
        next(f)  # Skip the header.
        for line in f:
            fields = line.strip().split("\t")
            ID = fields[ASSEMBLY_ACCESSION_NUMBER_COLUMN]
            taxIDs
            taxID = fields[ASSEMBLY_TAXID_COLUMN]
            pairs.append((ID, taxID))
    return pairs


def get_assembly_taxid(file: str) -> set[str]:
    """Obtain the taxonomy identifiers in an NCBI genome summary file.

    Args:
        file: NCBI genome summary file path.

    Returns: Set of taxonomy identifiers.
    """
    taxIDs = set()
    with open(file, "r") as f:
        next(f)  # Skip the first line.
        next(f)  # Skip the header.
        for line in f:
            fields = line.strip().split("\t")
            taxID = fields[ASSEMBLY_TAXID_COLUMN]
            taxIDs.add(taxID)
    return taxIDs


def _get_taxonomy_parents(lines: list[str], tax_ids: set[str]) -> list:
    """Obtain the the NCBI taxonomy nodes.

    Args:
        lines: Content of the `nodes.dmp` file of the NCBI taxonomy.
        tax_ids: Set of taxonomy IDs to retain.

    Returns: A list of tuples formatted as (taxid, parent_taxid, rank).
    """
    result = []
    for line in lines:
        fields = line.strip().split("\t")
        taxid = fields[TAXONOMY_TAXID_COLUMN]
        if taxid not in tax_ids:
            continue
        parent = fields[TAXONOMY_PARENT_TAXID_COLUMN]
        rank = fields[TAXONOMY_RANK_COLUMN]
        result.append((taxid, parent, rank))
    return result


def get_all_taxonomy_parents(file: str, tax_ids: set[str]) -> list:
    """Recursively fetch all parents in the NCBI taxonomy.

    Args:
        file: File path to the `nodes.dmp` file of the NCBI taxonomy,
        tax_ids: Set of taxonomy IDs to retain.
    """
    hierarchy = []
    subset = tax_ids
    lines = []
    with open(file, "r") as f:
        for line in f:
            lines.append(line)
    while True:
        result = _get_taxonomy_parents(lines, subset)
        if len(result) <= 1:
            break
        hierarchy.append(result)
        subset = set([p[1] for p in result])
        print(f"Looking for taxonomic parents: {len(subset)}")
    return hierarchy


def resolve_taxonomy(taxIDs: set[str], taxonomy_nodes: dict) -> rx.PyDiGraph:
    """TODO: Convert a list of taxIDs into a taxonomic tree.

    Args:
        taxIDs: List of NCBI taxonomy identifiers.
        taxonomy_nodes: Dictionary representing the taxonomy network, as
            returned by the function `stelaro.data.ncbi.get_taxonomy_nodes`.

    Returns: A network representing the taxonomy.
    """
    graph = rx.PyDiGraph()
    taxid_to_index = {}
    for taxID in taxIDs:
        if taxID not in taxid_to_index:
            taxid_to_index[taxID] = graph.add_node(taxID)
            parent = taxonomy_nodes[taxID][0]
            if parent not in taxid_to_index:
                taxid_to_index[parent] = graph.add_node(parent)
            graph.add_edge(parent, taxid_to_index[taxID], 1)
    # TODO: Resolve the root
    return graph


def taxid_to_names(file: str, taxIDs: list[str]) -> list[str]:
    """TODO: Convert taxIDs into names."""
    pass
