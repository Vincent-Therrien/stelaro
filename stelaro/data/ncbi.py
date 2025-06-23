"""NCBI data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import os
from random import shuffle
import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import stelaro.stelaro as stelaro_rust


ASSEMBLY_ACCESSION_NUMBER_COLUMN = 0
ASSEMBLY_TAXID_COLUMN = 5
TAXONOMY_TAXID_COLUMN = 0
TAXONOMY_PARENT_TAXID_COLUMN = 2
TAXONOMY_RANK_COLUMN = 4
NAMES_TAXID_COLUMN = 0
NAME_COLUMN = 2
NAME_TYPE_COLUMN = 6


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
            taxIDs.append((ID, taxID))
    return taxIDs


def get_assembly_taxid(file: str) -> set[str]:
    """Obtain the taxonomy identifiers in an NCBI genome summary file.

    There can be multiple reference genomes for one taxon, so the number of
    elements returned by this function is inferior to the number of reference
    genomes.

    Args:
        file: NCBI genome summary file path.

    Returns: Set of taxonomy identifiers.
    """
    taxIDs = {}
    with open(file, "r") as f:
        next(f)  # Skip the first line.
        next(f)  # Skip the header.
        for line in f:
            fields = line.strip().split("\t")
            taxID = fields[ASSEMBLY_TAXID_COLUMN]
            reference_genome = fields[ASSEMBLY_ACCESSION_NUMBER_COLUMN]
            if taxID not in taxIDs:
                taxIDs[taxID] = []
            taxIDs[taxID].append(reference_genome)
    return taxIDs


def _get_taxonomy_parents(
        lines: list[str],
        tax_ids: set[str],
        levels: list[str] = None
        ) -> list:
    """Obtain the the NCBI taxonomy nodes.

    Args:
        lines: Content of the `nodes.dmp` file of the NCBI taxonomy.
        tax_ids: Set of taxonomy IDs to retain.
        level: An optional name to restrict the taxa to a specific rank.

    Returns: A list of tuples formatted as (taxid, parent_taxid, rank).
    """
    result = []
    for line in lines:
        fields = line.strip().split("\t")
        taxid = fields[TAXONOMY_TAXID_COLUMN]
        if taxid not in tax_ids:
            continue
        if levels and rank not in levels:
            continue
        rank = fields[TAXONOMY_RANK_COLUMN]
        parent = fields[TAXONOMY_PARENT_TAXID_COLUMN]
        result.append((taxid, parent, rank))
    return result


def get_all_taxonomy_parents(file: str, tax_ids: set[str]) -> tuple:
    """Recursively fetch all parents in the NCBI taxonomy.

    Args:
        file: File path to the `nodes.dmp` file of the NCBI taxonomy,
        tax_ids: Set of taxonomy IDs to retain.

    Return: A tuple of dict: (parent to children, tax_id to rank)
    """
    id_to_parent = {}
    id_to_rank = {}
    with open(file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            taxid = fields[TAXONOMY_TAXID_COLUMN]
            rank = fields[TAXONOMY_RANK_COLUMN]
            parent = fields[TAXONOMY_PARENT_TAXID_COLUMN]
            id_to_parent[taxid] = parent
            id_to_rank[taxid] = rank
    return id_to_parent, id_to_rank
    subset = tax_ids
    parents = {}
    ranks = {}
    while True:
        print(f"Looking for taxonomic parents: {len(subset)}")
        current_subset = subset.copy()
        subset = set()
        for child in current_subset:
            if child not in id_to_parent:
                print(f"Reference genome with tax ID {child} has to taxonomy.")
                continue
            parent, rank = id_to_parent[child], id_to_rank[child]
            ranks[child] = rank
            subset.add(parent)
            if parent not in parents:
                parents[parent] = set()
            parents[parent].add(child)
        if len(subset) == 1:
            break
    return parents, ranks


def resolve_taxonomy(
        parents: list[tuple],
        ranks: dict,
        tax_ids: set[str],
        names: dict,
        ) -> tuple[rx.PyDiGraph, dict]:
    """Convert a list of taxIDs into a taxonomic tree.

    Args:
        taxonomy_nodes: Dict returned by `get_all_taxonomy_parents`.

    Returns: A network representing the taxonomy and a dictionary that maps
        taxonomic IDs to node indices.
    """
    lineages = []
    for taxon in tax_ids:
        lineage = []
        index = taxon
        if taxon not in parents:
            print(f"Taxon {taxon} has no known lineage.")
            continue
        while True:
            if index not in names:
                print(f"Taxon {index} is nameless.")
                name = "Undefined"
            else:
                name = names[index]
            if index not in ranks:
                print(f"Taxon {index} is rankless.")
                rank = "Undefined"
            else:
                rank = ranks[index]
            lineage.append((taxon, name, rank))
            if index == parents[index]:
                break
            index = parents[index]
        lineage = [e for e in reversed(lineage)]
        lineages.append(lineage)
    return lineages


def taxid_to_names(file: str, tax_ids: set[str]) -> dict:
    """Convert taxIDs into names.

    Args:
        file: Path of the `names.dmp` file.
        tax_ids: NCBI taxonomic identifiers to link to a name.

    Returns: A dictionary that maps a taxonomy identifier to a name.
    """
    names = {}
    tax_ids = tax_ids.copy()
    with open(file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            tax_id = fields[NAMES_TAXID_COLUMN]
            name = fields[NAME_COLUMN]
            name_type = fields[NAME_TYPE_COLUMN]
            if tax_id in tax_ids and name_type == "scientific name":
                names[tax_id] = name
                tax_ids.remove(tax_id)
    return names


def find_depth(lineage, depth: tuple[str]) -> tuple[str]:
    result = []
    for _, name, rank in lineage:
        if rank == depth[len(result)]:
            result.append(name)
        if len(result) == len(depth):
            break
    return tuple(result)


def find_granularity(lineage, granularity: str) -> tuple[str]:
    result = []
    for _, name, rank in lineage:
        if rank == granularity:
            result.append(name)
    return tuple(result)


def bin_genomes(
        lineages: list[tuple],
        depth: tuple[str],
        granularity_level: str,
        tax_to_genome: dict,
        max_bin_size: int,
        n_min_bins: int,
        n_max_bins: int,
        ) -> list:
    """Split a set of reference genomes into bins.

    Args:
        lineages: List of taxonomic lineages.
        depth: Name of the taxonomic level at which to bin the genomes.
            E.g. `("acellular root", "realm")` will group by virus realms.
        granularity_level: Name of the taxonomic level at which to split the
            genomes. E.g. `genus` will group by genus.
        tax_to_genome: Dictionary that maps taxonomic IDs to reference genomes.
        n_min_bins: Minimum number of elements in each granularity level.
        max_bin_size: Maximum number of elements in each granularity level.
        n_max_bins: Maximum number of bins.
    """
    datasets = {}
    n_initial = len(lineages)
    n = 0
    for lineage in lineages:
        key = find_depth(lineage, depth)
        if len(key) != len(depth):
            # print(f"Incomplete lineage: {lineage}")
            continue
        granularity = find_granularity(lineage, granularity_level)
        if len(granularity) != 1:
            # print(f"Cannot find granularity: {lineage}")
            continue
        if key not in datasets:
            datasets[key] = {}
        if granularity not in datasets[key]:
            datasets[key][granularity] = []
        taxon = lineage[-1][0]
        datasets[key][granularity] += tax_to_genome[taxon]
        n += 1
    print(f"Out of {n_initial} reconstructed taxonomies, retained {n}.")
    clean_dataset = []
    n_retained = 0
    for key, value in datasets.items():
        clean_value = []
        for granularity, reference_genomes in value.items():
            if max_bin_size and len(reference_genomes) > max_bin_size:
                references = reference_genomes.copy()
                shuffle(references)
                cropped_genomes = reference_genomes[:max_bin_size]
                clean_value.append([granularity, cropped_genomes])
            else:
                clean_value.append([granularity, reference_genomes])
        if len(clean_value) < n_min_bins:
            print(f"`{key}` contains {len(clean_value)} values, dropping.")
            continue
        if len(clean_value) > n_max_bins:
            print(f"`{key}` contains {len(clean_value)} values, reducing.")
            shuffle(clean_value)
            clean_value = clean_value[:n_max_bins]
        for v in clean_value:
            n_retained += len(v[1])
        clean_dataset.append([key, clean_value])
    print(f"Retained {n_retained} reference genomes to balance the dataset.")
    return clean_dataset


def visualize_taxonomy(graph, root, depth) -> None:
    G = rx.PyDiGraph()
    G_nodes = {root: G.add_node(graph[root])}

    def recurse(node: int, i: int):
        if i > depth:
            return
        edges = graph.out_edges(node)
        if edges:
            for edge in edges:
                _, next_node, _ = edge
                G_nodes[next_node] = G.add_child(G_nodes[node], graph[next_node], 1)
                recurse(next_node, i + 1)

    recurse(root, 0)
    mpl_draw(
        G,
        with_labels=True,
        labels=str,
        edge_labels=str,
        node_color=(1.0, 1.0, 1.0),
        node_size=500
    )
