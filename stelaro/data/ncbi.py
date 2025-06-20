"""NCBI data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import os
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
    lines = []
    with open(file, "r") as f:
        for line in f:
            lines.append(line)
    species = _get_taxonomy_parents(lines, tax_ids)
    subset = set([p[1] for p in species])
    parents = {}
    ranks = {}
    while True:
        result = _get_taxonomy_parents(lines, subset)
        if len(result) <= 1:
            break
        subset = set([p[1] for p in result])
        print(f"Looking for taxonomic parents: {len(subset)}")
        for r in result:
            child, parent, rank = r[0], r[1], r[2]
            ranks[child] = rank
            if parent not in parents:
                parents[parent] = set()
            parents[parent].add(child)
    return parents, ranks


def resolve_taxonomy(
        parents: list[tuple],
        ranks: dict,
        ) -> tuple[rx.PyDiGraph, dict]:
    """Convert a list of taxIDs into a taxonomic tree.

    Args:
        taxonomy_nodes: Dict returned by `get_all_taxonomy_parents`.

    Returns: A network representing the taxonomy and a dictionary that maps
        taxonomic IDs to node indices.
    """
    graph = rx.PyDiGraph()
    taxid_to_index = {}
    n = 0
    for parent, children in parents.items():
        for child in children:
            if parent in taxid_to_index:
                if child in taxid_to_index:
                    graph.add_edge(
                        taxid_to_index[parent],
                        taxid_to_index[child],
                        1
                    )
                else:
                    label = (child, ranks[child])
                    taxid_to_index[child] = graph.add_child(
                        taxid_to_index[parent],
                        label,
                        1
                    )
            elif child in taxid_to_index:
                label = (parent, ranks[parent])
                taxid_to_index[parent] = graph.add_parent(
                    taxid_to_index[child],
                    label,
                    1
                )
            else:
                label = (parent, ranks[parent])
                taxid_to_index[parent] = graph.add_node(label)
                label = (child, ranks[child])
                taxid_to_index[child] = graph.add_child(
                    taxid_to_index[parent],
                    label,
                    1
                )
            n += 1
    return graph, taxid_to_index


def collapse_taxonomy(
        graph: rx.PyDiGraph,
        taxid_to_index: dict,
        levels: tuple[str] = (
            "domain",
            "realm",
            "phylum",
            "kingdom",
            "class",
            "order",
            "family",
            "genus",
            "species",
            "no rank",
        )
    ) -> list[tuple]:
    """Collapse an NCBI taxonomy.

    Args:
        graph: Graph returned by `resolve_taxonomy`.
        taxid_to_index: Dictionary returned by `resolve_taxonomy`.
        levels: Name of the levels to retain in the collapsed taxonomy.

    Returns: A list of the taxonomy of each species.
    """
    species = []
    for tax_id, node in taxid_to_index.items():
        if graph[node][1] == "species":
            species.append(tax_id)

    def find_lineage(node: int) -> list[str]:
        lineage = [graph[node]]
        while True:
            edges = graph.in_edges(node)
            if len(edges) == 1:
                parent, child, _ = edges[0]
                if parent == child:
                    break
                lineage.append(graph[parent])
                node = parent
            elif len(edges) == 0:
                break
            else:
                raise RuntimeError("Unexpected number of parents.")
        lineage = reversed(lineage)
        return [e for e in lineage if e[1] in levels]

    lineages = [find_lineage(taxid_to_index[s]) for s in species]
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


def split_non_similar_genomes(
        lineages: list[tuple],
        depth: int,
        granularity_level: int,
        names: dict,
        tax_to_genome: dict,
        ) -> list:
    datasets = {}
    for lineage in lineages:
        key = [e for e in lineage[:depth]]
        key = tuple([names[k[0]] for k in key])
        granular_level = [e for e in lineage[-granularity_level:]]
        granular_level = tuple([names[g[0]] for g in granular_level])
        if key not in datasets:
            datasets[key] = {}
        if granular_level not in datasets[key]:
            datasets[key][granular_level] = []
        species = lineage[-1][0]
        print(species, tax_to_genome[species])
        datasets[key][granular_level] += tax_to_genome[species]
    return datasets


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
