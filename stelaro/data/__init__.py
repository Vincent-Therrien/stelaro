"""
    The modules in this directory serve as (1) bindings for the `data` Rust
    module that installs and processes data and (2) a Python module for data
    visualization.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

from random import sample, shuffle
import rustworkx as rx
import matplotlib.axes as plt
from matplotlib.axes._axes import Axes
from rustworkx.visualization import mpl_draw

import stelaro.stelaro as stelaro_rust

__all__ = ["ncbi", "gtdb", ]


def synthetic_metagenome(
        src: str,
        genomes: str,
        dst: str,
        reads: int,
        length: int,
        length_deviation: int = 0,
        indels: int = 0,
        indels_deviation: int = 0
        ) -> None:
    stelaro_rust.synthetic_metagenome(
        src,
        genomes,
        dst,
        reads,
        length,
        length_deviation,
        indels,
        indels_deviation
    )


def get_index_size(index: str) -> int:
    """Get the size of the genomes stored at an index."""
    total = 0
    with open(index, "r") as f:
        next(f)  # Skip the header.
        for line in f:
            fields = line.strip().split("\t")
            size = int(fields[-1])
            total += size
    return total


def read_genome(file: str) -> list[str]:
    """Read a reference genome."""
    return stelaro_rust.read_fasta(file)["obj"]


class Taxonomy():
    """Graph representation of a taxonomy with 7 levels.

    This class is mainly based on the GTDB taxonomy, but it is meant to be
    flexible enough to accept NCBI taxonomies for references genomes that are
    not classified by the GTDB project (e.g. eukaryotes and viruses).

    Attributes:
        graph: A directed graph that points at lower taxonomic levels. Each
            node corresponds to a taxon. The leaves are reference genomes.
        genomes: A dictionary that maps a database to a dictionary that maps
            a node index representing a species to a list of identifiers of
            reference genomes of that species.
        root: The index of the root of the graph.
        databases: List of databases used by the Taxonomy,
        genome_counts: A dictionary that maps a graph species index (i.e. a
            species node) to the number of reference genomes in that species.
    """
    DATABASES = ("refseq", "genbank")
    TAXONOMIC_LEVELS = (
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species"
    )

    def __init__(
            self,
            databases: tuple[str] = ("refseq", )
            ):
        """Create a taxonomy manipulation object.

        Args:
            databases: Genome reference databases to be used. Supported
                databases: `refseq`, `genbank`.
        """
        self.graph = rx.PyDiGraph()
        self.root = self.graph.add_node("root")
        assert set(databases) <= set(Taxonomy.DATABASES), (
            f"Invalid databases `{databases}`. "
            + f"Valid databases: `{Taxonomy.DATABASES}`.")
        self.databases = databases
        self.genomes = {}
        for database in databases:
            self.genomes[database] = {}
        self.genome_counts = {}

    def _get_GTDB_line_identifier(self, line: str) -> str:
        """Get the genome assembly accession ID from a GTDB taxonomy file."""
        identifier = line.split("\t")[0]
        return identifier[len("RS_"):]  # Remove the GTDB-specific prefix.

    def _get_GTDB_line_taxonomy(self, line: str) -> list[str]:
        """Get the full taxonomy of a line in a GTDB taxonomy file."""
        line = line.split("\t")[-1]
        fields = line.strip().split(';')
        assert len(fields) == len(Taxonomy.TAXONOMIC_LEVELS), (
            f"Unexpected line: {line}")
        return [f[3:] for f in fields]  # Remove the `d__` prefixes.

    def _count_reference_genomes(self) -> int:
        """Count the number of leaves in the graph."""
        total = 0
        for _, n in self.genome_counts.items():
            total += n
        return total

    def _add_genome(
            self,
            database: str,
            identifier: str,
            taxonomy: list[str]
            ) -> None:
        """Add a genome to the current object."""
        index = self.root
        level = 0
        for field in taxonomy:
            edges = self.graph.out_edges(index)
            if edges:
                for edge in edges:
                    _, node_index, _ = edge
                    if self.graph[node_index] == taxonomy[level]:
                        index = node_index
                        break
                else:
                    index = self.graph.add_child(index, field, 1)
            else:
                index = self.graph.add_child(index, field, 1)
            level += 1
        if index in self.genome_counts:
            self.genome_counts[index] += 1
        else:
            self.genome_counts[index] = 1
        if index in self.genomes[database]:
            self.genomes[database][index].append(identifier)
        else:
            self.genomes[database][index] = [identifier]

    def read_GTDB_file(self, path: str) -> None:
        """Read a GTDB taxonomy file and add its content to the object."""
        with open(path, "r") as f:
            for line in f:
                if line.startswith("RS_"):
                    database = "refseq"
                elif line.startswith("GB_"):
                    database = "genbank"
                if database in self.databases:
                    identifier = self._get_GTDB_line_identifier(line)
                    taxonomy = self._get_GTDB_line_taxonomy(line)
                    self._add_genome(database, identifier, taxonomy)

    def find_node(self, path: list[str] = [], ) -> int:
        if not path:
            return self.root
        index = self.root
        # Find the top-level node.
        for field in path:
            edges = self.graph.out_edges(index)
            if edges:
                for edge in edges:
                    _, node_index, _ = edge
                    if self.graph[node_index] == field:
                        index = node_index
                        break
            else:
                raise RuntimeError(f"Could not find `{path}`.")
        return index

    def extract(
            self,
            path: list[str] = [],
            depth: int = float("inf")
            ) -> rx.PyDiGraph:
        """Extract a subgraph from the taxonomy.

        Returns: Tuple containing (graph, root).
        """
        index = self.find_node(path)
        # Extract the subgraph.
        levels = [[index], ]
        while True:
            nodes = []
            for node in levels[-1]:
                edges = self.graph.out_edges(node)
                for edge in edges:
                    _, new_node, _ = edge
                    nodes.append(new_node)
            if not nodes:
                break  # Reached the lowest taxonomic level.
            levels.append(nodes)
            if len(levels) > depth:
                break  # Reached the maximum depth.
        nodes = []
        for level in levels:
            nodes += level
        return self.graph.subgraph(nodes), index

    def visualize(self, axis, path: list[str] = [], ) -> None:
        """Visualize the taxonomy graph in a pyplot graph."""
        G = self.extract(path)[0]
        mpl_draw(
            G,
            with_labels=True,
            labels=str,
            edge_labels=str,
            node_color=(1.0, 1.0, 1.0),
            node_size=500
        )

    def _get_n_leaves(self, node: int) -> int:
        """Obtain the number of leaves that can be accessed from a node."""
        def recurse(n: int):
            count = 0
            edges = self.graph.out_edges(n)
            if edges:
                for edge in edges:
                    _, node_index, _ = edge
                    count += recurse(node_index)
                return count
            else:
                return self.genome_counts[n]
        return recurse(node)

    def _get_n_nodes(self, node: int, depth: int) -> int:
        """Obtain the number of nodes at a certain depth from a node."""
        def recurse(n: int, d: int):
            count = 0
            edges = self.graph.out_edges(n)
            if d <= 0:
                return len(edges)
            if edges:
                for edge in edges:
                    _, node_index, _ = edge
                    count += recurse(node_index, d - 1)
                return count
            else:
                return 1
        return recurse(node, depth)

    def get_leaves(self, node: int) -> list[int]:
        """Obtain the number of nodes at a certain depth from a node."""
        nodes = []

        def recurse(n: int):
            edges = self.graph.out_edges(n)
            if edges:
                for edge in edges:
                    _, next_index, _ = edge
                    recurse(next_index)
            else:
                return nodes.append(n)

        recurse(node)
        return nodes

    def resolve_taxonomy(self, node: int) -> list[str]:
        """Determine the taxonomy from a node index."""
        index = node
        path = [self.graph[node]]
        while True:
            edges = self.graph.in_edges(index)
            if edges:
                parent, _, _ = edges[0]
                path.append(self.graph[parent])
                index = parent
            else:
                path = list(reversed(path))
                path = path[1:]  # Remove the root.
                return path

    def get_genome_identifiers(self, nodes: list[int]) -> list[str]:
        references = []
        for database in self.genomes:
            for node in nodes:
                if node in self.genomes[database]:
                    identifiers = self.genomes[database][node]
                    path = self.resolve_taxonomy(node)
                    references.append((identifiers, path))
        return references

    def _print_line(self, node, depth, level, width) -> str:
        """Print a line in a taxonomy summary table."""
        column_width = width // depth
        text = self.graph[node]
        count = f" ({len(self.graph.out_edges(node))})"
        if len(text + count) > column_width - 2:
            text = text[:column_width - 6 - len(count)] + "..."
        prefix = "| " + (" " * level * column_width) + text + count
        return prefix + " " * (width - len(prefix)) + "|"

    def _print_section(self, node: int, depth: int, width: int) -> None:
        """Print a section in a taxonomy summary table."""
        column_width = width // depth

        def recurse(n: int, level: int):
            print(self._print_line(n, depth, level, width) + "           |")
            for edge in self.graph.out_edges(n):
                _, node_index, _ = edge
                if level < depth - 2:
                    recurse(node_index, level + 1)
                else:
                    print(
                        self._print_line(
                            node_index, depth, level + 1, width
                        ),
                        end=""
                    )
                    suffix = " " + str(self._get_n_leaves(node_index))
                    print(
                        suffix
                        + " " * (len("| N Genomes |") - len(suffix) - 2)
                        + "|"
                    )

        recurse(node, 0)
        header = "-" * (column_width - 1) + "+"
        print("+" + header * depth + "-----------+")

    def print(
            self,
            path: list[str],
            depth: int = None,
            width: int = 120,
            ) -> None:
        """Print a table that summarizes the taxonomy."""
        root = self.root
        # Find the top-level node.
        for i, field in enumerate(path):
            edges = self.graph.out_edges(root)
            if edges:
                for edge in edges:
                    _, node_index, _ = edge
                    if self.graph[node_index] == field:
                        root = node_index
                        break
                else:
                    rank = Taxonomy.TAXONOMIC_LEVELS[i]
                    raise RuntimeError(f"`{field}` is not a `{rank}`")
            else:
                raise RuntimeError(f"Could not find `{path}`.")
        if path:
            taxon = Taxonomy.TAXONOMIC_LEVELS[len(path) - 1]
            N = self._get_n_leaves(root)
            print(f"Taxonomy within the {taxon} {path[-1]} ({N:_} genomes):")
        taxa = Taxonomy.TAXONOMIC_LEVELS[len(path):len(path) + depth]
        width -= len("| N Genomes |")
        column_width = width // len(taxa)
        header = "=" * (column_width - 1) + "+"
        print("+" + header * len(taxa) + "===========+")
        for i, taxon in enumerate(taxa):
            count = self._get_n_nodes(root, i)
            text = taxon + f" ({count})"
            padding = (column_width - 2) - len(text)
            print("| " + text + " " * padding, end="")
        print("| N Genomes |")
        print("+" + header * len(taxa) + "===========+")
        for edge in self.graph.out_edges(root):
            _, node_index, _ = edge
            self._print_section(node_index, depth, width)


def bin_genomes(
        taxonomy: Taxonomy,
        path: tuple[str] = None,
        depth: int = 1,
        n_minimum: int = 0,
        n_maximum: int = float("inf"),
        verbosity: int = 1
        ) -> dict:
    """Extract collections of reference genomes at particular taxonomic orders.

    Args:
        taxonomy: A Taxonomy object representing the dataset.
        path: Taxonomic path in which to extract reference genomes. For
            instance, `("Bacteria", "Nitrospirota", )` will select the
            reference genomes from the Nitrospirota phylum only. If `None`,
            starts at the root (i.e. above domain rank).
        depth: Number of taxonomic levels that will be binned. For instance,
            if the `path` is `("Bacteria", "Nitrospirota", )` and the value
            of `depth` is `1`, this function will return a dictionary that
            contains the two classes of the Nitrospirota phylum. If the depth
            is `2`, this function will return a dictionary that contains the
            three orders in that phylum, nested in their respective class.
        n_minimum: Inclusive minimum number of reference genomes. Taxa with
            fewer reference genomes are pruned.
        n_maximum: Inclusive maximum number of reference genomes in a single
            species.

    Returns: A dictionary that maps taxonomic levels to lists of reference
        genomes.
    """
    n_total, n_pruned, n_capped, n_selected, n_species = 0, 0, 0, 0, 0

    def recurse(node: int, rank: int):
        nonlocal n_total, n_pruned, n_capped, n_selected, n_species
        label = taxonomy.graph[node]
        # Recursion end condition: the depth is reached, fetch the leaves.
        if rank > depth:
            nodes = taxonomy.get_leaves(node)
            raw_reference_genomes = taxonomy.get_genome_identifiers(nodes)
            n = sum([len(r[0]) for r in raw_reference_genomes])
            n_total += n
            if n < n_minimum:
                if verbosity > 1:
                    print(f"Taxon {label} contains {n} genomes. Dropping.")
                n_pruned += n
                return []
            reference_genomes = []
            for references, path in raw_reference_genomes:
                if len(references) > n_maximum:
                    if verbosity > 1:
                        print(f"Species {path} contains {n} genomes. "
                              + f" Selecting {n_maximum} genomes.")
                    n_capped += len(references) - n_maximum
                    reference_genomes.append(
                        (sample(references, n_maximum), path)
                    )
                    print(path, len(references), len(reference_genomes[-1][0]))
                else:
                    reference_genomes.append((references, path))
            n_species += len(reference_genomes)
            n_selected += sum([len(r[0]) for r in reference_genomes])
            return reference_genomes
        # Recursion
        taxa = {}
        edges = taxonomy.graph.out_edges(node)
        for edge in edges:
            _, node2, _ = edge
            taxa[taxonomy.graph[node2]] = recurse(node2, rank + 1)
            if not taxa[taxonomy.graph[node2]]:
                del taxa[taxonomy.graph[node2]]
        return taxa

    index = taxonomy.find_node(path)
    genomes = recurse(index, 1)
    if verbosity:
        print(f"Detected {n_total} reference genomes.")
        print(f"Detected {n_species} species.")
        print(f"Pruned {n_pruned} reference genomes from low frequency taxa.")
        print(f"Removed {n_capped} reference genomes from large taxa.")
        print(f"Selected {n_selected} reference genomes.")
    return genomes


def group_by_depth(taxonomies: list, depth: int) -> list:
    """Transforms a list of (references, (taxonomy)) into a list of
    identifiers belonging to distinct taxa, as defined by the depth.
    """
    bins = {}
    for reference_genomes, taxon in taxonomies:
        key = tuple(taxon[:-depth])
        if key in bins:
            bins[key] += reference_genomes
        else:
            bins[key] = reference_genomes
    groups = []
    for _, v in bins.items():
        groups.append(v)
    return groups


def non_similar_sets(
        taxa: dict,
        proportions: tuple[float],
        depth: int = 1
        ) -> tuple[dict]:
    """Split an extracted reference genome dictionary into non-overlapping sets
    of reference genomes.

    Args:
        taxa: Collection of reference genomes produced by `extract_genomes`.
        proportions: Approximate proportion of genomes in each set.
        depth: Taxonomic rank at which to consider reference genomes distinct.
            If `1`, species of separate genus will be considered different.

    Returns: Tuple of divided taxa. Each element of the tuple contain different
        reference genomes.

    Raises: RuntimeError if there are not enough genomes.
    """
    assert sum(proportions) == 1.0, "Proportions must sum to one."

    def recurse(sub_dict: dict) -> dict:
        result = {}
        for key, value in sub_dict.items():
            if type(value) is list:
                elements = group_by_depth(value, depth).copy()
                n = len(elements)
                indices = [round(p * n) for p in proportions]
                shuffle(elements)
                bins = []
                start = 0
                for i, index in enumerate(indices):
                    current_bin = []
                    for b in elements[start:start + index]:
                        current_bin += b
                    bins.append(current_bin)
                    start += index
                result[key] = bins
            else:
                result[key] = recurse(value)
        return result

    def recurse_prune(sub_dict: dict, index: int):
        """Remove all lists except the `index` one."""
        result = {}
        for key, value in sub_dict.items():
            if type(value) is list:
                result[key] = value[index]  # Select on list.
            else:
                result[key] = recurse_prune(value, index)
        return result

    split = recurse(taxa)
    bins = []
    for i in range(len(proportions)):
        bins.append(recurse_prune(split.copy(), i))

    return bins


class TaxonomyVector():
    """Represent taxa in a flat vector.

    Attributes:
        reference_genomes: A list containing (taxonomy, identifiers) tuples.
    """

    def __init__(self, taxa: dict):
        self.reference_genomes = []

        def recurse(sub_dict: dict, path: list[str] = []):
            for key, value in sub_dict.items():
                new_path = path + [key]
                if type(value) is list:
                    self.reference_genomes.append([tuple(new_path), value])
                else:
                    recurse(value, new_path)

        recurse(taxa)
        self.depth = len(self.reference_genomes[0][0])

    def get_distance(self, A: int, B: int) -> int:
        distance = 0
        for a, b in zip(
                sorted(self.reference_genomes[A][0], reverse=True),
                sorted(self.reference_genomes[B][0], reverse=True),
                ):
            if a == b:
                break
            else:
                distance += 1
        return distance
