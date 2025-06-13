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
    RANKS = (
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
        assert len(fields) == len(Taxonomy.RANKS), (
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
        """Get the reference genome identifiers for species nodes."""
        references = []
        for database in self.genomes:
            for node in nodes:
                if node in self.genomes[database]:
                    references.append(self.genomes[database][node])
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
                    rank = Taxonomy.RANKS[i]
                    raise RuntimeError(f"`{field}` is not a `{rank}`")
            else:
                raise RuntimeError(f"Could not find `{path}`.")
        if path:
            taxon = Taxonomy.RANKS[len(path) - 1]
            N = self._get_n_leaves(root)
            print(f"Taxonomy within the {taxon} {path[-1]} ({N:_} genomes):")
        taxa = Taxonomy.RANKS[len(path):len(path) + depth]
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

    def get_bins(self, node: int, depth: int):
        """Returns (taxonomy path, leaf nodes) tuples."""
        bins = []

        def recurse(path: str, node: int):
            nonlocal bins
            if len(path) + 1 > depth:
                leaves = self.get_leaves(node)
                bins.append((path, leaves))
            else:
                edges = self.graph.out_edges(node)
                for edge in edges:
                    _, child, _ = edge
                    recurse(path + [self.graph[child]], child)

        recurse([], node)
        return bins

    def bin_genomes(
            self,
            depth: int,
            granularity_level: int,
            min_granularity: int = 1,
            n_min_reference_genomes_per_bin: int = 0,
            n_max_reference_genomes_per_species: int = float('inf'),
            max_bin_size: int = float("inf"),
            n_max_bins: int = float("inf")
            ) -> dict:
        """Extract collections of reference genomes at particular taxonomic
        orders.

        Args:
            depth: Number of taxonomic levels that will be kept distinct. If
                `1`, the genomes will be binned by domain. If `2,` they will
                be binned by domain and phyla.
            granularity_level: The number of taxonomic rank to consider genomes
                unknown. If `0`, the reference genomes will be binned by
                species. If `1`, the reference genomes will be binned by genus.
            min_granularity: Minimum number of bins. For instance, if
                `granularity_level` is `1`, the bins comprising less than 1
                genus will be pruned.
            n_min_reference_genomes_per_bin: Minimum number of reference
                genomes in a bin. Smaller bins are pruned.
            n_max_reference_genomes_per_species: Maximum number of reference
                genomes in a single species to bin at most.
            max_bin_size: Maximum number of reference genomes in a bin.
            n_max_bins: Maximum number of bins.

        Returns: A dictionary that maps taxonomic levels to bins of reference
            genomes.
        """
        bins = []
        n_total, n_pruned, n_capped, n_selected = 0, 0, 0, 0
        n_taxa, n_eliminated_taxa = 0, 0
        granularity_label = Taxonomy.RANKS[-1 - granularity_level]

        def recurse(path: list[str], node: int):
            nonlocal bins, n_total, n_pruned, n_capped, n_selected
            nonlocal n_taxa, n_eliminated_taxa
            if len(path) > depth:  # End
                n_taxa += 1
                min_rank = len(Taxonomy.RANKS) - depth - granularity_level
                bin_ = self.get_bins(node, min_rank)
                reference_genomes = []
                bin_names = []
                n_genomes = 0
                for b in bin_:
                    bin_names.append(b[0][-1])
                    nodes = b[1]
                    species = self.get_genome_identifiers(nodes)
                    n_genomes += sum([len(s) for s in species])
                    reference_genomes.append(species)
                n_total += n_genomes
                # Eliminate small bins.
                if len(bin_) < min_granularity:
                    n_pruned += n_genomes
                    n_eliminated_taxa += 1
                    print(f"Taxon {path} not retained; {len(bin_)}"
                          + f" {granularity_label} in {self.databases}.")
                    return
                # Eliminate bins that contain too few genomes.
                if n_genomes < n_min_reference_genomes_per_bin:
                    n_pruned += n_genomes
                    n_eliminated_taxa += 1
                    print(f"Taxon {path} not retained ({n_genomes} genomes).")
                    return
                # Remove references when there are too many in one species.
                capped_reference_genomes = []
                for bin_ in reference_genomes:
                    new_bin = []
                    for species in bin_:
                        m = len(species)
                        s = n_max_reference_genomes_per_species  # alias
                        if m > s:
                            n_capped += m - s
                            elements = species.copy()
                            shuffle(elements)
                            new_bin += elements[:s]
                        else:
                            new_bin += species
                    n_selected += len(new_bin)
                    capped_reference_genomes.append(new_bin)
                zipped_bin = [
                    (a, b) for a, b in
                    zip(bin_names, capped_reference_genomes)
                ]
                bins.append((path[1:], zipped_bin))
                return
            edges = self.graph.out_edges(node)
            for edge in edges:
                _, child, _ = edge
                recurse(path + [self.graph[child]], child)

        recurse(["root"], self.root)
        print(f"Found {n_taxa} taxa, retained {n_taxa - n_eliminated_taxa}.")
        print(f"Detected {n_total} reference genomes.")
        print(f"Pruned {n_pruned} reference genomes from low frequency taxa.")
        print(f"Removed {n_capped} reference genomes from large species.")
        print(f"Selected {n_selected} reference genomes.")
        capped_bins = []
        n_final = 0
        for path, bin_ in bins:
            # Eliminate large numbers of bins (e.g. remove genus).
            elements = bin_.copy()
            if len(bin_) > n_max_bins:
                shuffle(elements)
                elements = elements[:n_max_bins]
            # Eliminate large numbers of reference genomes (e.g. in genus).
            capped_elements = []
            for bin_name, reference_genomes in elements:
                capped_references = reference_genomes.copy()
                if len(reference_genomes) > max_bin_size:
                    shuffle(capped_references)
                    capped_references = capped_references[:max_bin_size]
                capped_elements.append((bin_name, capped_references))
                n_final += len(capped_references)
            capped_bins.append((path, capped_elements))
        print(f"Filtered {n_final} reference genomes.")
        return capped_bins


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
                result[key] = value[index]
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
                    elements = value.copy()
                    shuffle(elements)
                    self.reference_genomes.append([tuple(new_path), elements])
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

    def clip_size(self, N: int):
        """Reduce the size of each row."""
        for i in range(len(self.reference_genomes)):
            if len(self.reference_genomes[i][1]) > N:
                self.reference_genomes[i][1] = self.reference_genomes[i][1][:N]

    def get_n_genomes(self) -> int:
        total = 0
        for i in range(len(self.reference_genomes)):
            total += len(self.reference_genomes[i][1])
        return total
