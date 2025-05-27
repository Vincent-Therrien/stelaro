"""GTDB data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2025
    - License: MIT
"""

import os
import rustworkx as rx
import matplotlib.axes as plt
from matplotlib.axes._axes import Axes
import stelaro.stelaro as stelaro_rust


def install_trees(path: str, force: bool = False) -> None:
    """Download and decompress phylogenetic trees.

    Args:
        path: Directory in which to install data.
        force: If `True`, install even if the files are already installed.
    """
    stelaro_rust.install("gtdb", "trees", path, force)


def install_taxonomy(path: str, force: bool = False) -> None:
    """Download and decompress taxonomic information for bacteria and archaea.

    Args:
        path: Directory in which to install data.
        force: If `True`, install even if the files are already installed.
    """
    stelaro_rust.install("gtdb", "taxonomy", path, force)


class Taxonomy():
    """Graph representation of the GTDB taxonomy, with 7 levels. Each node in
    the `graph` corresponds to a taxon. The leaves are species.
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
        """Create a GTDB taxonomy manipulation object.

        Args:
            files: List of GTDB taxonomy files.
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

    def _get_line_identifier(self, line: str) -> str:
        """Get the genome assembly accession ID from a GTDB taxonomy file."""
        identifier = line.split("\t")[0]
        return identifier[len("RS_"):]  # Remove the GTDB-specific prefix.

    def _get_line_taxonomy(self, line: str) -> list[str]:
        """Get the full taxonomy of a line in a GTDB taxonomy file."""
        line = line.split("\t")[-1]
        fields = line.strip().split(';')
        assert len(fields) == len(Taxonomy.TAXONOMIC_LEVELS), (
            f"Unexpected line: {line}")
        return [f[3:] for f in fields]  # Remove the `d__` prefixes.

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
        self.genomes[database][identifier] = index

    def read_file(self, path: str) -> None:
        """Read a GTDB taxonomy file and add its content to the object."""
        with open(path, "r") as f:
            for line in f:
                if line.startswith("RS_"):
                    database = "refseq"
                elif line.startswith("GB_"):
                    database = "genbank"
                if database in self.databases:
                    identifier = self._get_line_identifier(line)
                    taxonomy = self._get_line_taxonomy(line)
                    self._add_genome(database, identifier, taxonomy)

    def extract(
            self,
            path: list[str] = [],
            depth: int = float("inf")
            ) -> rx.PyDiGraph:
        """Extract a subgraph from the taxonomy.

        Returns: Tuple containing (graph, root).
        """
        if not path:
            return self.graph
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

    def visualize(self, axis, G=None) -> None:
        """Visualize the taxonomy graph in a pyplot graph."""
        if G is None:
            G = self.graph
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
            print(f"Taxonomy within the {taxon} {path[-1]}:")
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
