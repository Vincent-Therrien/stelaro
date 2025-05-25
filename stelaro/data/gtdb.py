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
    N_TAXONOMIC_LEVELS = 7

    def __init__(
            self,
            files: list[str],
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

    def _get_line_identifier(self, line: str) -> str:
        identifier = line.split("\t")[0]
        return identifier[len("RS_"):]  # Remove the GTDB-specific prefix.

    def _get_line_taxonomy(self, line: str) -> list[str]:
        line = line.split("\t")[-1]
        fields = line.strip().split(';')
        assert len(fields) == Taxonomy.N_TAXONOMIC_LEVELS, (
            f"Unexpected line: {line}")
        return [f[3:] for f in fields]  # Remove the `d__` prefixes.

    def _add_genome(
            self,
            database: str,
            identifier: str,
            taxonomy: list[str]
        ) -> None:
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
        self.genomes[database][identifier] = index

    def read_file(self, path: str) -> None:
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

    def visualize(self, axis) -> None:
        mpl_draw(
            self.graph,
            with_labels=True,
            labels=str,
            edge_labels=str,
            node_color=(1.0, 1.0, 1.0),
            node_size=500
        )
