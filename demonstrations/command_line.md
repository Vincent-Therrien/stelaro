# Process Metagenomic Data through the Command-Line

This file explains how to use `stelarilo`, the command-line program of the `stelaro` project, to
process metagenomic data.

**Note**: The executable is referred as `stelarilo` throughout this file. If you use Windows, you'll
have to replace all instances of that word by `stelarilo.exe`.


## Help Menu

You can view the usage of the program by executing the command:

```
stelarilo --help
```


## Commands

The following commands assume that there is a directory with read-write access called `data/`
accessible in the working directory.


### Install Data

The `install` command downloads and decompresses data.


#### Taxonomy

For example, the following instruction downloads the NCBI taxonomy for nucleotide sequences
(~32 GB) into the `data/` directory:

```
mkdir data  # This directory will be used to store the results of the commands.
mkdir data/ncbi_taxonomy
stelarilo install --origin ncbi --name taxonomy --dst data/ncbi_taxonomy
```

This downloads the following files:

- `nucl_gb.accession2taxid`: a list of identifiers for GenBank (curated list).
- `nucl_gb.accession2taxid.gz.md5`: checksum for the GenBank file.
- `nucl_wgs.accession2taxid`: a list of identifiers for Whole Genome Shotgun sequences (less curated list)
- `nucl_wgs.accession2taxid.gz.md5`: checksum for the Whole Genome Shotgun file.

If you execute the previous command again, the program will notice that the taxonomy is already
installed and will not download it another time. To download the taxonomy regardless, you can add
the `--force` option.


#### Genome Summaries

The following command downloads the reference genome summaries from the NCBI (~175 MB):

```
mkdir data/ncbi_genome_summaries
stelarilo install --origin ncbi --name genome_summaries --dst data/ncbi_genome_summaries
```

The command downloads a list of tab-separated value files. Each row corresponds to a genome
assembly.


#### GTDB Tree

The following line downloads phylogenetic trees from the GTDB project (~6 MB).

```
stelarilo install --origin gtdb --name trees --dst data/gtdb_trees
```


### Sample Genomes

The following command reads genome summaries and outputs an **index** of sampled genomes. The index
comprises lines each formatted as the following tuple: `(ID, URL, type)`, where `ID` is a unique
genome identifier, `URL` is the address to download the genome, and `type` is the type of reference
genome (e.g. `bacteria`, `archaea`, `viral`, etc.).

```
mkdir data/sample
stelarilo sample-genomes \
    --origin ncbi \
    --input data/ncbi_genome_summaries \
    --dst data/sample/index.tsv \
    --fraction 0.01
```

The argument `fraction` is the approximate proportion of genomes sampled from the genome summaries.
Given that the NCBI contains 407929 reference genomes as of November 2024, you may want to sample
a subset of genomes to create a more manageable genome database during tests. Using a value of
`0.01` randomly samples around 4080 genomes.


### Install Genomes

Install genomes sampled with the command presented above:

```
stelarilo install-genomes \
    --input data/sample/index.tsv \
    --dst data/sample
```

The files listed in `data/sample/index.tsv` should be be installed in the `data/sample` directory.


### Simulate a Metagenomic Experiment

Simulate a metagenomic sequencing experiment by extracting substrings from the installed genomes.

```
stelarilo synthetic-metagenome \
    --index data/sample/index.tsv \
    --genomes data/sample \
    --dst data/sample/synthetic_dataset.fasta
    --reads 10 \
    --length 100 \
```
