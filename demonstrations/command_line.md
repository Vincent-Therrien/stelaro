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

The `install` command downloads and decompresses data. For example, the following instruction
downloads the NCBI taxonomy for nucleotide sequences (~32 GB) into the `data/` directory:

```
mkdir data/ncbi_taxonomy
stelarilo install --source ncbi --name taxonomy --dst ncbi_taxonomy/data
```

If you execute the previous command again, the program will notice that the taxonomy is already
installed and will not download it another time. To download the taxonomy regardless, you can add
the `--force` option.

The following command downloads the reference genome summaries from the NCBI (~175 MB):

```
mkdir data/ncbi_genome_summaries
stelarilo install --source ncbi --name genome_summaries --dst data/ncbi_genome_summaries
```


## Sample Genomes

The following command reads genome summaries and outputs an **index** of sampled genomes. The index
comprises lines each formatted as the following tuple: `(ID, URL, type)`, where `ID` is a unique
genome identifier, `URL` is the address to download the genome, and `type` is the type of reference
genome (e.g. `bacteria`, `archaea`, `viral`, etc.).

```
mkdir data/sample
stelarilo.exe sample-genomes \
    --origin ncbi \
    --input data\ncbi_genome_summaries \
    --dst data/sample/index.tsv \
    --fraction 0.01
```

The argument `fraction` is the approximate amount of genomes sampled from the genome summaries.
Given that the NCBI contains 407929 reference genomes as of November 2024, you may want to sample
a subset of genomes to create a more manageable genome database during tests. Using a value of
`0.01` randomly samples around 407 genomes.
