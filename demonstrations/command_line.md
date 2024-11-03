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


### Install Data

The `install` command downloads and decompresses data. For example, the following instruction
downloads the NCBI taxonomy for nucleotide sequences into the `data/` directory:

```
stelarilo install --source ncbi --name taxonomy --dst data
```

If you execute the previous command again, the program will notice that the taxonomy is already
installed and will not download it another time. To download the taxonomy regardless, you can add
the `--force` option.
