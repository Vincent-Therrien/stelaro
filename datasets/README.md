# Dataset Directory

The files in this directory contain datasets used to train and evaluate neural
networks.

## `version_1.json`

Comprises **genera** organized by **phyla**. Generated with the following code:

```python
T = stelaro.data.Taxonomy(("refseq", ))
urls = get_urls(SUMMARY_DIRECTORY)
T.read_GTDB_file(TAXONOMY_DIRECTORY + "/bac120_taxonomy.tsv", urls)
T.read_GTDB_file(TAXONOMY_DIRECTORY + "/ar53_taxonomy.tsv", urls)
dataset = T.bin_genomes(
    depth=2,  # The dataset will have a resolution at the level of phyla.
    granularity_level=1,  # Data points will be split by genus.
    min_granularity=10,  # The minimum number of genus by data point.
    n_min_reference_genomes_per_bin=50,  # Minimum number of reference genomes by genus.
    n_max_reference_genomes_per_species=3,  # Maximum number of reference genomes for a species.
    max_bin_size=20,  # Maximum number of reference genomes by genus.
    n_max_bins=50  # Maximum number of data points for each phylum.
)
with open('version_1.json', 'w') as f:
    json.dump(dataset, f, indent=4)
```
