Literature Review
=================

Metagenomic tools analyze microbiomes from sequencing data, which makes them useful in biological
research to detect pathogens, study antimicrobial resistance, and predict illnesses, among others.
Although not yet used in clinical settings, many researchers have developed models that can process
metagenomic data with increasing efficacy :ref:`roy2024:`.

Selective amplification (e.g. 16S, 18S, ITS) of specific regions of microbial genomes have been
widely used in metagenomics studies, but they introduce bias and omit elements during analysis
:ref:`mcintyre2017`. Shotgun sequencing is thus becoming a more reliable way to study microbiomes
for a variety of tasks:

- Classification
- Abundance estimation
- Identification


Classification
--------------

Several methods can classify sequences of a sample into taxa:

- **Alignment** of sequences
- **Composition** with k-mer analysis
- **Phylogenetics** using models of sequence evolution


GPU Acceleration
----------------

- 2012 homology search: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036060
- 2012 analysis: https://link.springer.com/article/10.1186/1752-0509-6-S1-S16
- 2021 sequence assembly: https://dl.acm.org/doi/abs/10.1145/3458817.3476212
- 2021 classification: https://dl.acm.org/doi/abs/10.1145/3472456.3472460


References
----------

.. bibliography:: references.bib
