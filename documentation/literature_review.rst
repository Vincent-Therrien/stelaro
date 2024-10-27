Literature Review
=================

Metagenomics tools analyze microbiomes from sequencing data, which makes them useful in biological
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


References
----------

.. bibliography:: references.bib
