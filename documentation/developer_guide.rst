Developer Guide
===============

Stelaro is a metagenomic software tool written in Rust with a Python binding.


Organization
++++++++++++

The source code is organized as follows:

- `src`: Rust source code.
  - `data`: Download data.
  - `io`: Functions to read and write genome sequence files.
  - `kernels`: Hardware acceleration kernels.
  - `utils`: Utility modules (e.g. console output formatting).


Machine Learning Model Draft
++++++++++++++++++++++++++++

The proposed machine learning model aims at **classifying contigs** into taxonomic groups to profile
metagenomic samples and **identify annotated sequences** in metagenomic samples such as
antimicrobial resistance genes.


Components
----------

- The **assembler** creates contigs from reads.
- The **read simulator** generates reads from reference genomes and taxonomic profiles
  :ref:`fritz2019`. Accelerated by GPUs.
- The **sequence pre-processor** converts the raw sequences into a compressed format to accelerate
  training (e.g. tokenization, tetra-mers, ...). Accelerated by GPUs.
- The **taxonomic binning model** uses processed contigs to identify the taxa present in a sample.
- The **element identifier** finds relevant elements in the genomes, such as AMR genes.
