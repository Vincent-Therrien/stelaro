Developer Guide
===============

Stelaro is a metagenomic software tool written in Rust that can be used through the command line and
a Python binding.


Data Analysis Pipeline
++++++++++++++++++++++

.. note::

   This pipeline is subject to changes

The pipeline **classifies contigs** into taxonomic groups to profile metagenomic samples and
**identifies annotated sequences** in metagenomic samples such as antimicrobial resistance genes.

.. image:: images/pipeline.svg


+-----------------------+---------------------------------------+----------------------------------+
| Component             | Integration / Implementation          | Progress                         |
+=======================+=======================================+==================================+
| Reference genome      | Rust interface of the NCBI database   | Completed                        |
| database              |                                       |                                  |
+-----------------------+---------------------------------------+----------------------------------+
| Metagenomic reads     | Domain-specific datasets              | Not tested                       |
+-----------------------+---------------------------------------+----------------------------------+
| Taxonomic database    | Rust interface to GTDB                | Not done                         |
+-----------------------+---------------------------------------+----------------------------------+
| Annotation database   | Rust interface to CARD                | Not done                         |
+-----------------------+---------------------------------------+----------------------------------+
| Read simulator        | Rust program                          | Done                             |
+-----------------------+---------------------------------------+----------------------------------+
| Quality control       | External tools (FastQC, Trimmomatic)  | Not integrated                   |
+-----------------------+---------------------------------------+----------------------------------+
| Sequence assembler    | Rust program or external tool         | Not done                         |
+-----------------------+---------------------------------------+----------------------------------+
| Taxonomic profiler    | Rust program or external tool         | Not done                         |
+-----------------------+---------------------------------------+----------------------------------+
| Functional element    | Attention-based neural network        | Not done                         |
| identifier            |                                       |                                  |
+-----------------------+---------------------------------------+----------------------------------+


Input
-----

- **Metagenomic reads** from a sequencing device.
   - List of databases:
      - MarineMetagenomeDB :cite:`nataala2022`
      - MBnify: https://www.ebi.ac.uk/metagenomics
      - MG-RAST: https://www.mg-rast.org/index.html
   - The pipeline itself does not comprise facilities that can trim adapters and discard
     low-quality reads.
   - External tools like Trimmomatic or SolexaQA are to be used to pre-process raw reads.
- **Synthetic reads** generated from reference genomes. Used when training the models. The reference
  genomes are taken from the NCBI.
   - A **read simulator** generates data from taxonomic profiles :cite:`fritz2019`. This is more
     representative of real metagenomic samples than random sampling.
- A **Taxonomy database** to profile the reads such as GTDB or the one of the NCBI.
- A **Sequence annotation database** to identify elements of interest such as antimicrobial
  resistance genes.
   - CARD database :cite:`hackenberger2024`.
   - Greengenes (ribosomal DNA) https://ngdc.cncb.ac.cn/databasecommons/database/id/3120
   - SILVA (ribosomal DNA)
   - Ribosomal Database Project (ribosomal DNA)


Data Processing
---------------

- The **sequence assembler** creates contigs from pre-processed sequence reads.
   - This component uses either De Bruijn graphs or overlap-layout-consensus to assemble the reads.
   - The assembler will be either hardware-accelerated on GPUs or performed by an external tool.
- The **sequence processor** converts the contigs into a compressed format.
   - Techniques used in natural language processing, such as tokenization (BERT), are to be used by
     this component.
   - Conversion into tetra-mers are also considered.
- The **taxonomic binning model** uses processed contigs to assign them to taxonomic profiles.
   - This is currently envisioned as an attention-based neural network, but a rule-based program
     will be used if performances are disappointing.
- The **functional element identifier** finds relevant elements in the genomes, such as AMR genes.
   - This is envisioned as an attention-based neural network.


Output
------

- **Taxonomic profiles** ascribing taxonomic levels to reads.
- **Functional element predictions** ascribing potential functions to reads.


Organization
++++++++++++

The source code is organized as follows:

- `src`: Rust source code.
   - `data`: Download data.
   - `io`: Functions to read and write genome sequence files.
   - `kernels`: Hardware acceleration kernels.
   - `utils`: Utility modules (e.g. console output formatting).


References
----------

.. bibliography:: references.bib
