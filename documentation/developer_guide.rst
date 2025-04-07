Developer Guide
===============

Stelaro is a metagenomic software tool written in Rust with a Python binding.


Organization
------------

The source code is organized as follows:

- `src`: Rust source code.
  - `data`: Download data.
  - `io`: Functions to read and write genome sequence files.
  - `kernels`: Hardware acceleration kernels.
  - `utils`: Utility modules (e.g. console output formatting).

