# stelaro

Genomics toolbox designed for high performances and interpretability. Based on Rust, compatible
with Python, and accelerated by multithreading and GPUs.

This project is at an early stage. Planned elements are marked with checkboxes ([ ]).


## Installation

Install the project in a Python virtual environment with
[Maturin](https://pypi.org/project/maturin/0.8.2/) by executing one of the following commands:

```
maturin develop --features opencl  # CPU and GPU installation.
maturin develop  # CPU-only installation.
```

- [ ] To do: Upload the package to PyPI

Build the project as a Rust library with one of the following commands:

```
cargo build --features opencl  # CPU and GPU installation.
cargo build  # CPU-only library.
```

- [ ] To do: Upload the library to crate.io.


## Planned features

- [ ] Create synthetic datasets.
- [ ] Stream data from large file collections.
- [ ] Process metagenomic data with k-mer and graph-based methods.
- [ ] Visualize results
- [ ] Measure performances


## Organization

- `demonstrations`: Usage examples with Jupyter notebooks.
- `documentation`: Documentation for Rust and Python. Written with Sphinx.
- `src`: **Rust** source code and **OpenCL** kernels.
- `stelaro`: **Python** source code.
- `test`: Test cases for Rust and Python.


## Validation Process

Run the following command to validate the project:

```
python3 ci.py
```

This will build all components, run all tests, and validate the coding style.
