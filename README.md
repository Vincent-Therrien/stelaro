# stelaro

Metagenomic toolbox.


## Installation

Build the **Rust** library and executable by executing:

```
cargo build
```

Install the CPU-only **Python** package with [Maturin](https://pypi.org/project/maturin/0.8.2/) by
executing:

```
maturin develop
```


## Organization

- `demonstrations`: Usage examples.
- `documentation`: Documentation for Rust and Python.
- `scripts`: model benchmarking scripts.
- `src`: Rust source code and OpenCL kernels.
- `stelaro`: Python source code.
- `test`: Test cases for Rust and Python.


## Comparison with Other Methods

Check out the [demonstrations](demonstrations/README.md) to see how input data are obtained and how
to train and test models.
