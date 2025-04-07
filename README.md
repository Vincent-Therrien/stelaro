# stelaro

Metagenomic toolbox designed for high performances and interpretability.

- Runs as an **executable** that processes metagenomic data through the **command-line**.
- Also runs as a **library** for Rust and Python.
- Built with Rust.
- Accelerated with multithreading and GPUs.

Check the [demonstrations](demonstrations/README.md) out for usage examples.

*This project is at an early stage. Planned elements are marked with checkboxes.*


## Installation


### Full Installation

You can install `stelaro` **with GPU support**, which accelerates the algorithms. This approach is
recommended. Your system needs the OpenCL runtime to build the full project. On Linux, follow this
[guide](https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md)
to install the runtime. On Windows, install the OpenCL runtime from the website of the vendor of
your GPU. After installing the runtime, you may have to place the file `libOpenCL.so` (Linux) or
`OpenCL.lib` (Windows) in the directory of the project to build `stelaro` with GPU support.

Build the full **Rust** library and executable by executing:

```
cargo build --features opencl
```

Install the full **Python** package with [Maturin](https://pypi.org/project/maturin/0.8.2/) by
executing:

```
maturin develop --features opencl
```


### CPU-Only Installation

You can install `stelaro` **without GPU support**. All features will be available, but there will be
no hardware acceleration. This approach is only recommended if your system does not support OpenCL.

Build the CPU-only **Rust** library and executable by executing:

```
cargo build
```

Install the CPU-only **Python** package with [Maturin](https://pypi.org/project/maturin/0.8.2/) by
executing:

```
maturin develop
```


## Planned features

- [ ] Create synthetic datasets.
- [ ] Stream data from large file collections.
- [ ] Process metagenomic data with k-mer and graph-based methods.
- [ ] Visualize results
- [ ] Measure performances
- [ ] Upload the package to PyPI
- [ ] Upload the library to crate.io.


## Organization

- `demonstrations`: Usage examples.
- `documentation`: Documentation for Rust and Python.
- `src`: Rust source code and OpenCL kernels.
- `stelaro`: Python source code.
- `test`: Test cases for Rust and Python.


## Validation Process

Run the following command to validate the project:

```
python3 ci.py
```

This will build all components, run all tests, and validate the coding style.
