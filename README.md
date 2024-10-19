# stelaro

Genomics toolbox designed for interpretability and high performances.

The project is at an early stage. More detailed descriptions of the planned features will be added.


## Architecture

- A Rust-based backend, which provides rapid, multi-threaded operations.
- A Python frontend, which enables users to call the backend with an intuitive API.
- OpenCL-based hardware acceleration, which can execute parallelizable algorithms on GPUs.


## Planned features

- [ ] Stream data from large file collections.
- [ ] Process metagenomic data with k-mer and graph-based methods.
- [ ] Visualize results
- [ ] Measure performances


## Organization

- `src` contains the Rust source code and OpenCL kernels.
- `stelaro` contains the Python source code.
- `test` contains the test cases.
- `docs` contains the documentation (written with Sphinx).
