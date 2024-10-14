# Stelaro Test Suite

This directory contains the test suite of the project. It comprises two sub-directories:

- `rust` contains test cases for the Rust side of stelaro (i.e. the low-level heavy lifting). The
  file `tests.rs` imports the test modules so that the command `cargo test` can execute the test
  cases of this subdirectory.
- `python` comprises test cases for the Python side of stelaro (i.e. high-level operations for
  more convenient use). These tests are written with `pytest`. To execute them, first install the
  `stelaro` library with the command `stelaro$ maturin develop` and the test dependencies with the
  command `stelaro$ pip install requirements-dev.txt`. You can then execute the test suite
  by running `stelaro$ pytest tests`.
