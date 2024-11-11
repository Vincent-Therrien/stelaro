use crate::data;
use crate::io::sequence;
use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::path::Path;

#[pyfunction]
/// Todo: replace `PyList::new_bound` by `PyList::new` in pyo3 version 0.23.0.
fn read_fasta<'py>(py: Python<'py>, filename: String) -> PyResult<PyObject> {
    let path = Path::new(&filename);
    let result = sequence::read_fasta(path);
    match result {
        Ok(value) => {
            let py_list = PyList::new_bound(
                py,
                value
                    .iter()
                    .map(|(s1, s2)| PyTuple::new_bound(py, &[s1.as_str(), s2.as_str()])),
            );
            return Ok(py_list.to_object(py));
        }
        Err(_e) => {
            panic!("Did not find the file `{}`.", filename);
        }
    }
}

#[pyfunction]
/// Todo: replace `PyList::new_bound` by `PyList::new` in pyo3 version 0.23.0.
fn read_fastq<'py>(py: Python<'py>, filename: String) -> PyResult<PyObject> {
    let path = Path::new(&filename);
    let result = sequence::read_fastq(path);
    match result {
        Ok(value) => {
            let py_list = PyList::empty_bound(py);
            for (id, seq, quality) in value {
                let integers: Vec<i32> = quality.iter().map(|&x| x as i32).collect();
                let py_tuple = (id, seq, integers).to_object(py);
                py_list.append(py_tuple)?;
            }
            return Ok(py_list.to_object(py));
        }
        Err(_e) => {
            panic!("Did not find the file `{}`.", filename);
        }
    }
}

// TODO: Remove this function after the binding is more developed.
fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// TODO: Remove this function after the binding is more developed.
#[pyfunction]
fn axb<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = axpy(a, x, y);
    z.into_pyarray_bound(py)
}

#[pyfunction]
fn install(origin: String, name: String, dst: String, force: bool) -> PyResult<()> {
    let path = Path::new(&dst);
    match data::install(origin, name, path, force) {
        Ok(_) => Ok(()),
        Err(error) => panic!("Installation failed: {}", error),
    }
}

#[pyfunction]
fn sample_genomes(
    origin: String,
    src: String,
    dst: String,
    sampling: String,
    fraction: f32,
) -> PyResult<()> {
    let input = Path::new(&src);
    let output = Path::new(&dst);
    match data::sample_genomes(origin, input, output, sampling, fraction) {
        Ok(_) => Ok(()),
        Err(error) => panic!("Sampling failed: {}", error),
    }
}

#[pyfunction]
fn install_genomes(input: String, dst: String) -> PyResult<()> {
    let input = Path::new(&input);
    let dst = Path::new(&dst);
    match data::install_genomes(input, dst, true) {
        Ok(_) => Ok(()),
        Err(error) => panic!("Genome installation failed: {}", error),
    }
}

#[pymodule]
fn stelaro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(read_fastq, m)?)?;
    m.add_function(wrap_pyfunction!(install, m)?)?;
    m.add_function(wrap_pyfunction!(sample_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(install_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(axb, m)?)?;
    Ok(())
}
