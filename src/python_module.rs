use crate::data;
use crate::io::sequence;
use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::path::Path;

#[derive(IntoPyObject)]
struct Sequences {
    obj: Vec<(String, String)>,
}

#[pyfunction]
fn read_fasta<'py>(py: Python<'py>, filename: String) -> PyResult<Bound<'_, PyAny>> {
    let path = Path::new(&filename);
    let result = sequence::read_fasta(path);
    match result {
        Ok(value) => {
            let list = Sequences {
                obj: value.iter().cloned().collect(),
            }
            .into_pyobject(py)?;
            Ok(list.into_any())
        }
        Err(_e) => {
            panic!("Did not find the file `{}`.", filename);
        }
    }
}

#[derive(IntoPyObject)]
struct QSequences {
    obj: Vec<(String, String, Vec<u8>)>,
}

#[pyfunction]
fn read_fastq<'py>(py: Python<'py>, filename: String) -> PyResult<Bound<'_, PyAny>> {
    let path = Path::new(&filename);
    let result = sequence::read_fastq(path);
    match result {
        Ok(value) => {
            let list = QSequences {
                obj: value.iter().cloned().collect(),
            }
            .into_pyobject(py)?;
            Ok(list.into_any())
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
    z.into_pyarray(py)
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
