use crate::data;
use crate::io::sequence;
use ndarray::Array3;
use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArray3, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::path::Path;

#[derive(IntoPyObject)]
struct Sequences {
    obj: Vec<(String, String)>,
}

#[pyfunction]
fn read_fasta<'py>(py: Python<'py>, filename: String) -> PyResult<Bound<'py, PyAny>> {
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
fn read_fastq<'py>(py: Python<'py>, filename: String) -> PyResult<Bound<'py, PyAny>> {
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

#[pyfunction]
fn synthetic_metagenome(
    index: String,
    genomes: String,
    dst: String,
    reads: u32,
    length: u32,
    length_deviation: u32,
    indels: u32,
    indels_deviation: u32,
) -> PyResult<()> {
    let index = Path::new(&index);
    let genomes = Path::new(&genomes);
    let dst = Path::new(&dst);
    match data::synthetic_metagenome(
        index,
        genomes,
        dst,
        reads,
        length,
        length_deviation,
        indels,
        indels_deviation,
    ) {
        Ok(_) => Ok(()),
        Err(error) => panic!("Synthetic metagenome generation failed: {}", error),
    }
}

#[pyfunction]
fn synthetic_sample<'py>(
    py: Python<'py>,
    index: String,
    genomes: String,
    reads: u32,
    length: u32,
    length_deviation: u32,
    indels: u32,
    indels_deviation: u32,
    encoding: String,
) -> Bound<'py, PyArray3<f32>> {
    let dimension = length + length_deviation;
    let mut tensor = Array3::<f32>::zeros((reads as usize, dimension as usize, 1));
    // TODO: fill the tensor
    tensor.into_pyarray(py)
}

#[pymodule]
fn stelaro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(read_fastq, m)?)?;
    m.add_function(wrap_pyfunction!(install, m)?)?;
    m.add_function(wrap_pyfunction!(sample_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(install_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_metagenome, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_sample, m)?)?;
    Ok(())
}
