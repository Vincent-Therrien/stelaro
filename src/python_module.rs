use crate::data;
use crate::data::{read_index_file, sample_synthetic_sequence};
use crate::io::sequence;
use crate::transform;
use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::error::Error;
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

fn encode(encoding: String, sequence: &str, size: usize) -> Result<Array2<u8>, Box<dyn Error>> {
    if encoding == "onehot" {
        return Ok(transform::onehot(sequence, size)?);
    }
    Err("Failed encoding the sequence.".into())
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
) -> (Bound<'py, PyArray3<u8>>, Bound<'py, PyList>) {
    let processed_length = length + length_deviation;
    let mut tensor = Array3::<u8>::zeros((reads as usize, processed_length as usize, 4));
    let mut identifiers = Vec::new();
    identifiers.reserve(reads as usize);
    let genomes = Path::new(&genomes);
    let index = Path::new(&index);
    let index = read_index_file(index).unwrap();
    for i in 0..reads {
        loop {
            match sample_synthetic_sequence(
                &index,
                genomes,
                length,
                length_deviation,
                indels,
                indels_deviation,
            ) {
                Ok((sequence, identifier)) => {
                    let matrix = encode(
                        encoding.clone(),
                        sequence.as_str(),
                        processed_length as usize,
                    )
                    .unwrap();
                    tensor.index_axis_mut(Axis(0), i as usize).assign(&matrix);
                    identifiers.push(identifier);
                    break;
                }
                Err(_error) => (),
            };
        }
    }
    let list = PyList::new(py, identifiers).unwrap();
    (tensor.into_pyarray(py), list)
}
/// from stelaro.stelaro import synthetic_sample
/// synthetic_sample("data/classification_dataset/bacteria.tsv", "data/classification_dataset/", 5, 10, 0, 0, 0, "onehot")

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
