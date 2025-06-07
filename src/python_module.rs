use crate::data;
use crate::data::{read_index_file, sample_synthetic_sequence};
use crate::io::sequence;
use crate::transform;
use ndarray::{Array3, Axis};
use numpy::{IntoPyArray, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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
) -> (Bound<'py, PyArray3<u8>>, Bound<'py, PyList>) {
    let processed_length = length + length_deviation;
    let dimension = transform::encoding_dimension(&encoding).unwrap();
    let mut tensor = Array3::<u8>::zeros((reads as usize, processed_length as usize, dimension));
    let mut identifiers = Vec::new();
    identifiers.reserve(reads as usize);
    let genomes = Path::new(&genomes);
    let index = Path::new(&index);
    let index = read_index_file(index).unwrap();
    for i in 0..reads {
        let n_attempts = 5;
        for attempt in 1..=n_attempts {
            match sample_synthetic_sequence(
                &index,
                genomes,
                length,
                length_deviation,
                indels,
                indels_deviation,
            ) {
                Ok((sequence, identifier)) => {
                    let matrix = transform::encode(
                        encoding.clone(),
                        sequence.as_str(),
                        processed_length as usize,
                    )
                    .unwrap();
                    tensor.index_axis_mut(Axis(0), i as usize).assign(&matrix);
                    identifiers.push(identifier);
                    break;
                }
                Err(_error) => {
                    if attempt == n_attempts {
                        identifiers.push(format!("ERROR").to_string());
                    }
                }
            };
        }
    }
    let list = PyList::new(py, identifiers).unwrap();
    (tensor.into_pyarray(py), list)
}

#[pyfunction]
fn decode<'py>(
    py: Python<'py>,
    tensor: Bound<'py, PyArray3<u8>>,
    encoding: String,
) -> Bound<'py, PyList> {
    let mut sequences = Vec::new();
    let size = tensor.len().unwrap();
    sequences.reserve(size);
    let tensor: Array3<u8> = tensor.to_owned_array();
    for i in 0..size {
        match transform::decode(
            encoding.clone(),
            tensor.index_axis(Axis(0), i).to_owned(),
            size,
        ) {
            Ok(sequence) => sequences.push(sequence),
            Err(_) => sequences.push("ERROR".to_string()),
        }
    }
    let sequences = PyList::new(py, sequences).unwrap();
    sequences
}

#[pyfunction]
fn profile_kmer<'py>(
    py: Python<'py>,
    index: String,
    directory: String,
    k: u32,
) -> Bound<'py, PyDict> {
    let index = Path::new(&index);
    let directory = Path::new(&directory);
    let map = transform::kmer::profile(directory, index, k.clone() as usize);
    let dict = PyDict::new(py);
    for (kmer, count) in map {
        let _ = dict.set_item(kmer, count);
    }
    dict
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
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(profile_kmer, m)?)?;
    Ok(())
}
