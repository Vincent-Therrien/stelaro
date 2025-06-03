//! Data transformation module.
use ndarray::{array, Array1, Array2};
use std::collections::HashMap;
use std::error::Error;

mod decoding;
mod encoding;
pub mod kmer;

lazy_static! {
    static ref onehot_nt_code: HashMap<char, Array1<u8>> = {
        let mut m = HashMap::new();
        m.insert('A', array![1, 0, 0, 0]); // Adenosine
        m.insert('a', array![1, 0, 0, 0]); // Adenosine
        m.insert('C', array![0, 1, 0, 0]); // Cytosine
        m.insert('c', array![0, 1, 0, 0]); // Cytosine
        m.insert('G', array![0, 0, 1, 0]); // Guanine
        m.insert('g', array![0, 0, 1, 0]); // Guanine
        m.insert('T', array![0, 0, 0, 1]); // Thymine
        m.insert('t', array![0, 0, 0, 1]); // Thymine
        m.insert('N', array![0, 0, 0, 0]); // Indeterminate
        m.insert('n', array![0, 0, 0, 0]); // Indeterminate
        m
    };
}

pub fn encoding_dimension(name: &str) -> Result<usize, Box<dyn Error>> {
    match name {
        "onehot" => Ok(4),
        _ => Err(format!("Invalid encoding: {}", name).into()),
    }
}

pub fn encode(encoding: String, sequence: &str, size: usize) -> Result<Array2<u8>, Box<dyn Error>> {
    if encoding == "onehot" {
        return Ok(encoding::onehot(sequence, size)?);
    }
    Err("Failed encoding the sequence.".into())
}

pub fn decode(encoding: String, matrix: Array2<u8>, size: usize) -> Result<String, Box<dyn Error>> {
    if encoding == "onehot" {
        return Ok(decoding::onehot(matrix, size)?);
    }
    Err("Failed decoding the sequence.".into())
}
