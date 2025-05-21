//! Data transformation module.
use ndarray::{array, Array1, Array2, Axis};
use std::collections::HashMap;
use std::error::Error;

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

pub fn onehot(sequence: &str, size: usize) -> Result<Array2<u8>, Box<dyn Error>> {
    let mut vector = Array2::<u8>::zeros((size, 4));
    for i in 0..sequence.len() {
        match onehot_nt_code.get(&sequence.chars().nth(i).unwrap()) {
            Some(c) => vector.index_axis_mut(Axis(0), i as usize).assign(c),
            None => return Err(format!("Invalid character at position {}", i).into()),
        };
    }
    Ok(vector)
}
