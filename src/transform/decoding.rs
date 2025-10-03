//! Transform tensors into nucleotide character sequences.

use ndarray::{Array2, Axis};
use std::error::Error;

pub fn onehot(matrix: Array2<u8>, size: usize) -> Result<String, Box<dyn Error>> {
    let mut sequence = String::with_capacity(size);
    for element in matrix.axis_iter(Axis(0)) {
        let base = match element.as_slice() {
            Some([1, 0, 0, 0]) => 'A',
            Some([0, 1, 0, 0]) => 'C',
            Some([0, 0, 1, 0]) => 'G',
            Some([0, 0, 0, 1]) => 'T',
            _ => 'N',
        };
        sequence.push(base);
    }
    Ok(sequence)
}
