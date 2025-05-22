//! Transform character sequences in tensors.

use crate::transform::onehot_nt_code;
use ndarray::{Array2, Axis};
use std::error::Error;

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
