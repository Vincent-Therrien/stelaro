/// Compress reference genome files into tensor files.

use std::collections::HashMap;
use std::path::Path;
use ndarray::{Array1, Array2};
use std::error::Error;

/// Compress sequence files into vector files.
///
/// This function creates pairs of files named `x_<i>` and `y_<i>`. `x_<i>`
/// contains compressed genomes. `y_<i>` contains the corresponding identifiers.
///
/// * `sequence_file_directory`: Directory that contains the original sequences.
/// * `output_directory`: Directory in which to write results.
/// * `max_file_size`: Maximum size of a compressed file.
/// * `sequences`: Map a file identifier to a numerical identifier. The sequences are compressed
///   in the order in which they are listed.
pub fn database(
    sequence_file_directory: &Path,
    output_directory: &Path,
    max_file_size: usize,
    index: &Vec<(String, u64)>
) -> Result<u32, Box<dyn Error>> {
    Ok(0)
}
