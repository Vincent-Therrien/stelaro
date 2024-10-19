//! Sequence file input / output utility functions.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Read a FASTA file and return a vector of (ID, sequence) pairs.
pub fn read_fasta(path: &Path) -> io::Result<Vec<(String, String)>> {
    const ID_LINE_BEGINNING: char = '>';
    const COMMENT_LINE_BEGINNING: char = ';';
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut fasta_data = Vec::new();
    let mut current_id = String::new();
    let mut current_sequence = String::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with(COMMENT_LINE_BEGINNING) {
            continue;
        }
        if line.starts_with(ID_LINE_BEGINNING) {
            if !current_id.is_empty() {
                fasta_data.push((current_id.clone(), current_sequence.clone()));
                current_sequence.clear();
            }
            current_id = line[1..].to_string(); // Remove the '>' character
        } else {
            current_sequence.push_str(&line);
        }
    }

    // Push the last sequence in the file.
    if !current_id.is_empty() {
        fasta_data.push((current_id, current_sequence));
    }

    Ok(fasta_data)
}
