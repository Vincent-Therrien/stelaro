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
            current_id = line[1..].to_string(); // Remove the '>' character.
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

#[derive(Debug)]
pub struct FastqQualityError;

/// Convert an ASCII sequence of FASTQ quality score into integer values.
fn get_fastq_quality(quality: &String) -> Result<Vec<u8>, FastqQualityError> {
    const LOWEST_QUALITY: u8 = '!' as u8;
    const HIGHEST_QUALITY: u8 = '~' as u8;
    let mut scores = Vec::new();
    for character in quality.chars() {
        let ascii_value: u8 = character as u8;
        if ascii_value < LOWEST_QUALITY || ascii_value > HIGHEST_QUALITY {
            return Err(FastqQualityError);
        } else {
            scores.push(ascii_value - LOWEST_QUALITY);
        }
    }
    Ok(scores)
}

/// Read a FASTQ file and return a vector of (ID, sequence, quality) tuples.
/// The function supports multi-line sequences.
pub fn read_fastq(path: &Path) -> io::Result<Vec<(String, String, Vec<u8>)>> {
    const ID_LINE_BEGINNING: char = '@';
    const LINE_SPLIT_BEGINNING: char = '+';
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut fastq_data = Vec::new();
    let mut current_id = String::new();
    let mut current_sequence = String::new();
    let mut current_quality = String::new();
    let mut checking_quality = false;
    for line in reader.lines() {
        let line = line?;
        if line.starts_with(LINE_SPLIT_BEGINNING) && line.len() == 1 {
            checking_quality = true;
        } else if line.starts_with(ID_LINE_BEGINNING) && !checking_quality {
            if !current_id.is_empty() {
                match get_fastq_quality(&current_quality) {
                    Ok(q) => {
                        fastq_data.push((current_id.clone(), current_sequence.clone(), q));
                    }
                    Err(_e) => {
                        eprintln!("Error in sequence quality: {:?}", current_id);
                    }
                }
                current_sequence.clear();
                current_quality.clear();
            }
            current_id = line[1..].to_string(); // Remove the '@' character.
        } else if !checking_quality {
            current_sequence.push_str(&line);
        } else {
            current_quality.push_str(&line);
            if current_quality.len() >= current_sequence.len() {
                checking_quality = false;
            }
        }
    }
    // Push the last sequence in the file.
    if !current_id.is_empty() {
        match get_fastq_quality(&current_quality) {
            Ok(q) => {
                fastq_data.push((current_id.clone(), current_sequence.clone(), q));
            }
            Err(_e) => {
                eprintln!("Error in sequence quality: {:?}", current_id);
            }
        }
    }
    Ok(fastq_data)
}
