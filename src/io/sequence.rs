//! Sequence file input / output utility functions.

use std::fs::File;
use std::io::Error;
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
struct FastqQualityError;

/// Convert an ASCII sequence of FASTQ quality scores into integer values.
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

/// Process a section of a FASTQ file that contains:
/// - Identifier
/// - Sequence
/// - `+` sign (ignored)
/// - Quality
fn process_fastq_section(lines: &Vec<String>) -> Result<(String, String, Vec<u8>), Error> {
    const ID_LINE_BEGINNING: char = '@';
    const LINE_SPLIT_BEGINNING: char = '+';
    let mut current_id = String::new();
    let mut current_sequence = String::new();
    let mut current_quality = String::new();
    let mut checking_quality = false;
    for line in lines {
        if line.starts_with(LINE_SPLIT_BEGINNING) && line.len() == 1 {
            checking_quality = true;
        } else if line.starts_with(ID_LINE_BEGINNING) && !checking_quality {
            current_id = line[1..].to_string(); // Remove the '@' character.
        } else if !checking_quality {
            current_sequence.push_str(&line);
        } else {
            current_quality.push_str(&line);
        }
    }
    let quality = match get_fastq_quality(&current_quality) {
        Ok(q) => q,
        Err(_q) => Vec::new(),
    };
    Ok((current_id, current_sequence, quality))
}

/// Read a FASTQ file and return a vector of (ID, sequence, quality) tuples.
/// The function supports multi-line sequences.
pub fn read_fastq(path: &Path) -> io::Result<Vec<(String, String, Vec<u8>)>> {
    const ID_LINE_BEGINNING: char = '@';
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut fastq_data = Vec::new();
    let mut current_lines = Vec::new();
    if let Some(first_line) = lines.next() {
        let first_line = first_line?;
        current_lines.push(first_line);
    }
    for line in lines {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == ID_LINE_BEGINNING {
                let tuple = process_fastq_section(&current_lines).unwrap();
                let (_, ref s, ref q) = tuple;
                if s.len() <= q.len() {
                    fastq_data.push(tuple);
                    current_lines.clear();
                }
            }
        }
        current_lines.push(line);
    }
    fastq_data.push(process_fastq_section(&current_lines).unwrap());
    Ok(fastq_data)
}
