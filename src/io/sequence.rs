//! Sequence file input / output utility functions.

use std::collections::HashSet;
use std::fs::File;
use std::io::Error;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const FASTA_ID_LINE_BEGINNING: char = '>';
const FASTA_COMMENT_LINE_BEGINNING: char = ';';
const FASTQ_ID_LINE_BEGINNING: char = '@';
const FASTQ_LINE_SPLIT_BEGINNING: char = '+';
const NUCLEOTIDE_CHARACTERS: [char; 5] = ['N', 'A', 'T', 'C', 'G'];

// FASTA

/// Process a vector of lines corresponding to one sequence in a FASTA file.
fn process_fasta_section(lines: &Vec<String>) -> Result<(String, String), Error> {
    // TODO: Add a test case
    let mut id = String::new();
    let mut sequence = String::new();
    let mut id_passed = false;
    for line in lines {
        if line.starts_with(FASTA_COMMENT_LINE_BEGINNING) {
            continue;
        }
        if line.starts_with(FASTA_ID_LINE_BEGINNING) {
            id = line[1..].to_string(); // Remove the line beginning character.
            id_passed = true;
        } else if id_passed {
            sequence.push_str(&line);
        }
    }
    Ok((id, sequence))
}

/// Read a FASTA file and return a vector of (ID, sequence) pairs.
pub fn read_fasta(path: &Path) -> io::Result<Vec<(String, String)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut fasta_data = Vec::new();
    let mut current_lines = Vec::new();
    if let Some(first_line) = lines.next() {
        let first_line = first_line?;
        current_lines.push(first_line);
    }
    for line in lines {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == FASTA_ID_LINE_BEGINNING {
                fasta_data.push(process_fasta_section(&current_lines).unwrap());
                current_lines.clear();
            }
        }
        current_lines.push(line);
    }
    fasta_data.push(process_fasta_section(&current_lines).unwrap());
    Ok(fasta_data)
}

/// Read `n` FASTA sequences from the reader.
/// * `reader`: File reader
/// * `n`: Number of sequences to read from the file reader.
pub fn read_fasta_sequences(reader: BufReader<File>, n: u32) -> io::Result<Vec<(String, String)>> {
    let mut lines = reader.lines();
    let mut current_lines = Vec::new();
    let mut fasta_data = Vec::new();
    let mut count: u32 = 0;
    if let Some(first_line) = lines.next() {
        let first_line = first_line?;
        current_lines.push(first_line);
    }
    let mut reached_file_end = true;
    for line in lines {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == FASTA_ID_LINE_BEGINNING {
                fasta_data.push(process_fasta_section(&current_lines).unwrap());
                current_lines.clear();
                count += 1;
                if count >= n {
                    reached_file_end = false;
                    break;
                }
            }
        }
        current_lines.push(line);
    }
    if reached_file_end {
        fasta_data.push(process_fasta_section(&current_lines).unwrap());
    }
    Ok(fasta_data)
}

/// Read a sequence in a FASTA file starting at the specified number of bytes.
/// * `path`: FASTA file path.
/// * `i`: Offset.
/// * `n`: Number of bytes to read.
pub fn read_fasta_section(path: &Path, i: u64, n: u64) -> io::Result<String> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(i + 1))?;
    // Read a few more bytes than specified to account for newline and carriage return characters.
    let n_bytes_to_read = (n as f64 * 1.05).round() as usize;
    let mut buffer = vec![0; n_bytes_to_read];
    let bytes_read = file.read(&mut buffer)?;
    let result = String::from_utf8_lossy(&buffer[..bytes_read])
        .to_string()
        .replace('\n', "")
        .replace('\r', "");
    let allowed: HashSet<char> = NUCLEOTIDE_CHARACTERS.into_iter().collect();
    if result.chars().any(|c| !allowed.contains(&c)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Encountered an ID.",
        ));
    }
    Ok(result.chars().take(n as usize).collect())
}

/// Count the number of sequences stored in a FASTA file.
pub fn count_fasta_sequences(path: &Path) -> io::Result<u32> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut count: u32 = 0;
    for line in reader.lines() {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == FASTA_ID_LINE_BEGINNING {
                count += 1;
            }
        }
    }
    Ok(count)
}

// FASTQ

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
    let mut current_id = String::new();
    let mut current_sequence = String::new();
    let mut current_quality = String::new();
    let mut checking_quality = false;
    for line in lines {
        if line.starts_with(FASTQ_LINE_SPLIT_BEGINNING) && line.len() == 1 {
            checking_quality = true;
        } else if line.starts_with(FASTQ_ID_LINE_BEGINNING) && !checking_quality {
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
            if first_char == FASTQ_ID_LINE_BEGINNING {
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

/// Read FASTQ sequences from the reader until `n` sequences are read.
pub fn read_fastq_section(
    reader: BufReader<File>,
    n: u32,
) -> io::Result<Vec<(String, String, Vec<u8>)>> {
    let mut lines = reader.lines();
    let mut current_lines = Vec::new();
    let mut fastq_data = Vec::new();
    let mut count: u32 = 0;
    if let Some(first_line) = lines.next() {
        let first_line = first_line?;
        current_lines.push(first_line);
    }
    let mut reached_file_end = true;
    for line in lines {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == FASTQ_ID_LINE_BEGINNING {
                let tuple = process_fastq_section(&current_lines).unwrap();
                let (_, ref s, ref q) = tuple;
                if s.len() <= q.len() {
                    fastq_data.push(tuple);
                    current_lines.clear();
                    count += 1;
                    if count >= n {
                        reached_file_end = false;
                        break;
                    }
                }
            }
        }
        current_lines.push(line);
    }
    if reached_file_end {
        fastq_data.push(process_fastq_section(&current_lines).unwrap());
    }
    Ok(fastq_data)
}

/// Count the number of sequences in a FASTQ file.
pub fn count_fastq_sequences(path: &Path) -> io::Result<u32> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut count: u32 = 0;
    for line in reader.lines() {
        let line = line?;
        if let Some(first_char) = line.chars().next() {
            if first_char == FASTQ_LINE_SPLIT_BEGINNING {
                count += 1;
            }
        }
    }
    Ok(count)
}
