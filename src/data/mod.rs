use log::info;
use rand::Rng;
use rand::prelude::*;
use std::fs::{remove_file, File};
use std::io::{BufRead, BufReader, Error, Write, ErrorKind};
use std::path::Path;

use crate::utils::progress;
use crate::io::sequence;

mod download;
mod ncbi;

/// Install a dataset from an authority source.
/// * `origin`: Name of the data authority.
/// * `name`: Name of the dataset to install.
/// * `dst`: Name of the directory in which to save the dataset.
/// * `force`: If `true`, install the dataset even if it is already installed.
pub fn install(origin: String, name: String, dst: &Path, force: bool) -> Result<(), Error> {
    let _ = match origin.as_str() {
        "ncbi" => match name.as_str() {
            "taxonomy" => ncbi::download_taxonomy(dst, force),
            "genome_summaries" => ncbi::download_genome_summaries(dst, force),
            _ => panic!("Unsupported name `{}`.", name),
        },
        _ => panic!("Unsupported origin `{}`.", origin),
    };
    Ok(())
}

/// Sample a list of genome IDs and URLs from a dataset and write it in a destination file.
/// * `origin`: Name of the data authority.
/// * `src`: Directory that contains the installed genome summaries.
/// * `dst`: File in which to save the genome index. Each line of the output file is formatted as:
///   `<genome file name><tab><genome download URL><tab><optional informative fields>`
///   The genome file name acts as a unique ID.
/// * `sampling`: Sampling mode. Either `full` or `micro`.
/// * `fraction`: Fraction of genomes to sampling among the entire list of genomes.
pub fn sample_genomes(
    origin: String,
    src: &Path,
    dst: &Path,
    sampling: String,
    fraction: f32,
) -> Result<(), Error> {
    let _ = match origin.as_str() {
        "ncbi" => match ncbi::sample_genomes(src, dst, sampling, fraction) {
            Ok(_) => (),
            Err(err) => panic!("Error: {err}"),
        },
        _ => panic!("Unsupported origin `{}`.", origin),
    };
    Ok(())
}

/// Install a set of genomes listed in a file.
/// * `index`: Index file that contains the genome IDs and URLs, on each line, separated by tabs.
/// * `dst`: Name of the directory in which to save the genomes (one per file).
/// * `force`: If `true`, install genomes even if they are already installed.
pub fn install_genomes(index: &Path, dst: &Path, force: bool) -> Result<(), Error> {
    const ID_COLUMN: usize = 0;
    const URL_COLUMN: usize = 1;
    // Count lines.
    let f = File::open(index)?;
    let reader = BufReader::new(f);
    let line_count = reader.lines().count() - 1;
    // Read the file
    let index = File::open(index)?;
    let reader = BufReader::new(index);
    let mut lines = reader.lines();
    lines.next(); // Skip the header.
    let pb = progress::new_bar(line_count as u64);
    for (i, line) in lines.enumerate() {
        let line = line?;
        let elements = line.split("\t").collect::<Vec<&str>>();
        let url = elements[URL_COLUMN];
        let id = elements[ID_COLUMN];
        let download_path = dst.join(format!("{}{}", id, "_tmp"));
        let install_path = dst.join(id);
        if force || !install_path.exists() {
            let _ = match download::https(url, &download_path, false) {
                Ok(_) => (),
                Err(_) => info!("Failed to download {}.", download_path.display()),
            };
            let _ = download::decompress_gz(&download_path, &install_path, false);
            let _ = remove_file(download_path);
        }
        pb.set_position(i as u64);
    }
    Ok(())
}

fn read_index_file(src: &Path) -> Result<Vec<String>, Error> {
    let mut genomes = Vec::new();
    let file = File::open(src)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next(); // Skip the header.
    for line in lines {
        let line = line?;
        let identifier: String = line.split("\t").collect::<Vec<&str>>()[0].to_string();
        genomes.push(identifier);
    }
    Ok(genomes)
}

/// Simulate one sequence from a reference genome.
/// * `src`: Genome file path.
/// * `length`: Number of nucleotides in the sequence ot generate.
/// * `indels`: Number of indels to introduce.
///
/// Returns a tuple formatted as: (sequence identifier, offset, sequence).
fn simulate_sequence(src: &Path, length: usize, indels: u32) -> Result<(String, u64, String), Error> {
    let n_sequences = sequence::count_fasta_sequences(src)?;
    let mut sequence_indices: Vec<u32> = (0..n_sequences).collect();
    sequence_indices.shuffle(&mut thread_rng());
    let mut rng = rand::thread_rng();
    for index in sequence_indices {
        let (sequence_id, current_sequence) = sequence::read_sequence_at(src, index)?;
        let len = current_sequence.len();
        if len > length {
            let offset: usize = rng.gen_range(0..len - length);
            let mut synthetic_sequence = (&current_sequence[offset..offset + length]).to_string();
            if indels > 0 {
                // add indels
            }
            return Ok((sequence_id, offset as u64, synthetic_sequence))
        }
    }
    Err(Error::new(ErrorKind::Other, "Could not generate a synthetic sequence."))
}

/// Simulate a metagenomic experiment by randomly sampling sequences from a set of genomes. The
/// resulting file is in the FASTA format whose reads are formatted as follows:
///
/// ---
/// ><genome_file_name><tab><original sequence><tab><offset><tab><indels>
/// <sequence>
///
/// ---
///
/// Where `genome_file_name` identifies the source genome for the sequence, `original sequence` is
/// the name of the sequence in the reference genome used to sample the synthetic read, `offset` is
/// the number of nucleotides skipped at the start of the reference sequence, and `indels` is the
/// number of introduced indels.
///
/// Arguments
///
/// * `index`: Index file that contains the list of genomes to sample. Each line must be formatted
///   as: `<genome file name><tab><genome download URL><tab><optional informative fields>`.
/// * `genomes`: Directory that contains the installed genomes.
/// * `dst`: File in which to save the metagenomic simulation. The sequences are saved in the FASTA
///   format. Each identifier is formatted as: `<genome ID> <start index> <end index>`.
/// * `reads`: Number of sequences to generate.
/// * `length`: Average length of a generated sequence.
/// * `length_deviation`: Maximum deviation for the number of nucleotides in a generated sequence.
/// * `indels`: Average number of indels in a generated sequence.
/// * `indels_deviation`: Maximum deviation for the number of indels in a generated sequence.
pub fn synthetic_metagenome(
    index: &Path,
    genomes: &Path,
    dst: &Path,
    reads: u32,
    length: u32,
    length_deviation: u32,
    indels: u32,
    indels_deviation: u32,
) -> Result<(), Error> {
    let genome_identifiers = read_index_file(index).unwrap();
    let mut rng = rand::thread_rng();
    let pb = progress::new_bar(reads as u64);
    let dst_repr = dst.to_str().unwrap().to_string();
    let mut output = match File::create(dst) {
        Ok(f) => f,
        Err(_error) => {
            panic!("Failed to create file '{}'", dst_repr)
        }
    };
    for i in 0..reads {
        let i_length: u32 = rng.gen_range(length - length_deviation..length + length_deviation);
        let i_indels: u32 = rng.gen_range(indels - indels_deviation..indels + indels_deviation);
        let genome_identifier = genome_identifiers.choose(&mut rng).unwrap();
        let genome_filename = Path::new(&genome_identifier);
        let genome = genomes.join(genome_filename);
        let (sequence_identifier, offset, read) = match simulate_sequence(genome.as_path(), i_length as usize, i_indels) {
            Ok(s) => s,
            Err(_error) => {
                info!("Failed to generate a read.");
                (String::new(), 0, String::new())
            }
        };
        let _ = output.write_fmt(format_args!(">{}\t{}\t{}\t{}\n", genome_identifier, sequence_identifier, offset, i));
        let _ = output.write_fmt(format_args!("{}\n\n", read));
        pb.set_position(i as u64);
    }
    Ok(())
}
