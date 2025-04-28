use log::info;
use rand::{seq::SliceRandom, Rng};
use std::fs::{remove_file, File};
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::path::Path;

use crate::io::sequence;
use crate::utils::progress;

mod download;
mod gtdb;
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
            _ => panic!("Unsupported name `{}` for source `NCBI`.", name),
        },
        "gtdb" => match name.as_str() {
            "trees" => gtdb::install_trees(dst, force),
            _ => panic!("Unsupported name `{}` for source `GTDB`.", name),
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

/// Read an index file whose rows are formatted as tab-separated values for genome ID, FTP URL,
/// category, and genome size.
/// * `src`: Index file path.
///
/// Returns a tuple containing the genome identifier and genome size.
fn read_index_file(src: &Path) -> Result<Vec<(String, u64)>, Error> {
    let mut index = Vec::new();
    let file = File::open(src)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next(); // Skip the header.
    for line in lines {
        let line = line?;
        let fields = line.split("\t").collect::<Vec<&str>>();
        let identifier: String = fields[0].to_string();
        let genome_size: u64 = fields[3].parse::<u64>().unwrap();
        index.push((identifier, genome_size));
    }
    Ok(index)
}

/// Applies exactly `num_indels` insertions or deletions to the DNA sequence.
/// Each indel is either an insertion (random nucleotides) or a deletion (removing existing nucleotides).
/// `max_indel_size` controls the maximum size of a single insertion or deletion.
fn insert_indels(seq: &str, num_indels: usize, max_indel_size: usize) -> String {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<char> = seq.chars().collect();
    let nucleotides = ['A', 'C', 'G', 'T'];

    for _ in 0..num_indels {
        let pos = rng.gen_range(0..=dna.len());
        let indel_size = rng.gen_range(1..=max_indel_size);
        let is_insertion = rng.gen_bool(0.5);

        if is_insertion {
            // Generate random bases to insert
            let insertion: Vec<char> = (0..indel_size)
                .map(|_| *nucleotides.choose(&mut rng).unwrap())
                .collect();
            dna.splice(pos..pos, insertion);
        } else {
            // Delete bases, but only if enough remain
            let delete_len = indel_size.min(dna.len().saturating_sub(pos));
            if delete_len > 0 {
                dna.drain(pos..pos + delete_len);
            }
        }
    }

    dna.into_iter().collect()
}

/// Simulate one sequence from a reference genome.
/// * `src`: Genome file path.
/// * `genome_size`: Number of base pairs in the genome.
/// * `length`: Number of nucleotides in the sequence ot generate.
/// * `indels`: Number of indels to introduce.
///
/// Returns a tuple formatted as: (sequence, offset).
fn simulate_sequence(
    src: &Path,
    genome_size: u64,
    length: u64,
    indels: u32,
) -> Result<(String, u64), Error> {
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let offset = rng.gen_range(0..genome_size - length);
        let sequence = match sequence::read_fasta_section(src, offset, length) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if indels > 0 {
            return Ok((insert_indels(sequence.as_str(), indels as usize, 4), offset));
        } else {
            return Ok((sequence, offset));
        }
    }
    Err(Error::new(
        ErrorKind::Other,
        "Could not generate a synthetic sequence.",
    ))
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
    let index = read_index_file(index).unwrap();
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
        let i_length: u32 = match length_deviation {
            0 => length,
            _ => rng.gen_range(length - length_deviation..length + length_deviation),
        };
        let i_indels: u32 = match indels_deviation {
            0 => 0,
            _ => rng.gen_range(indels - indels_deviation..indels + indels_deviation),
        };
        let index_row = index.choose(&mut rng).unwrap();
        let (identifier, genome_size) = index_row;
        let genome_filepath = genomes.join(Path::new(&identifier));
        match simulate_sequence(
            genome_filepath.as_path(),
            *genome_size,
            i_length as u64,
            i_indels,
        ) {
            Ok((read, offset)) => {
                let _ = output.write_fmt(format_args!(">{}\t{}\t{}\n", identifier, offset, i));
                let _ = output.write_fmt(format_args!("{}\n\n", read));
                pb.set_position(i as u64);
            }
            Err(_error) => {
                info!(
                    "Failed to generate a read: {} {}",
                    genome_filepath.display(),
                    genome_size
                );
            }
        };
    }
    Ok(())
}
