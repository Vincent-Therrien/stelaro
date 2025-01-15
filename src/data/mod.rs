use log::info;
use rand::Rng;
/// Interface functions for the `data` module.
use std::fs::{remove_file, File};
use std::io::{BufRead, BufReader, Error, Write};
use std::path::Path;

// use crate::io;
use crate::utils::progress;

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

/// Sample a list of genome IDs and URLs from a dataset.
/// * `origin`: Name of the data authority.
/// * `src`: Directory that contains the installed genome summaries.
/// * `dst`: File in which to save the genome index. Each line of the output file is formatted as:
///   `<genome file name><tab><genome download URL><tab><optional informative fields>`
///   THe genome file name acts as a unique ID.
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

/// Simulate one sequence from a reference genome.
/// * `src`: Genome file path.
/// * `length`: Number of nucleotides in the sequence ot generate.
/// * `indels`: Number of indels to introduce.
fn simulate_sequence(src: &Path, length: u32, indels: u32) -> Result<(u64, String), Error> {
    let mut sequence = String::new();
    Ok((0, sequence))
}

/// Simulate a metagenomic experiment by randomly sampling sequences from a set of genomes. The
/// resulting file is in the FASTA format whose reads are formatted as follows:
///
/// ---
/// ><genome_file_name><tab><offset><tab><indels>
/// <sequence>
///
/// ---
///
/// Where `genome_file_name` identifies the source genome for the sequence, `offset` is the index of
/// the nucleotide in the reference genome at which the sampled sequence starts, and `indels` is the
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
    let mut rng = rand::thread_rng();
    let pb = progress::new_bar(reads as u64);
    let dst_repr = dst.to_str().unwrap().to_string();
    let mut output = match File::create(dst) {
        Ok(f) => f,
        Err(error) => {
            panic!("Failed to create file '{}'", dst_repr)
        }
    };
    for i in 0..reads {
        let n: u32 = rng.gen_range(length - length_deviation..length + length_deviation);
        let i: u32 = rng.gen_range(indels - indels_deviation..indels + indels_deviation);
        let genome_file_name = ""; // TODO
        let genome = Path::new(genome_file_name);
        let (offset, read) = match simulate_sequence(genome, n, i) {
            Ok(i) => i,
            Err(error) => {
                info!("Failed to generate a read.");
                (0, String::new())
            }
        };
        let _ = output.write_fmt(format_args!(">{}\t{}\t{}\n", genome_file_name, offset, i));
        let _ = output.write_fmt(format_args!("{}\n\n", read));
        pb.inc(1);
    }
    Ok(())
}
