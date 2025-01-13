use log::info;
// use rand::Rng;
/// Interface functions for the `data` module.
use std::fs::{remove_file, File};
use std::io::Error;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::io;
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

// /// Simulate one sequence from a reference genome.
// fn simulate_sequence(src: &Path, range: (u64, u64)) -> Result<String, Error> {
//     Ok(())
// }

// /// Simulate a metagenomic experiment by randomly sampling sequences from a set of genomes.
// /// * `index`: Index file that contains the list of genomes to sample. Each line must be formatted
// ///   as: `<genome file name><tab><genome download URL><tab><optional informative fields>`.
// /// * `src`: Directory that contains the installed genomes.
// /// * `dst`: File in which to save the metagenomic simulation. The sequences are saved in the FASTA
// ///   format. Each identifier is formatted as: `<genome ID> <start index> <end index>`.
// /// * `range`: Minimum and maximum lengths of the simulated metagenomic sequences. The length of
// ///   each sequence is determined with a uniform probability distribution.
// /// * `n_sequences`: Number of simulated sequences to generate from the genomes.
// /// * `indel`: Minimum and maximum number of indels to randomly add in each simulated sequence.
// ///   If no value is provided, no indels are added.
// pub fn synthetic_metagenome(
//     index: &Path,
//     src: &Path,
//     dst: &Path,
//     n_sequences: u64,
//     range: (u64, u64),
//     indels: Option<(u64, u64)>,
// ) -> Result<(), Error> {
//     let mut rng = rand::thread_rng();
//     let delta = match range {
//         (a, b) => {
//             if a >= b {
//                 panic!("The first element of `range` must be less than the second element.");
//             } else {
//                 b - a
//             }
//         }
//     };
//     let pb = progress::new_bar(n_sequences as u64);
//     for i in 0..n_sequences {
//         pb.inc(1);
//     }
//     Ok(())
// }
