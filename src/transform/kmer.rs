//! Perform operations related to K-mers.

use crate::data::read_index_file;
use crate::io::sequence;
use crate::utils::progress;
use std::collections::HashMap;
use std::path::Path;

fn generate_kmer_map(k: usize) -> HashMap<String, u64> {
    let alphabet = ['A', 'C', 'G', 'T'];
    let mut kmers = HashMap::new();
    generate_kmers_recursive(&mut String::new(), k, &alphabet, &mut kmers);
    kmers
}

fn generate_kmers_recursive(
    prefix: &mut String,
    k: usize,
    alphabet: &[char],
    map: &mut HashMap<String, u64>,
) {
    if prefix.len() == k {
        map.insert(prefix.clone(), 0);
        return;
    }
    for &c in alphabet {
        prefix.push(c);
        generate_kmers_recursive(prefix, k, alphabet, map);
        prefix.pop();
    }
}

/// Count the K-mers in a sequence. Only supports the characters {A, C, G, T}.
/// * `sequence`: Arbitrary sequence in which to count K-mers.
/// * `K`: K-mer size.
///
/// Returns a dictionary of `4**K`` elements mapping a K-mer to its count in the sequence.
pub fn count(sequence: &str, k: usize) -> HashMap<String, u64> {
    let mut kmer_map = generate_kmer_map(k);
    let seq_len = sequence.len();
    for i in 0..=(seq_len - k) {
        let kmer = &sequence[i..i + k];
        if let Some(count) = kmer_map.get_mut(kmer) {
            *count += 1;
        }
    }
    kmer_map
}

fn count_fast(
    sequence: &str,
    k: usize,
    mut kmer_map: HashMap<String, u64>,
) -> HashMap<String, u64> {
    let seq_len = sequence.len();
    for i in 0..=(seq_len - k) {
        let kmer = &sequence[i..i + k];
        if let Some(count) = kmer_map.get_mut(kmer) {
            *count += 1;
        }
    }
    kmer_map
}

/// Combine K-mer counts stored in two dictionaries.
pub fn fuse(dictionary: &mut HashMap<String, u64>, d: &HashMap<String, u64>) {
    for (key, value) in d {
        let counter = dictionary.entry(key.clone()).or_insert(0);
        *counter += value;
    }
}

/// Split a string into vectors that do not contain `N` characters.`
fn split_at_n(input: &String) -> Vec<String> {
    input
        .split('N') // split at each 'N'
        .filter(|s| !s.is_empty()) // ignore empty segments (e.g. between consecutive Ns)
        .map(|s| s.to_string()) // convert &str to String
        .collect()
}

/// Generate a K-mer profile for a set of reference genomes.
/// * `genome_directory`: Directory in which the reference genome files are stored.
/// * `index`: TSV file containing the reference genome IDs.
/// * `k`: Size of the K-mers.
pub fn profile(genome_directory: &Path, index: &Path, k: usize) -> HashMap<String, u64> {
    let index = read_index_file(index).unwrap();
    let mut map = generate_kmer_map(k);
    let progress_bar = progress::new_bar(index.len() as u64);
    for (identifier, _size) in index {
        let genome_filepath = genome_directory.join(Path::new(&identifier));
        let genome_filepath = Path::new(&genome_filepath);
        match sequence::read_fasta(genome_filepath) {
            Ok(sequences) => {
                let kmer_map = generate_kmer_map(k);
                for (_id, sequence) in sequences {
                    let n_free_sequences = split_at_n(&sequence);
                    for partial_sequence in n_free_sequences {
                        let partial_map = kmer_map.clone();
                        let partial_map = count_fast(partial_sequence.as_str(), k, partial_map);
                        fuse(&mut map, &partial_map);
                    }
                }
            }
            Err(_e) => {
                panic!("Did not find the file `{}`.", genome_filepath.display());
            }
        }
        progress_bar.inc(1);
    }
    map
}
