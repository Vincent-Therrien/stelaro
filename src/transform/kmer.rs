//! Perform operations related to K-mers.

use std::collections::HashMap;

fn generate_kmer_map(k: usize) -> HashMap<String, u32> {
    let alphabet = ['A', 'C', 'G', 'T'];
    let mut kmers = HashMap::new();
    generate_kmers_recursive(&mut String::new(), k, &alphabet, &mut kmers);
    kmers
}

fn generate_kmers_recursive(
    prefix: &mut String,
    k: usize,
    alphabet: &[char],
    map: &mut HashMap<String, u32>,
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
pub fn count(sequence: &str, k: usize) -> HashMap<String, u32> {
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
