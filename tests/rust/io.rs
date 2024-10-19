#[cfg(test)]
mod io {
    use std::path::PathBuf;
    #[test]
    fn fasta() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/nucleotide_sequence.fasta");
        let result = stelaro::io::sequence::read_fasta(path.as_path());
        match result {
            Ok(value) => {
                assert_eq!(value.len(), 2);
                for (id, seq) in &value {
                    println!("Key: {}, Value: {}", id, seq);
                }
            }
            Err(_e) => {
                panic!("Did not find the file.");
            }
        }
    }
}
