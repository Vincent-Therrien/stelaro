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
                let (_, first_sequence) = &value[0];
                assert_eq!(first_sequence, "AACCGGTTAACCGGTT");
                let (_, second_sequence) = &value[1];
                assert_eq!(second_sequence, "AACCGGTT");
            }
            Err(_e) => {
                panic!("Did not find the file.");
            }
        }
    }

    #[test]
    fn fastq() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/nucleotide_sequence.fastq");
        let result = stelaro::io::sequence::read_fastq(path.as_path());
        match result {
            Ok(value) => {
                assert_eq!(value.len(), 3);
                let (_, first_sequence, first_quality) = &value[0];
                println!("S: {}", first_sequence);
                assert_eq!(first_sequence, "AACCGGTTAACCGGTT");
                assert_eq!(
                    *first_quality,
                    vec![0, 0, 0, 93, 93, 93, 93, 93, 0, 0, 0, 93, 93, 93, 93, 93]
                );
                let (_, second_sequence, second_quality) = &value[1];
                assert_eq!(second_sequence, "TTAACCGG");
                assert_eq!(*second_quality, vec![5, 6, 31, 4, 14, 13, 13, 15]);
                let (_, third_sequence, third_quality) = &value[2];
                assert_eq!(third_sequence, "AACC");
                assert_eq!(*third_quality, vec![0, 0, 0, 0]);
            }
            Err(_e) => {
                panic!("Did not find the file.");
            }
        }
    }
}
