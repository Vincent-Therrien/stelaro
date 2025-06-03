#[cfg(test)]
mod transform {
    use std::collections::HashMap;
    #[test]
    fn test_kmer_counting() {
        let seq = "GTAGAGCTGT";
        let kmer_1 = stelaro::transform::kmer::count(seq, 1);
        let mut expected = HashMap::new();
        expected.insert(String::from("A"), 2);
        expected.insert(String::from("C"), 1);
        expected.insert(String::from("G"), 4);
        expected.insert(String::from("T"), 3);
        assert_eq!(kmer_1, expected);
        let kmer_2 = stelaro::transform::kmer::count(seq, 2);
        let mut expected = HashMap::new();
        expected.insert(String::from("AA"), 0);
        expected.insert(String::from("AC"), 0);
        expected.insert(String::from("AG"), 2);
        expected.insert(String::from("AT"), 0);
        expected.insert(String::from("CA"), 0);
        expected.insert(String::from("CC"), 0);
        expected.insert(String::from("CG"), 0);
        expected.insert(String::from("CT"), 1);
        expected.insert(String::from("GA"), 1);
        expected.insert(String::from("GC"), 1);
        expected.insert(String::from("GG"), 0);
        expected.insert(String::from("GT"), 2);
        expected.insert(String::from("TA"), 1);
        expected.insert(String::from("TC"), 0);
        expected.insert(String::from("TG"), 1);
        expected.insert(String::from("TT"), 0);
        assert_eq!(kmer_2, expected);
    }
}
