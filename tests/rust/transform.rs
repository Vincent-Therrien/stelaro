#[cfg(test)]
mod transform {
    use std::collections::HashMap;
    #[test]
    fn kmer_counting() {
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

    #[test]
    fn kmer_dictionary_fusion() {
        let mut kmer1 = HashMap::new();
        kmer1.insert(String::from("AA"), 0);
        kmer1.insert(String::from("AC"), 0);
        kmer1.insert(String::from("AG"), 2);
        kmer1.insert(String::from("AT"), 0);
        kmer1.insert(String::from("CA"), 0);
        kmer1.insert(String::from("CC"), 0);
        kmer1.insert(String::from("CG"), 0);
        kmer1.insert(String::from("CT"), 1);
        kmer1.insert(String::from("GA"), 1);
        kmer1.insert(String::from("GC"), 1);
        kmer1.insert(String::from("GG"), 0);
        kmer1.insert(String::from("GT"), 2);
        kmer1.insert(String::from("TA"), 1);
        kmer1.insert(String::from("TC"), 0);
        kmer1.insert(String::from("TG"), 1);
        kmer1.insert(String::from("TT"), 0);
        let kmer2 = kmer1.clone();
        stelaro::transform::kmer::fuse(&mut kmer1, &kmer2);
        for (key, value) in &kmer2 {
            assert_eq!(value * 2, *kmer1.entry(key.clone()).or_insert(0));
        }
    }
}
