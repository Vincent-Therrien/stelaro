//! Data formatting functions used to transform input data into usable formats.

use std::collections::HashMap;

lazy_static! {
    static ref IUPAC_nt_code_full: HashMap<char, u8> = {
        let mut m = HashMap::new();
        m.insert('A', 0b1000); // Adenosine
        m.insert('C', 0b0100); // Cytosine
        m.insert('G', 0b0010); // Guanine
        m.insert('T', 0b0001); // Thymine
        m.insert('U', 0b0001); // Uracil
        m.insert('R', 0b1010); // Purine (A or G)
        m.insert('Y', 0b0101); // Pyrimidine (C, T, or U)
        m.insert('M', 0b1100); // Amino group (A or C)
        m.insert('K', 0b0011); // Keto group (G, T, or U)
        m.insert('S', 0b0110); // Strong interaction (C or G)
        m.insert('W', 0b1001); // Weak interaction (A, T, or U)
        m.insert('H', 0b1101); // Not G (A, T, C, or U)
        m.insert('B', 0b0111); // Not A (A, C, T, or U)
        m.insert('V', 0b1110); // Not T or U (A, C, or G)
        m.insert('C', 0b1011); // Not C (A, G, T, or U)
        m.insert('N', 0b1111); // Any (A, C, G, T, or U)
        m
    };
}

lazy_static! {
    static ref IUPAC_4_nt_code: HashMap<char, u8> = {
        let mut m = HashMap::new();
        m.insert('A', 0b00); // Adenosine
        m.insert('a', 0b00); // Adenosine
        m.insert('C', 0b01); // Cytosine
        m.insert('c', 0b01); // Cytosine
        m.insert('G', 0b10); // Guanine
        m.insert('g', 0b10); // Guanine
        m.insert('T', 0b11); // Thymine
        m.insert('t', 0b11); // Thymine
        m.insert('U', 0b11); // Uracil
        m.insert('u', 0b11); // Uracil
        m
    };
}
