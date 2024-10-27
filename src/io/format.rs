//! Data formatting functions used to transform input data into usable formats.

use std::collections::HashMap;

lazy_static! {
    static ref IUB_IUPAC_nt_code: HashMap<char, u32> = {
        let mut m = HashMap::new();
        m.insert('A', 0b00000000);
        m.insert('A', 0b00000000);
        m.insert('A', 0b00000000);
        m.insert('A', 0b00000000);
        m
    };
}

//pub fn transform_sequence(sequence: String) -> Result<Vec<u8>, String> {}
