//! NCBI data downlaod module.
//! `NUCL_TAXID_URLS` refers to the nucleotide taxonomy identifier mappings.

use crate::data::download;
use std::collections::HashMap;
use std::fs;
use std::io::Error;
use std::path::Path;

const FTP_SERVER_ROOT: &str = "https://ftp.ncbi.nih.gov/pub/";
const ACCESSION_2_TAXID_PATH: &str = "taxonomy/accession2taxid/";
const CHECKSUM_SUFFIX: &str = ".md5";

lazy_static! {
    static ref NUCL_TAXID_URLS: HashMap<String, String> = {
        let mut m = HashMap::new();
        m.insert(
            String::from("gb"),
            String::from("nucl_gb.accession2taxid.gz"),
        );
        m.insert(
            String::from("wgs"),
            String::from("nucl_wgs.accession2taxid.gz"),
        );
        m
    };
}

fn is_already_downloaded(local_checksum: &Path, remote_checksum: String) -> bool {
    let local = match fs::read_to_string(local_checksum) {
        Ok(content) => content,
        Err(_) => return false,
    };
    let tmp_filename = format!("{}{}", local_checksum.display(), ".tmp");
    let path = Path::new(&tmp_filename);
    let _ = download::https(&remote_checksum, path);
    let remote = fs::read_to_string(path).unwrap();
    let _ = fs::remove_file(path);
    local == remote
}

/// Download taxonomy from the NCBI
/// * `path` - Directory in which to save the taxonomy.
/// * `force` - If true, download even if the files are already installed.
pub fn download_taxonomy(path: &Path, force: bool) -> Result<(), Error> {
    fs::create_dir_all(path)?;
    const DIR: &str = "ncbi_taxonomy";
    fs::create_dir_all(path.join(DIR))?;
    for (_, filename) in NUCL_TAXID_URLS.iter() {
        let url = format!("{}{}{}", FTP_SERVER_ROOT, ACCESSION_2_TAXID_PATH, filename);
        let checksum_url = format!("{}{}", url, CHECKSUM_SUFFIX);
        let local_path = path.join(DIR).join(filename);
        let local_checksum_path = path
            .join(DIR)
            .join(format!("{}{}", filename, CHECKSUM_SUFFIX));
        if force {
            let _ = download::https(&url, &local_path);
            let decompressed_name = local_path.with_extension("");
            if !Path::new(&decompressed_name).exists() {
                let _ = download::decompress_gz(&local_path, &decompressed_name);
            }
            let _ = download::https(&checksum_url, &local_checksum_path);
        } else {
            if is_already_downloaded(&local_checksum_path, checksum_url.to_string()) {
                println!("The file `{}` is already installed.", filename);
            } else {
                let _ = download::https(&url, &local_path);
                let _ = download::https(&checksum_url, &local_checksum_path);
            }
            let decompressed_name = local_path.with_extension("");
            if !Path::new(&decompressed_name).exists() {
                let _ = download::decompress_gz(&local_path, &decompressed_name);
            }
        }
    }
    Ok(())
}
