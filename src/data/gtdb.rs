//! GTDB data download module.

use log::info;
use std::fs;
use std::io::Error;
use std::path::Path;

use crate::data::download;

const SERVER_ROOT: &str = "https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/";
const TAXONOMY_RELEVANT_FILES: &'static [&'static str] = &[
    "ar53.tree.gz", // Archaea reference tree.
    "ar53_taxonomy.tsv.gz",
    "ar53_metadata.tsv.gz",
    "bac120.tree.gz", // Bacterial reference tree.
];

/// Check if a GTDB file is already downloaded.
fn is_already_downloaded(local_file: &Path, _remote_file: String) -> bool {
    if local_file.exists() {
        return true;
    }
    return false;
}

/// Download the latest GTDB trees.
/// * `path`: Directory in which to write the trees.
/// * `force`: If true, download even if the files are already installed.
pub fn install_trees(path: &Path, force: bool) -> Result<(), Error> {
    fs::create_dir_all(path)?;
    for filename in TAXONOMY_RELEVANT_FILES.iter() {
        let url = format!("{}{}", SERVER_ROOT, filename);
        let local_path = path.join(filename);
        if force {
            let _ = download::https(&url, &local_path, true);
            download::decompress_archive(&local_path);
        } else {
            if is_already_downloaded(&local_path, filename.to_string()) {
                info!("The file `{}` is already installed.", filename);
            } else {
                let _ = download::https(&url, &local_path, true);
                download::decompress_archive(&local_path);
            }
        }
    }
    info!("The GTDB trees are installed at `{}`.", path.display());
    Ok(())
}
