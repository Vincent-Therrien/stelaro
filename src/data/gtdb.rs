//! GTDB data download module.

use log::info;
use std::fs;
use std::io::Error;
use std::path::Path;

use crate::data::download;

const SERVER_ROOT: &str = "https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/";
const TREE_FILES: &'static [&'static str] = &[
    "ar53.tree.gz",   // Archaea reference tree.
    "bac120.tree.gz", // Bacterial reference tree.
];
const TAXONOMY_FILES: &'static [&'static str] = &[
    "ar53_taxonomy.tsv.gz",   // Archaea taxonomy.
    "bac120_taxonomy.tsv.gz", // Bacterial taxonomy.
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
    for filename in TREE_FILES.iter() {
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

/// Download the latest GTDB taxonomy.
/// * `path`: Directory in which to write the taxonomy files.
/// * `force`: If true, download even if the files are already installed.
pub fn install_taxonomy(path: &Path, force: bool) -> Result<(), Error> {
    fs::create_dir_all(path)?;
    for filename in TAXONOMY_FILES.iter() {
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
    info!(
        "The GTDB taxonomy files are installed at `{}`.",
        path.display()
    );
    Ok(())
}
