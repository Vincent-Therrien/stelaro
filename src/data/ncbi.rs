//! NCBI data downlaod module.
//! `NUCL_TAXID_URLS` refers to the nucleotide taxonomy identifier mappings.

use log::info;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Error;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::data::download;

const SERVER_ROOT: &str = "https://ftp.ncbi.nih.gov/";
const ACCESSION_2_TAXID_PATH: &str = "pub/taxonomy/accession2taxid/";
const REFERENCE_GENOME_PATH: &str = "genomes/refseq/";
const REFERENCE_GENOME_LIST: &'static [&'static str] = &[
    "archaea",
    "bacteria",
    "fungi",
    "invertebrate",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
    "viral",
];
const MICRO_REFERENCE_GENOME_LIST: &'static [&'static str] =
    &["archaea", "bacteria", "fungi", "protozoa", "viral"];
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

fn decompress_archive(local_path: &Path) {
    let decompressed_name = local_path.with_extension("");
    if !Path::new(&decompressed_name).exists() {
        let _ = download::decompress_gz(&local_path, &decompressed_name);
    } else {
        info!(
            "The file `{}` is already decompressed.",
            decompressed_name.display()
        );
    }
    match fs::remove_file(local_path) {
        Ok(_) => info!("Deleted the archive: {}", local_path.display()),
        Err(_) => (),
    }
}

/// Download taxonomy from the NCBI
/// * `path` - Directory in which to save the taxonomy.
/// * `force` - If true, download even if the files are already installed.
pub fn download_taxonomy(path: &Path, force: bool) -> Result<(), Error> {
    fs::create_dir_all(path)?;
    for (_, filename) in NUCL_TAXID_URLS.iter() {
        let url = format!("{}{}{}", SERVER_ROOT, ACCESSION_2_TAXID_PATH, filename);
        let checksum_url = format!("{}{}", url, CHECKSUM_SUFFIX);
        let local_path = path.join(filename);
        let local_checksum_path = path.join(format!("{}{}", filename, CHECKSUM_SUFFIX));
        if force {
            let _ = download::https(&url, &local_path);
            decompress_archive(&local_path);
        } else {
            if is_already_downloaded(&local_checksum_path, checksum_url.to_string()) {
                info!("The file `{}` is already installed.", filename);
            } else {
                let _ = download::https(&url, &local_path);
                let _ = download::https(&checksum_url, &local_checksum_path);
            }
            decompress_archive(&local_path);
        }
    }
    info!("The taxonomy is installed at `{}`.", path.display());
    Ok(())
}

/// Download the assembly summaries of reference genomes.
/// * `path` - Directory in which to save the reference genome summaries.
/// * `force` - If true, download even if the files are already installed.
pub fn download_genome_summaries(path: &Path, force: bool) -> Result<(), Error> {
    fs::create_dir_all(path)?;
    const FILENAME: &str = "assembly_summary.txt";
    const EXTENSION: &str = ".txt";
    for genome_summary in REFERENCE_GENOME_LIST {
        let filename = format!("{}{}", genome_summary, EXTENSION);
        let local_path = path.join(filename);
        let exists = match fs::read_to_string(local_path.clone()) {
            Ok(_) => true,
            Err(_) => false,
        };
        if exists && !force {
            info!("The file `{}` is already installed.", genome_summary);
            continue;
        } else {
            let url = format!(
                "{}{}{}/{}",
                SERVER_ROOT, REFERENCE_GENOME_PATH, genome_summary, FILENAME
            );
            let _ = download::https(&url, &local_path);
        }
    }
    info!("Genome summaries are installed at `{}`.", path.display());
    Ok(())
}

pub fn sample_genomes(
    src: &Path,
    sampling: String,
    dst: &Path,
    fraction: f32,
) -> Result<(), Error> {
    const ID_COLUMN: usize = 0;
    const URL_COLUMN: usize = 19;
    let categories = match sampling.as_str() {
        "micro" => MICRO_REFERENCE_GENOME_LIST,
        "full" => REFERENCE_GENOME_LIST,
        _ => panic!("Unrecognized sampling: {}", sampling),
    };
    let dst_repr = dst.to_str().unwrap().to_string();
    let mut file = match File::create(dst).or(Err(format!("Failed to create file '{}'", dst_repr)))
    {
        Ok(f) => f,
        Err(_) => panic!("Cannot create file: {}", dst.display()),
    };
    let mut count: u32 = 0;
    let mut rng = rand::thread_rng();
    let _ = file.write(format!("ID\tURL\tcategory\n").as_bytes());
    for category in categories {
        let src_path = src.join(format!("{}.txt", category));
        let input = match File::open(src_path.clone()) {
            Ok(f) => f,
            Err(_) => panic!("Cannot open the file: {}", src_path.clone().display()),
        };
        let mut lines = BufReader::new(input).lines();
        lines.next(); // Skip the first comment line.
        lines.next(); // SKip the header.
        for line in lines {
            if rng.gen::<f32>() > fraction {
                continue;
            }
            let line = line?;
            let elements = line.split("\t").collect::<Vec<&str>>();
            let _ = file.write(
                format!(
                    "{}\t{}\t{}\n",
                    elements[ID_COLUMN], elements[URL_COLUMN], category
                )
                .as_bytes(),
            );
            count += 1;
        }
    }
    info!("Wrote {} lines.", count);
    Ok(())
}
