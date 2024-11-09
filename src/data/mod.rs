/// Interface functions for the `data` module.
use std::io::Error;
use std::path::Path;

mod download;
mod ncbi;

pub fn install(origin: String, name: String, dst: &Path, force: bool) -> Result<(), Error> {
    let _ = match origin.as_str() {
        "ncbi" => match name.as_str() {
            "taxonomy" => ncbi::download_taxonomy(dst, force),
            "genome_summaries" => ncbi::download_genome_summaries(dst, force),
            _ => panic!("Unsupported name `{}`.", name),
        },
        _ => panic!("Unsupported origin `{}`.", origin),
    };
    Ok(())
}

pub fn sample_genomes(
    origin: String,
    src: &Path,
    sampling: String,
    dst: &Path,
    fraction: f32,
) -> Result<(), Error> {
    let _ = match origin.as_str() {
        "ncbi" => match ncbi::sample_genomes(src, sampling, dst, fraction) {
            Ok(_) => (),
            Err(err) => panic!("Error: {err}"),
        },
        _ => panic!("Unsupported origin `{}`.", origin),
    };
    Ok(())
}
