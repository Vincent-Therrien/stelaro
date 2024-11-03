/// Interface functions for the `data` module.
use std::io::Error;
use std::path::Path;

mod download;
mod ncbi;

pub fn install(source: String, name: String, dst: &Path, force: bool) -> Result<(), Error> {
    let _ = match source.as_str() {
        "ncbi" => match name.as_str() {
            "taxonomy" => ncbi::download_taxonomy(dst, force),
            _ => panic!("Unsupported name `{}`.", name),
        },
        _ => panic!("Unsupported source `{}`.", source),
    };
    Ok(())
}
