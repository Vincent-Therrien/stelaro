use flate2::read::GzDecoder;
use log::info;
use reqwest::blocking::Client;
use reqwest::header::CONTENT_LENGTH;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{self, copy, BufReader, Read, Write};
use std::path::Path;
use tar::Archive;

use crate::utils::progress;

const BUFFER_SIZE: usize = 8192;

pub fn https(url: &str, dst: &Path, display_progress: bool) -> Result<(), Box<dyn Error>> {
    let client = Client::new();
    let mut response = client.get(url).send()?.error_for_status()?;
    let total_size = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|len| len.to_str().ok()?.parse::<u64>().ok())
        .unwrap_or(0);
    if display_progress {
        println!("Downloading `{}`.", url);
    }
    let pb = match display_progress {
        true => Some(progress::new_bar(total_size)),
        false => None,
    };
    let dst_repr = dst.to_str().unwrap().to_string();
    let mut file = File::create(dst).or(Err(format!("Failed to create file '{}'", dst_repr)))?;
    let mut buffer = [0; BUFFER_SIZE];
    let mut downloaded: u64 = 0;
    while let Ok(bytes_read) = response.read(&mut buffer) {
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        if pb.is_some() {
            pb.as_ref().unwrap().set_position(downloaded);
        }
    }
    Ok(())
}

pub fn decompress_gz(src: &Path, dst: &Path, display_progress: bool) -> io::Result<u64> {
    let gz_file = File::open(src)?;
    let mut decompressed_file = File::create(dst)?;
    let total_size = gz_file.metadata()?.len();
    let mut decoder = GzDecoder::new(gz_file);
    let src_repr = dst.to_str().unwrap().to_string();
    if display_progress {
        println!("Decompressing `{}`.", src_repr);
    }
    let pb = match display_progress {
        true => Some(progress::new_bar(total_size)),
        false => None,
    };
    let mut buffered_reader = BufReader::new(&mut decoder);
    let mut buffer = [0; BUFFER_SIZE];
    let mut total_n_bytes: u64 = 0;
    loop {
        let bytes_read = buffered_reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        total_n_bytes += bytes_read as u64;
        copy(&mut &buffer[..bytes_read], &mut decompressed_file)?;
        if pb.is_some() {
            pb.as_ref().unwrap().inc(bytes_read as u64);
        }
    }
    Ok(total_n_bytes)
}

pub fn decompress_tar(tar_path: &Path, output_dir: &Path) -> io::Result<()> {
    let file = File::open(tar_path)?;
    let mut archive = Archive::new(file);
    archive.unpack(output_dir)?;
    match fs::remove_file(tar_path) {
        Ok(_) => info!("Deleted the archive: {}", tar_path.display()),
        Err(_) => info!("Failed to delete the archive: {}", tar_path.display()),
    }
    Ok(())
}

pub fn decompress_archive(local_path: &Path) {
    let decompressed_name = local_path.with_extension("");
    if !Path::new(&decompressed_name).exists() {
        let _ = decompress_gz(&local_path, &decompressed_name, false);
    } else {
        info!(
            "The file `{}` is already decompressed.",
            decompressed_name.display()
        );
    }
    match fs::remove_file(local_path) {
        Ok(_) => info!("Deleted the archive: {}", local_path.display()),
        Err(_) => info!("Failed to delete the archive: {}", local_path.display()),
    }
}
