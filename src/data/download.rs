use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use reqwest::header::CONTENT_LENGTH;
use std::error::Error;
use std::fs::File;
use std::io::{self, copy, BufReader, Read, Write};
use std::path::Path;

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

pub fn decompress_gz(src: &Path, dst: &Path, display_progress: bool) -> io::Result<()> {
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
    loop {
        let bytes_read = buffered_reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        copy(&mut &buffer[..bytes_read], &mut decompressed_file)?;
        if pb.is_some() {
            pb.as_ref().unwrap().inc(bytes_read as u64);
        }
    }
    Ok(())
}
