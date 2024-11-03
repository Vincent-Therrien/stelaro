use reqwest::blocking::Client;
use reqwest::header::CONTENT_LENGTH;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::utils::progress;

pub fn https(url: &str, dst: &Path) -> Result<(), Box<dyn Error>> {
    let client = Client::new();
    let mut response = client.get(url).send()?.error_for_status()?;
    let total_size = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|len| len.to_str().ok()?.parse::<u64>().ok())
        .unwrap_or(0);
    println!("Downloading `{}`.", url);
    let pb = progress::new_bar(total_size);
    let dst_repr = dst.to_str().unwrap().to_string();
    let mut file = File::create(dst).or(Err(format!("Failed to create file '{}'", dst_repr)))?;
    let mut buffer = [0; 8192];
    let mut downloaded: u64 = 0;
    while let Ok(bytes_read) = response.read(&mut buffer) {
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }
    Ok(())
}
