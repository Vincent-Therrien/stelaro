use indicatif::{ProgressBar, ProgressStyle};

/// Creates a new progress bar with a shared style.
///
/// # Arguments
/// * `total_size` - The total size (in bytes) for the progress bar to track.
pub fn new_bar(total_size: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{elapsed} [{bar:40.magenta}] {bytes} / {total_bytes} ETA: {eta}")
            .unwrap()
            .progress_chars("#=-"),
    );
    pb
}
