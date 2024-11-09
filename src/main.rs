/// `stelarilo` is a command-line program based on the `stelaro` library.
use clap::{Parser, Subcommand};
use std::path::Path;

use stelaro::data;

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Install (download and decompress) reference data.
    Install {
        /// Origin of the data to install. Supported origins: ncbi
        #[arg(short, long, required = true)]
        origin: Option<String>,

        /// Directory in which to save the data. A subdirectory is automatically create to store
        /// the data downloaded by this command.
        #[arg(short, long, required = true)]
        dst: Option<String>,

        /// Name of the dataset to install. Supported names:
        /// taxonomy (~32 GB), genome_summaries (~175 MB)
        #[arg(short, long, required = true)]
        name: Option<String>,

        /// Force installation even if the data are already installed.
        #[arg(short, long)]
        force: bool,
    },

    /// Sample reference genome identifiers from a dataset of references and write them in a file.
    SampleGenomes {
        /// Origin of the data to install. Supported origins: ncbi
        #[arg(short, long, required = true)]
        origin: Option<String>,

        /// Directory in which the list of reference genomes is installed.
        #[arg(short, long, required = true)]
        input: Option<String>,

        /// File in which to save the reference genomes.
        #[arg(short, long, required = true)]
        dst: Option<String>,

        /// Indicates how to perform sampling. Supported options:
        /// `full` (sample genomes URLs),
        /// `micro` (sample genomes of archaea, bacteria, fungi, and viruses).
        /// Default: `micro`.
        #[arg(short, long, required = false)]
        sampling: Option<String>,

        /// Fraction of elements to randomly sample. Default: 1.0.
        #[arg(short, long, required = false)]
        fraction: Option<f32>,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Install {
            origin,
            name,
            dst,
            force,
        }) => {
            let origin: String = match origin {
                Some(s) => s.to_string(),
                None => panic!("No origin provided."),
            };
            let name: String = match name {
                Some(n) => n.to_string(),
                None => panic!("No dataset name provided."),
            };
            let dst: &str = match dst {
                Some(d) => d.as_str(),
                None => panic!("No directory provided."),
            };
            let path = Path::new(dst);
            match data::install(origin, name, path, *force) {
                Ok(_) => (),
                Err(error) => panic!("Installation failed: {}", error),
            }
        }
        Some(Commands::SampleGenomes {
            origin,
            input,
            dst,
            sampling,
            fraction,
        }) => {
            let origin: String = match origin {
                Some(s) => s.to_string(),
                None => panic!("No origin provided."),
            };
            let input: &str = match input {
                Some(s) => s.as_str(),
                None => panic!("No input directory provided."),
            };
            let input_path = Path::new(input);
            let dst: &str = match dst {
                Some(d) => d.as_str(),
                None => panic!("No output file provided."),
            };
            let dst_path = Path::new(dst);
            let sampling: &str = match sampling {
                Some(s) => s.as_str(),
                None => "micro",
            };
            let fraction: f32 = match fraction {
                Some(f) => *f,
                None => 1.0,
            };
            match data::sample_genomes(origin, input_path, dst_path, sampling.to_string(), fraction)
            {
                Ok(_) => (),
                Err(error) => panic!("Genome sampling failed: {}", error),
            }
        }
        None => {}
    };
}
