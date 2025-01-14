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

        /// Directory in which to save the data.
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

    /// Install genomes listed in a file.
    InstallGenomes {
        /// Name of the file that contains the URLs of the genomes.
        #[arg(short, long, required = true)]
        input: Option<String>,

        /// Directory in which to save the data.
        #[arg(short, long, required = true)]
        dst: Option<String>,

        /// Force installation even if the data are already installed.
        #[arg(short, long)]
        force: bool,
    },

    /// Create a simulated metagenomic sample dataset.
    SyntheticMetagenome {
        /// Path to the index file that contains the URLs of the genomes.
        #[arg(short, long, required = true)]
        index: Option<String>,

        /// Name of the directory that contains the installed genomes.
        #[arg(short, long, required = true)]
        genomes: Option<String>,

        /// Name of the file in which the synthetic metagenome is saved.
        #[arg(short, long, required = true)]
        dst: Option<String>,

        /// Number of reads to generate.
        #[arg(short, long, required = true)]
        reads: Option<i32>,

        /// Average length of a synthetic read.
        #[arg(short, long, required = true)]
        length: Option<i32>,

        /// Maximum deviation for the length of a synthetic read. Default: 0.
        #[arg(long, required = false)]
        length_deviation: Option<i32>,

        /// Average number of indels in a synthetic read. Default: 0.
        #[arg(long, required = false)]
        indels: Option<i32>,

        /// Maximum deviation for the number of indels in a synthetic read. Default: 0.
        #[arg(long, required = false)]
        indels_deviation: Option<i32>,
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
        Some(Commands::InstallGenomes { input, dst, force }) => {
            let input: &str = match input {
                Some(i) => i.as_str(),
                None => panic!("No input directory provided."),
            };
            let input = Path::new(input);
            let dst: &str = match dst {
                Some(d) => d.as_str(),
                None => panic!("No output file provided."),
            };
            let dst = Path::new(dst);
            match data::install_genomes(input, dst, *force) {
                Ok(_) => (),
                Err(error) => panic!("Genome installation failed: {}", error),
            }
        }
        Some(Commands::SyntheticMetagenome {
            index,
            genomes,
            dst,
            reads,
            length,
            length_deviation,
            indels,
            indels_deviation,
        }) => {
            let index: &str = match index {
                Some(i) => i.as_str(),
                None => panic!("No source directory provided."),
            };
            let index = Path::new(index);
            let genomes: &str = match genomes {
                Some(i) => i.as_str(),
                None => panic!("No genome directory provided."),
            };
            let genomes = Path::new(genomes);
            let dst: &str = match dst {
                Some(d) => d.as_str(),
                None => panic!("No output directory provided."),
            };
            let dst = Path::new(dst);
            let reads: u32 = match reads {
                Some(i) => {
                    if *i > 0 {
                        *i as u32
                    } else {
                        panic!("The number of reads must be positive and non-zero.")
                    }
                }
                None => panic!("No number of reads provided."),
            };
            let length: u32 = match length {
                Some(i) => {
                    if *i > 0 {
                        *i as u32
                    } else {
                        panic!("The length must be positive and non-zero.")
                    }
                }
                None => panic!("No length provided."),
            };
            let length_deviation: u32 = match length_deviation {
                Some(i) => {
                    if *i > 0 && *i < length as i32 {
                        *i as u32
                    } else {
                        panic!("The length deviation must be less than the length and positive.")
                    }
                }
                None => 0,
            };
            let indels: u32 = match indels {
                Some(i) => {
                    if *i > 0 {
                        *i as u32
                    } else {
                        panic!("The number of indels must be positive and non-zero.")
                    }
                }
                None => 0,
            };
            let indels_deviation: u32 = match indels_deviation {
                Some(i) => {
                    if *i > 0 && *i < indels as i32 {
                        *i as u32
                    } else {
                        panic!("The number of indels deviation must be less than the number of indels and positive.")
                    }
                }
                None => 0,
            };
            match data::synthetic_metagenome(
                index,
                genomes,
                dst,
                reads,
                length,
                length_deviation,
                indels,
                indels_deviation,
            ) {
                Ok(_) => (),
                Err(error) => panic!("Genome installation failed: {}", error),
            }
        }
        None => {}
    };
}
