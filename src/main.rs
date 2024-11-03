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
    /// Install metagenomic data.
    Install {
        /// Data source. Supported sources: ncbi
        #[arg(short, long, required = true)]
        source: Option<String>,

        /// Name of the dataset. Supported names: taxonomy
        #[arg(short, long, required = true)]
        name: Option<String>,

        /// Directory in which to save the data.
        #[arg(short, long, required = true)]
        dst: Option<String>,

        /// Force installation even if the data are already installed.
        #[arg(short, long)]
        force: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Install {
            source,
            name,
            dst,
            force,
        }) => {
            let source: String = match source {
                Some(s) => s.to_string(),
                None => panic!("No source provided."),
            };
            let name: String = match name {
                Some(n) => n.to_string(),
                None => panic!("No dataset name provided."),
            };
            let dst: &str = match dst {
                Some(d) => d.as_str(),
                None => panic!("No file provided."),
            };
            let path = Path::new(dst);
            match data::install(source, name, path, *force) {
                Ok(_) => (),
                Err(error) => panic!("Installation failed: {}", error),
            }
        }
        None => {}
    };
}
