use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Generate audio (placeholder)
    Generate {
        #[arg(short, long)]
        text: String,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Pocket TTS CLI (Candle Port)");

    match args.command {
        Some(Commands::Generate { text }) => {
            println!("Generating: {}", text);
        }
        None => {
            println!("Use --help for usage");
        }
    }

    Ok(())
}
