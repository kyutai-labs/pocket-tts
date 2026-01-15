use anyhow::Result;
use clap::Parser;
use pocket_tts::TTSModel;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pocket TTS - Rust/Candle Port")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Generate audio from text
    Generate {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Path to voice reference audio for voice cloning (optional)
        /// Also accepts predefined voice names like 'alba', 'marius', etc.
        #[arg(short, long)]
        voice: Option<String>,

        /// Output file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Generate {
            text,
            voice,
            output,
        } => {
            println!("Loading model...");
            let model = TTSModel::load("b6369a24")?;

            // Predefined voices
            let predefined_voices = [
                "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
            ];

            let voice_state = if let Some(ref v) = voice {
                if predefined_voices.contains(&v.as_str()) {
                    println!("Using predefined stock voice: {}", v);
                    // Try to find in HF cache on D:
                    let cache_path = format!(
                        "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\{}.safetensors",
                        v
                    );
                    let path = std::path::PathBuf::from(cache_path);
                    if path.exists() {
                        model.get_voice_state_from_prompt_file(path)?
                    } else {
                        anyhow::bail!(
                            "Predefined voice '{}' found in config but .safetensors file not found at {:?}",
                            v,
                            path
                        );
                    }
                } else {
                    println!("Using voice cloning from: {:?}", v);
                    model.get_voice_state(v)?
                }
            } else {
                // Default to alba stock voice
                println!("No voice specified, defaulting to stock voice: alba");
                let cache_path = "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\alba.safetensors";
                let path = std::path::PathBuf::from(cache_path);
                if path.exists() {
                    model.get_voice_state_from_prompt_file(path)?
                } else {
                    println!(
                        "Warning: alba.safetensors not found in cache, using empty state (not recommended)"
                    );
                    pocket_tts::voice_state::init_states(1, 1000)
                }
            };

            println!("Generating: \"{}\"", text);
            let audio = model.generate(&text, &voice_state)?;

            let dims = audio.dims();
            println!("Audio shape: {:?}", dims);

            let num_samples = if dims.len() == 2 { dims[1] } else { dims[0] };
            let duration_sec = num_samples as f32 / model.sample_rate as f32;

            println!("Saving to: {:?}", output);
            pocket_tts::audio::write_wav(&output, &audio, model.sample_rate as u32)?;

            println!(
                "Done! Generated {} samples ({:.2}s at {}Hz)",
                num_samples, duration_sec, model.sample_rate
            );
        }
    }

    Ok(())
}
