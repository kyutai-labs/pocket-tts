use pocket_tts::config::load_config;
use std::path::PathBuf;

fn main() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates
    path.pop(); // candle
    let config_path = path.join("pocket_tts").join("config").join("b6369a24.yaml");
    println!("Loading config from {:?}", config_path);
    let config = load_config(&config_path).unwrap();
    println!("Ratios: {:?}", config.mimi.seanet.ratios);
}
