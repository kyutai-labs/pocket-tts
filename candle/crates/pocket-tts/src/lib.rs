pub mod audio;
pub mod conditioners;
pub mod config;
pub mod models;
pub mod modules;
pub mod tts_model;
pub mod voice_state;
pub mod weights;

pub use tts_model::TTSModel;

pub type ModelState =
    std::collections::HashMap<String, std::collections::HashMap<String, candle_core::Tensor>>;
