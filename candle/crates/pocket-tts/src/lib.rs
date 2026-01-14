pub mod audio;
pub mod conditioners;
pub mod models;
pub mod modules;
pub mod weights;

pub type ModelState =
    std::collections::HashMap<String, std::collections::HashMap<String, candle_core::Tensor>>;
