//! Pocket TTS on ONNX Runtime.
//!
//! [`text`] turns what an agent wrote into what should be said and cuts it
//! into chunks; [`Engine`] turns one chunk into 24 kHz mono samples. Nothing
//! here plays audio or touches the network.

mod model;
pub mod text;

pub use model::{Cancel, ChunkStats, EndReason, Engine, Error, Voice};
