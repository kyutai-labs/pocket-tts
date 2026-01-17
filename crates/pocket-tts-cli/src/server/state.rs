//! Server state management

use pocket_tts::{ModelState, TTSModel};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<TTSModel>,
    /// Default voice state (pre-loaded at server start)
    pub default_voice_state: Arc<ModelState>,
    /// Lock to ensure sequential processing of generation requests
    /// (Matching Python's "not thread safe" / single worker behavior)
    pub lock: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(model: TTSModel, default_voice_state: ModelState) -> Self {
        Self {
            model: Arc::new(model),
            default_voice_state: Arc::new(default_voice_state),
            lock: Arc::new(Mutex::new(())),
        }
    }
}
