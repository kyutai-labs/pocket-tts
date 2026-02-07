//! Server state management

use pocket_tts::{ModelState, TTSModel};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct VoiceStateCache {
    capacity: usize,
    order: VecDeque<String>,
    entries: HashMap<String, Arc<ModelState>>,
}

impl VoiceStateCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<ModelState>> {
        let value = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(value)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn put(&mut self, key: String, value: Arc<ModelState>) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.touch(&key);
            return;
        }

        if self.entries.len() >= self.capacity
            && let Some(oldest) = self.order.pop_front()
        {
            self.entries.remove(&oldest);
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.to_string());
    }
}

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<TTSModel>,
    /// Default voice state (pre-loaded at server start)
    pub default_voice_state: Arc<ModelState>,
    /// LRU cache of resolved voice states for repeated requests.
    pub voice_cache: Arc<StdMutex<VoiceStateCache>>,
    /// Lock to ensure sequential processing of generation requests
    /// (Matching Python's "not thread safe" / single worker behavior)
    pub lock: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(
        model: TTSModel,
        default_voice_state: ModelState,
        voice_cache_capacity: usize,
    ) -> Self {
        Self {
            model: Arc::new(model),
            default_voice_state: Arc::new(default_voice_state),
            voice_cache: Arc::new(StdMutex::new(VoiceStateCache::new(voice_cache_capacity))),
            lock: Arc::new(Mutex::new(())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VoiceStateCache;
    use std::sync::Arc;

    #[test]
    fn voice_state_cache_evicts_lru() {
        let mut cache = VoiceStateCache::new(2);

        cache.put("a".to_string(), Arc::new(Default::default()));
        cache.put("b".to_string(), Arc::new(Default::default()));

        // Touch "a" so "b" becomes LRU.
        let _ = cache.get("a");
        cache.put("c".to_string(), Arc::new(Default::default()));

        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
        assert!(cache.get("b").is_none());
    }
}
