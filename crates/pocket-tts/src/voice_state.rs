//! Voice state management for streaming generation and voice cloning

use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Model state type for stateful modules
pub type ModelState = HashMap<String, HashMap<String, Tensor>>;

/// Initialize empty model state for all stateful modules
///
/// Creates a nested HashMap structure that will be populated
/// as modules run their forward passes.
pub fn init_states(_batch_size: usize, _seq_len: usize) -> ModelState {
    // Start with empty state - modules will populate as needed
    HashMap::new()
}

/// Get or create a module's state entry
pub fn get_or_create_state<'a>(
    state: &'a mut ModelState,
    module_name: &str,
) -> &'a mut HashMap<String, Tensor> {
    state.entry(module_name.to_string()).or_default()
}

/// Increment step counters in model state for all modules
///
/// This is used after processing tokens to update position information
/// for streaming generation.
pub fn increment_steps(state: &mut ModelState, key: &str, increment: usize) {
    for (_module_name, module_state) in state.iter_mut() {
        if let Some(step_tensor) = module_state.get_mut(key)
            && let Ok(current) = step_tensor.to_scalar::<i64>()
            && let Ok(new_tensor) = Tensor::new(current + increment as i64, step_tensor.device())
        {
            *step_tensor = new_tensor;
        }
    }
}

/// Get the current step/offset for a module
pub fn get_offset(state: &ModelState, module_name: &str) -> usize {
    state
        .get(module_name)
        .and_then(|s| s.get("offset"))
        .and_then(|t| t.to_scalar::<i64>().ok())
        .unwrap_or(0) as usize
}

/// Set the offset for a module
pub fn set_offset(state: &mut ModelState, module_name: &str, offset: usize) -> Result<()> {
    let module_state = get_or_create_state(state, module_name);
    let device = module_state
        .values()
        .next()
        .map(|t| t.device().clone())
        .unwrap_or(candle_core::Device::Cpu);
    module_state.insert("offset".to_string(), Tensor::new(offset as i64, &device)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_states() {
        let state = init_states(1, 100);
        assert!(state.is_empty());
    }

    #[test]
    fn test_get_or_create_state() {
        let mut state = init_states(1, 100);
        let module_state = get_or_create_state(&mut state, "test_module");
        assert!(module_state.is_empty());
        assert!(state.contains_key("test_module"));
    }

    #[test]
    fn test_offset_operations() -> Result<()> {
        let mut state = init_states(1, 100);

        // Initially offset is 0
        assert_eq!(get_offset(&state, "test"), 0);

        // Set offset
        set_offset(&mut state, "test", 42)?;
        assert_eq!(get_offset(&state, "test"), 42);

        Ok(())
    }
}
