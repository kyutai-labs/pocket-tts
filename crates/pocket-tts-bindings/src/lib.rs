use pocket_tts::{ModelState, TTSModel};
use pyo3::prelude::*;
use std::path::Path;

/// Python wrapper for the Rust TTSModel
#[pyclass]
struct PyTTSModel {
    inner: TTSModel,
}

#[pymethods]
impl PyTTSModel {
    /// Load the model from a specific checkpoint variant
    #[staticmethod]
    fn load(variant: &str) -> PyResult<Self> {
        let model = TTSModel::load(variant)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTTSModel { inner: model })
    }

    /// Load the model with custom parameters
    #[staticmethod]
    fn load_with_params(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> PyResult<Self> {
        let model = TTSModel::load_with_params(variant, temp, lsd_decode_steps, eos_threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTTSModel { inner: model })
    }

    /// Generate audio from text
    ///
    /// Returns:
    ///     One-dimensional list of floats representing the audio samples.
    #[pyo3(signature = (text, valid_voice_state=None))]
    fn generate(&self, text: &str, valid_voice_state: Option<&str>) -> PyResult<Vec<f32>> {
        // Create a default voice state or load one if provided
        // Ideally we should expose a VoiceState object to Python too, but for now
        // let's just make it simple or require a path to a voice prompt file?

        if let Some(path) = valid_voice_state {
            let state = self.get_voice_state(path)?;

            let audio_tensor = self
                .inner
                .generate(text, &state.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            println!(
                "DEBUG: Output tensor shape before flatten: {:?}",
                audio_tensor.shape()
            );
            let audio_data = audio_tensor
                .flatten_all()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            Ok(audio_data)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Voice state path must be provided for now",
            ))
        }
    }

    /// Create a voice state from an audio file path
    /// Create a voice state from an audio file path or safetensors file
    fn get_voice_state(&self, path: &str) -> PyResult<PyModelState> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let state = if ext == "safetensors" {
            self.inner
                .get_voice_state_from_prompt_file(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            self.inner
                .get_voice_state(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };

        Ok(PyModelState { inner: state })
    }

    /// Generate using the voice state
    fn generate_audio(&self, text: &str, voice_state: &PyModelState) -> PyResult<Vec<f32>> {
        let audio_tensor = self
            .inner
            .generate(text, &voice_state.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        println!(
            "DEBUG: Output tensor shape before flatten: {:?}",
            audio_tensor.shape()
        );
        let audio_data = audio_tensor
            .flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(audio_data)
    }
}

/// Python wrapper for ModelState
#[pyclass]
#[derive(Clone)]
struct PyModelState {
    inner: ModelState,
}

/// The main module exposed to Python
#[pymodule]
fn pocket_tts_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTTSModel>()?;
    m.add_class::<PyModelState>()?;
    Ok(())
}
