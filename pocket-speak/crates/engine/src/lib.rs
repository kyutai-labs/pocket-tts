//! Pocket TTS on upstream's own Rust implementation (xn-ptts).
//!
//! [`text`] turns what an agent wrote into what should be said and cuts it into
//! chunks; [`Engine`] turns one chunk into mono samples; [`models`] pins the
//! files each offered model needs. Nothing here plays audio or touches the
//! network.
//!
//! Most of the model code is upstream's, compiled in from `rust/src` by path
//! rather than copied, so each merge of upstream updates it here too, and an
//! upstream change to their interfaces fails this build instead of drifting:
//!
//! | module | from |
//! |---|---|
//! | `config` | `rust/src/config.rs`: model configs (embedded by `build.rs`) |
//! | `defaults` | `rust/src/defaults.rs` |
//! | `prompt` | `rust/src/text.rs`: preparing a chunk for the model |
//! | `weights` | `rust/src/weights.rs`: loading Kyutai's safetensors into xn-ptts |
//! | `tokenizer`, `download` | this crate, with the interfaces of upstream's |
//! | `synth` | this crate, ported from `rust/src/model.rs` and `rng.rs` |

#[allow(dead_code)]
#[rustfmt::skip] // upstream's file, formatted by upstream
#[path = "../../../../rust/src/config.rs"]
mod config;
#[allow(dead_code)]
#[rustfmt::skip] // upstream's file, formatted by upstream
#[path = "../../../../rust/src/defaults.rs"]
mod defaults;
mod download;
pub mod models;
#[allow(dead_code)]
#[rustfmt::skip] // upstream's file, formatted by upstream
#[path = "../../../../rust/src/text.rs"]
mod prompt;
mod synth;
pub mod text;
mod tokenizer;
#[allow(dead_code)]
#[rustfmt::skip] // upstream's file, formatted by upstream
#[path = "../../../../rust/src/weights.rs"]
mod weights;

pub use synth::{Cancel, ChunkStats, EndReason, Engine, Error, LoadOptions, Voice};
