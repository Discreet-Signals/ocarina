//! Core math types, built-in components, and the Ocarina prelude.
//!
//! `ocarina-core` is consumed by the runtime, the editor, and by every
//! user gameplay crate. It must stay lightweight — no editor deps, no
//! rendering deps beyond the RHI trait.

pub mod asset;
pub mod components;
pub mod input;
pub mod math;
pub mod prelude;
pub mod time;

pub use asset::AssetPath;
pub use input::{Input, Key, MouseButton};
pub use time::Time;

// Re-export the derive macro so user code only needs one import:
//   use ocarina_core::prelude::*;
//   #[derive(Component)] ...
pub use ocarina_derive::Component;
