//! Core math types, built-in components, and the Ocarina prelude.
//!
//! `ocarina-core` is consumed by the runtime, the editor, and by every
//! user gameplay crate. It must stay lightweight — no editor deps, no
//! rendering deps beyond the RHI trait.

pub mod components;
pub mod math;
pub mod prelude;

// Re-export the derive macro so user code only needs one import:
//   use ocarina_core::prelude::*;
//   #[derive(Component)] ...
pub use ocarina_derive::Component;
