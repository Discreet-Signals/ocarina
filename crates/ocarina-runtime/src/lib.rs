//! Shippable runtime engine.
//!
//! Loads a baked asset archive, deserializes the ECS world, ticks simulation,
//! renders via the RHI, runs audio. This crate **must not** depend on editor
//! crates — its dependency tree defines the ship-binary dep surface.
//!
//! Fleshed out in Phase 6.
