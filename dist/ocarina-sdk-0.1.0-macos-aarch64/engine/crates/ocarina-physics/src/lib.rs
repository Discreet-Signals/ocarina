//! Physics for Ocarina.
//!
//! Two layers:
//!
//! - [`components`] -- `Collider`, `RigidBody`, `Velocity` and friends.
//!   Engine-native types: no `rapier3d` types appear in their public
//!   surface, so any crate (CLI, LSP, editor) can depend on these
//!   without dragging the solver in.
//! - [`solver`] (gated behind the `solver` feature, on by default) --
//!   wraps `rapier3d::pipeline::PhysicsPipeline` as a [`Physics`]
//!   resource. The runtime calls [`Physics::step`] each tick to
//!   integrate dynamic bodies and resolve contacts.
//!
//! The split lets schema-only consumers skip rapier entirely. The
//! runtime, the editor's play mode (when it lands), and any binary
//! that actually simulates physics opts in via the `solver` feature.

pub mod components;

#[cfg(feature = "solver")]
pub mod solver;

#[cfg(feature = "solver")]
pub use solver::Physics;

// Re-export the components at the crate root for ergonomic use:
//   use ocarina_physics::{Collider, RigidBody, Velocity};
pub use components::{
    Collider, ColliderShape, RigidBody, RigidBodyKind, Velocity,
};
