//! Math types. Re-exports `glam`'s core types under stable names so downstream
//! code can depend on `ocarina_core::math::Vec3` instead of pinning a glam
//! version directly.

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
