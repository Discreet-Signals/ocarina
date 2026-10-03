//! Physics components.
//!
//! Authoring shape (`*.toml`):
//!
//! ```toml
//! [transform]
//! position = [0.0, 5.0, 0.0]
//!
//! [collider]
//! shape    = "sphere"
//! radius   = 0.5
//! restitution = 0.4
//!
//! [rigid_body]
//! kind = "dynamic"
//! mass = 1.0
//! ```
//!
//! Each entity carries at most one `Collider` and at most one
//! `RigidBody`. An entity with `Collider` but no `RigidBody` becomes
//! a fixed (unmovable, infinite-mass) collider. `Velocity` is
//! optional; the solver maintains it for dynamic bodies and reads it
//! for kinematic ones.

use ocarina_core::math::Vec3;
use ocarina_core::Component;
use serde::{Deserialize, Serialize};

// ─────────────────────────── Collider ───────────────────────────

/// A collision shape attached to an entity. Shape data is serialised
/// flat in TOML alongside `shape`:
///
/// ```toml
/// [collider]
/// shape  = "box"
/// size   = [1.0, 0.5, 1.0]
///
/// [collider]
/// shape  = "sphere"
/// radius = 0.5
///
/// [collider]
/// shape       = "capsule"
/// half_height = 0.5
/// radius      = 0.3
/// ```
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct Collider {
    /// Geometry. `serde(flatten)` lets the variant's fields sit
    /// alongside `shape = "..."` in TOML rather than nested under
    /// a sub-table.
    #[serde(flatten)]
    pub shape: ColliderShape,

    /// Local-space offset of the shape's centre from the entity's
    /// transform origin. Useful for cubes whose ECS pivot is at the
    /// base, etc.
    #[ocarina(default = "[0.0, 0.0, 0.0]")]
    #[serde(default)]
    pub offset: Vec3,

    /// 0..=1 multiplier on the relative velocity along the contact
    /// normal at impact. 0 = perfectly inelastic, 1 = perfect bounce.
    #[ocarina(default = "0.0")]
    #[serde(default)]
    pub restitution: f32,

    /// Coulomb friction coefficient. 0 = frictionless ice, 1 ≈ rubber
    /// on dry pavement.
    #[ocarina(default = "0.5")]
    #[serde(default = "default_friction")]
    pub friction: f32,

    /// True for trigger-style colliders that report contacts but
    /// generate no contact response. Useful for hitboxes and zones.
    #[ocarina(default = "false")]
    #[serde(default)]
    pub sensor: bool,
}

impl Default for Collider {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                size: Vec3::ONE,
            },
            offset: Vec3::ZERO,
            restitution: 0.0,
            friction: default_friction(),
            sensor: false,
        }
    }
}

fn default_friction() -> f32 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ColliderShape {
    /// Axis-aligned box. `size` is the FULL extent on each axis (a
    /// 1×1×1 box has `size = [1.0, 1.0, 1.0]`).
    Box {
        #[serde(default = "one_vec3")]
        size: Vec3,
    },
    /// Solid sphere of radius `radius`.
    Sphere {
        #[serde(default = "default_radius")]
        radius: f32,
    },
    /// Y-aligned capsule. Total height = `2 * (half_height + radius)`.
    Capsule {
        #[serde(default = "default_capsule_h")]
        half_height: f32,
        #[serde(default = "default_radius")]
        radius: f32,
    },
    /// Infinite-extent half-space, normal +Y. The world below `y =
    /// position.y` is solid. Cheaper than a giant box for ground
    /// planes.
    HalfSpace,
}

fn one_vec3() -> Vec3 {
    Vec3::ONE
}
fn default_radius() -> f32 {
    0.5
}
fn default_capsule_h() -> f32 {
    0.5
}

// ─────────────────────────── RigidBody ───────────────────────────

#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct RigidBody {
    #[ocarina(default = "\"dynamic\"")]
    #[serde(default)]
    pub kind: RigidBodyKind,

    /// Mass in kilograms. Ignored for `Static` and `Kinematic` bodies
    /// (which are treated as infinite mass by the solver). For
    /// dynamic bodies, `0.0` means "let the solver compute mass from
    /// the collider's volume × density 1.0."
    #[ocarina(default = "0.0")]
    #[serde(default)]
    pub mass: f32,

    /// Multiplier applied to global gravity. 0 = floating, 1 =
    /// normal, 2 = heavier-feeling. Negative for upward attraction.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_gravity_scale")]
    pub gravity_scale: f32,

    /// Linear damping rate (per second). Higher = faster slowdown.
    #[ocarina(default = "0.0")]
    #[serde(default)]
    pub linear_damping: f32,

    /// Angular damping rate (per second).
    #[ocarina(default = "0.0")]
    #[serde(default)]
    pub angular_damping: f32,

    /// Per-axis rotation locks. A character-controller cube that
    /// should turn (yaw) but never tip uses
    /// `lock_rotation_x = true, lock_rotation_z = true`.
    #[ocarina(default = "false")]
    #[serde(default)]
    pub lock_rotation_x: bool,
    #[ocarina(default = "false")]
    #[serde(default)]
    pub lock_rotation_y: bool,
    #[ocarina(default = "false")]
    #[serde(default)]
    pub lock_rotation_z: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            kind: RigidBodyKind::default(),
            mass: 0.0,
            gravity_scale: default_gravity_scale(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
        }
    }
}

fn default_gravity_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RigidBodyKind {
    /// Affected by gravity, forces, and contacts.
    #[default]
    Dynamic,
    /// Moved by code (set position / velocity directly). Pushes
    /// dynamic bodies but is not pushed back.
    Kinematic,
    /// Never moves. Treats infinite mass.
    Static,
}

// ─────────────────────────── Velocity ───────────────────────────

/// Linear + angular velocity. Updated each frame by the solver for
/// dynamic bodies; you can write to it for kinematic bodies and the
/// solver will move the body accordingly.
#[derive(Component, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Velocity {
    /// World-space linear velocity (m/s).
    #[ocarina(default = "[0.0, 0.0, 0.0]")]
    #[serde(default)]
    pub linear: Vec3,

    /// World-space angular velocity (rad/s, axis-angle).
    #[ocarina(default = "[0.0, 0.0, 0.0]")]
    #[serde(default)]
    pub angular: Vec3,
}
