//! Built-in components.
//!
//! These ship with the engine and are always available to scenes. Each
//! one gets `#[derive(Component)]`, which produces the `bevy_ecs::Component`
//! impl, a schema registration (picked up by the LSP / JSON emitter), and
//! a TOML loader (picked up by `ocarina-scene` to attach this component
//! to an entity when its section appears in a `.toml` file).

use serde::{Deserialize, Serialize};

use crate::asset::AssetPath;
use crate::math::{Mat4, Quat, Vec3, Vec4};
use crate::Component;

// Manual loader for the `[[script]]` array-of-tables form. The
// `inventory::submit!` lives there; declaring the module here ensures
// it links into the binary's inventory section.
mod script_loader;

/// Positions, rotates, and scales an entity in 3D space.
///
/// Transforms compose hierarchically through parent folders: a cave's
/// transform positions the cave, and its child folders (hitboxes,
/// decorations, etc.) inherit that transform.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Transform {
    /// World position, meters.
    #[ocarina(widget = "vec3", default = "[0.0, 0.0, 0.0]")]
    #[serde(default)]
    pub position: Vec3,

    /// Rotation as Euler angles in degrees (XYZ order).
    #[ocarina(widget = "euler_rotation", default = "[0.0, 0.0, 0.0]")]
    #[serde(default)]
    pub rotation: Vec3,

    /// Scale factor per axis.
    #[ocarina(widget = "vec3", default = "[1.0, 1.0, 1.0]")]
    #[serde(default = "one_vec3")]
    pub scale: Vec3,
}

fn one_vec3() -> Vec3 {
    Vec3::ONE
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Build the model matrix for this transform: T*R*S with rotation
    /// taken as Euler angles in degrees, applied YXZ (yaw, pitch,
    /// roll order). YXZ keeps yaw (Y) as the first axis, so it stays
    /// in the full [-180, 180] degree range without the asin-clamp
    /// flip that XYZ exhibits past 90 degrees of yaw. This matters
    /// for character controllers that read rotation.y back to compute
    /// movement direction.
    #[must_use]
    pub fn matrix(&self) -> Mat4 {
        let rot = Quat::from_euler(
            glam::EulerRot::YXZ,
            self.rotation.y.to_radians(),
            self.rotation.x.to_radians(),
            self.rotation.z.to_radians(),
        );
        Mat4::from_scale_rotation_translation(self.scale, rot, self.position)
    }
}

/// Geometry attached to an entity. One job: name an asset to draw.
/// Surface properties (texture, tint, future shader) live on
/// [`Material`].
///
/// `source` is an asset path. The mesh loader recognises three
/// shapes:
///
/// - `"engine/cube"`, `"engine/sphere"`, ...: reserved prefix for
///   built-in meshes baked into the engine binary. Available
///   without the project shipping any geometry of its own.
/// - `"assets/wood.obj"` (or any project-relative path): loaded
///   from the project folder via the OBJ loader.
/// - `""` (empty): no mesh, the entity does not render. Equivalent
///   to omitting the `[mesh]` section entirely; explicit empty is
///   provided for entities that want to toggle visibility from a
///   script without losing their authored path.
#[derive(Component, Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mesh {
    #[serde(default)]
    pub source: AssetPath,
}

/// Surface appearance for an entity that has a [`Mesh`]. Optional;
/// missing material means "neutral grey, no texture, default lit
/// shading." The shader path lands here when custom shaders ship.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Diffuse texture, project-relative. Empty means "no texture":
    /// fall back to flat-shaded `tint`.
    #[serde(default)]
    pub texture: AssetPath,

    /// Base color tint, RGBA in 0..1.
    #[ocarina(widget = "color", default = "[1.0, 1.0, 1.0, 1.0]")]
    #[serde(default = "white_vec4")]
    pub tint: Vec4,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            texture: AssetPath::default(),
            tint: white_vec4(),
        }
    }
}

fn white_vec4() -> Vec4 {
    Vec4::ONE
}

/// A camera. Exactly one active camera is used to render the scene.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Camera {
    /// Vertical field of view, degrees.
    #[ocarina(default = "55.0")]
    #[serde(default = "default_fov")]
    pub fov_y: f32,

    /// Near clip plane.
    #[ocarina(default = "0.05")]
    #[serde(default = "default_near")]
    pub near: f32,

    /// Far clip plane.
    #[ocarina(default = "1000.0")]
    #[serde(default = "default_far")]
    pub far: f32,
}

fn default_fov() -> f32 {
    55.0
}
fn default_near() -> f32 {
    0.05
}
fn default_far() -> f32 {
    1000.0
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y: default_fov(),
            near: default_near(),
            far: default_far(),
        }
    }
}

/// A human-readable name for an entity. Shown in the editor hierarchy;
/// also used by scripts to look entities up by name.
#[derive(Component, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Name {
    /// Display name.
    #[ocarina(default = "\"\"")]
    #[serde(default)]
    pub value: String,
}

/// One row of `[[script]]` on an entity. Carries the project-relative
/// path the runtime resolves against its script registry to construct
/// + attach the script.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScriptRef {
    #[serde(default)]
    pub path: AssetPath,
}

/// All scripts attached to one entity. Authored as repeated
/// `[[script]]` tables in TOML; this component holds them together so
/// the runtime can construct + tick each one in order.
///
/// The scene loader treats `script` as a special section that expects
/// an array of tables -- see the manual loader registration in
/// `ocarina-core/src/components/script.rs`.
#[derive(bevy_ecs::component::Component, Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scripts {
    pub items: Vec<ScriptRef>,
}
