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

/// Standard PBR surface description. Fed to the default scene
/// shader; user-supplied scene shaders may interpret the same fields
/// however they like (or read additional fields off custom
/// components later).
///
/// Each texture slot is an [`AssetPath`]. Empty means "no texture":
/// the shader falls back to the corresponding scalar / color value.
/// Empty paths resolve to a built-in 1×1 white texture in the
/// renderer so the bind group layout stays uniform across draws.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Base color (a.k.a. albedo or diffuse). Linear 0..1 RGBA.
    /// Multiplied with `albedo_texture` when one is bound.
    /// Aliased to the legacy field name `tint` for older scene
    /// TOMLs.
    #[ocarina(widget = "color", default = "[1.0, 1.0, 1.0, 1.0]")]
    #[serde(default = "white_vec4", alias = "tint")]
    pub albedo: Vec4,

    /// Albedo texture (sRGB-encoded color). Aliased to the legacy
    /// field name `texture`.
    #[serde(default, alias = "texture")]
    pub albedo_texture: AssetPath,

    /// Normal map in tangent space, encoded as RGB ∈ [0, 1] mapping
    /// to xyz ∈ [-1, 1]. Empty means "use vertex normals."
    #[serde(default)]
    pub normal_texture: AssetPath,

    /// 0 = dielectric (plastic, wood, fabric), 1 = pure metal.
    /// Multiplied by the blue channel of `metallic_roughness_texture`
    /// (glTF convention) when one is bound.
    #[ocarina(default = "0.0")]
    #[serde(default)]
    pub metallic: f32,

    /// 0 = mirror-smooth, 1 = fully diffuse. Multiplied by the green
    /// channel of `metallic_roughness_texture`.
    #[ocarina(default = "0.5")]
    #[serde(default = "default_roughness")]
    pub roughness: f32,

    /// glTF-style packed map: green = roughness, blue = metallic.
    /// Empty means use the scalar `metallic` / `roughness` values
    /// directly.
    #[serde(default)]
    pub metallic_roughness_texture: AssetPath,

    /// Self-illumination colour, linear RGB in 0..1 (think colour
    /// picker, not HDR). Added on top of lit shading; doesn't
    /// contribute to GI yet (no GI solver). The `emission_strength`
    /// scalar is the HDR multiplier so you can crank brightness
    /// without retyping every channel.
    ///
    /// Aliased to the legacy field name `emission` for older scene
    /// TOMLs.
    #[ocarina(widget = "color", default = "[0.0, 0.0, 0.0]")]
    #[serde(default, alias = "emission")]
    pub emission_color: Vec3,

    /// Brightness multiplier on `emission_color`. 1.0 = unit-bright;
    /// values >1.0 blow past sRGB white and read as "glowing" once
    /// tone mapping lands. 0.0 disables emission regardless of
    /// colour.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_emission_strength")]
    pub emission_strength: f32,

    /// Multiplied by `emission_color * emission_strength` when bound.
    #[serde(default)]
    pub emission_texture: AssetPath,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: white_vec4(),
            albedo_texture: AssetPath::default(),
            normal_texture: AssetPath::default(),
            metallic: 0.0,
            roughness: default_roughness(),
            metallic_roughness_texture: AssetPath::default(),
            emission_color: Vec3::ZERO,
            emission_strength: default_emission_strength(),
            emission_texture: AssetPath::default(),
        }
    }
}

fn white_vec4() -> Vec4 {
    Vec4::ONE
}
fn default_roughness() -> f32 {
    0.5
}
fn default_emission_strength() -> f32 {
    1.0
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

// ─────────────────────────── Sky ───────────────────────────

/// 3-stop gradient skybox. The renderer picks the first entity with
/// a `Sky` component and uses it as the scene's background. The
/// fragment shader reconstructs world-space view direction per pixel
/// and blends the three colours by `dir.y` -- looking straight down
/// shows only `bottom_color`, straight up only `top_color`, with the
/// horizon fading between them.
///
/// `intensity` is an HDR multiplier that can push past 1.0 (think
/// "noon glare"). `horizon_falloff` controls how sharply the
/// horizon line transitions: 1.0 is a soft gradient, higher values
/// pull the horizon into a tighter band.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sky {
    /// Colour at the zenith (`+Y` direction), linear RGB.
    #[ocarina(widget = "color", default = "[0.42, 0.55, 0.78]")]
    #[serde(default = "default_sky_top")]
    pub top_color: Vec3,

    /// Colour at the horizon line.
    #[ocarina(widget = "color", default = "[0.85, 0.78, 0.65]")]
    #[serde(default = "default_sky_horizon")]
    pub horizon_color: Vec3,

    /// Colour below the horizon (`-Y`); often a darker version of
    /// the ground tint or dusk colour.
    #[ocarina(widget = "color", default = "[0.10, 0.12, 0.16]")]
    #[serde(default = "default_sky_bottom")]
    pub bottom_color: Vec3,

    /// Sharpness of the horizon line. 1.0 = linear blend; higher
    /// values pull the horizon into a tighter band.
    #[ocarina(default = "2.0")]
    #[serde(default = "default_sky_falloff")]
    pub horizon_falloff: f32,

    /// HDR brightness multiplier on all three stops.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_sky_intensity")]
    pub intensity: f32,
}

impl Default for Sky {
    fn default() -> Self {
        Self {
            top_color: default_sky_top(),
            horizon_color: default_sky_horizon(),
            bottom_color: default_sky_bottom(),
            horizon_falloff: default_sky_falloff(),
            intensity: default_sky_intensity(),
        }
    }
}

fn default_sky_top() -> Vec3 {
    Vec3::new(0.42, 0.55, 0.78)
}
fn default_sky_horizon() -> Vec3 {
    Vec3::new(0.85, 0.78, 0.65)
}
fn default_sky_bottom() -> Vec3 {
    Vec3::new(0.10, 0.12, 0.16)
}
fn default_sky_falloff() -> f32 {
    2.0
}
fn default_sky_intensity() -> f32 {
    1.0
}

// ─────────────────────────── Lights ───────────────────────────

/// Sun-style directional light. Illuminates everything in the scene
/// from a single direction with no attenuation; physically the
/// approximation for "very far away light source." The light's
/// direction is the entity's local `-Z` axis after the entity's
/// `Transform.rotation` is applied (same convention as cameras).
/// Position is ignored.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DirectionalLight {
    /// Light color, linear RGB.
    #[ocarina(default = "[1.0, 1.0, 1.0]")]
    #[serde(default = "white_vec3")]
    pub color: Vec3,

    /// Brightness multiplier. 1.0 ≈ outdoor noon when combined with a
    /// neutral PBR material.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_light_intensity")]
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: white_vec3(),
            intensity: default_light_intensity(),
        }
    }
}

/// Omni-directional light at a point. Falls off with distance and
/// stops contributing past `range`. Position comes from the entity's
/// `Transform.position`.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PointLight {
    /// Light color, linear RGB.
    #[ocarina(default = "[1.0, 1.0, 1.0]")]
    #[serde(default = "white_vec3")]
    pub color: Vec3,

    /// Brightness multiplier.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_light_intensity")]
    pub intensity: f32,

    /// Maximum distance the light reaches, meters. Beyond this the
    /// shader contributes zero -- a hard cutoff that bounds shading
    /// cost. The smooth attenuation curve inside the range is
    /// physically motivated (inverse-square with a saturating
    /// near-field term); the cutoff is engineering.
    #[ocarina(default = "10.0")]
    #[serde(default = "default_light_range")]
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: white_vec3(),
            intensity: default_light_intensity(),
            range: default_light_range(),
        }
    }
}

/// Cone light. Like a [`PointLight`] but constrained to an angular
/// cone; full intensity inside `inner_angle`, smoothly falling to
/// zero by `outer_angle`. Direction is the entity's local `-Z` axis
/// after the entity's `Transform.rotation` is applied; position
/// comes from `Transform.position`.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpotLight {
    /// Light color, linear RGB.
    #[ocarina(default = "[1.0, 1.0, 1.0]")]
    #[serde(default = "white_vec3")]
    pub color: Vec3,

    /// Brightness multiplier.
    #[ocarina(default = "1.0")]
    #[serde(default = "default_light_intensity")]
    pub intensity: f32,

    /// Maximum distance the light reaches, meters.
    #[ocarina(default = "10.0")]
    #[serde(default = "default_light_range")]
    pub range: f32,

    /// Half-angle of the inner cone, degrees. Pixels with their
    /// surface vector inside this cone receive full directional
    /// intensity.
    #[ocarina(default = "20.0")]
    #[serde(default = "default_spot_inner_angle")]
    pub inner_angle: f32,

    /// Half-angle of the outer cone, degrees. Pixels between the
    /// inner and outer cones blend smoothly to zero. Outside the
    /// outer cone the light contributes nothing. Must satisfy
    /// `outer_angle >= inner_angle`.
    #[ocarina(default = "30.0")]
    #[serde(default = "default_spot_outer_angle")]
    pub outer_angle: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            color: white_vec3(),
            intensity: default_light_intensity(),
            range: default_light_range(),
            inner_angle: default_spot_inner_angle(),
            outer_angle: default_spot_outer_angle(),
        }
    }
}

fn white_vec3() -> Vec3 {
    Vec3::ONE
}
fn default_light_intensity() -> f32 {
    1.0
}
fn default_light_range() -> f32 {
    10.0
}
fn default_spot_inner_angle() -> f32 {
    20.0
}
fn default_spot_outer_angle() -> f32 {
    30.0
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
