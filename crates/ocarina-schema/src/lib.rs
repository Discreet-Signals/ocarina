//! Component schema metadata + TOML → ECS loader registry.
//!
//! Two kinds of entries are registered per component via `inventory::submit!`:
//!
//! - [`ComponentRegistration`]: returns a [`ComponentSchema`] describing the
//!   component's type, fields, and widget hints. Used by the LSP, the
//!   editor's lens panel, and the `ocarina schema` JSON emitter.
//! - [`ComponentLoader`]: a type-erased "parse this TOML value into this
//!   component type and attach it to the given entity" function. Used by
//!   `ocarina-scene` to walk a project directory and populate the ECS.
//!
//! Both entries are produced by `#[derive(Component)]` in `ocarina-derive`.

use serde::{Deserialize, Serialize};

// Re-export so the derive macro and user code can reach both crates
// through `ocarina_schema` without declaring them as direct deps.
pub use inventory;
pub use toml;

// ─────────────────────────────── Schema ────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentSchema {
    /// TOML section name (snake_case of the struct name unless overridden).
    pub name: String,
    /// Fully qualified Rust type path.
    pub rust_type: String,
    /// Doc comment from the struct.
    pub doc: String,
    pub fields: Vec<FieldSchema>,
    /// Absolute path to the `.rs` file where the struct is declared.
    /// Captured at compile time by the derive macro via
    /// `concat!(env!("CARGO_MANIFEST_DIR"), "/", file!())`. Empty string
    /// when the schema came from a pre-baked JSON dump that didn't have
    /// this field.
    #[serde(default)]
    pub source_file: String,
    /// 1-based line number of `#[derive(Component)]` in `source_file`.
    #[serde(default)]
    pub source_line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldSchema {
    pub name: String,
    pub ty: FieldType,
    pub doc: String,
    pub widget: Widget,
    /// The original Rust type identifier for this field, retained verbatim
    /// for the LSP ("Vec3", "Option<Color>", etc.).
    pub rust_type: String,
    /// TOML-literal default if one was declared via
    /// `#[ocarina(default = "...")]`.
    pub default_toml: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldType {
    Bool,
    Int,
    Uint,
    Float,
    String,
    Vec2,
    Vec3,
    Vec4,
    Quat,
    Color,
    /// Project-relative path to an asset on disk. Bake harvests every
    /// instance of this field type to build the shipping inclusion set.
    AssetPath,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Widget {
    /// Pick a reasonable default based on the field type.
    Auto,
    Scalar,
    Vec3,
    EulerRotation,
    Color,
    Path,
}

// ──────────────────────────── Registration ─────────────────────────────

/// Schema entry registered by `#[derive(Component)]`. The schema is
/// built lazily on demand so string ownership isn't an issue.
pub struct ComponentRegistration {
    pub build: fn() -> ComponentSchema,
}

inventory::collect!(ComponentRegistration);

// ────────────────────────────── Loader ─────────────────────────────────

/// Parses a TOML value into a specific component type and attaches it to
/// the entity. Paired with [`ComponentRegistration`] one-to-one.
pub struct ComponentLoader {
    /// TOML section name, e.g. `"transform"`.
    pub section_name: &'static str,
    /// Fully qualified Rust type path (informational).
    pub rust_type: &'static str,
    /// Actually attaches the component. Errors surface the TOML parse
    /// diagnostic back to the caller as a string.
    pub load: fn(
        entity: &mut bevy_ecs::world::EntityWorldMut<'_>,
        value: &toml::Value,
    ) -> Result<(), String>,
}

inventory::collect!(ComponentLoader);

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("unknown component section `{0}` (no #[derive(Component)] type matches)")]
    UnknownComponent(String),
    #[error("failed to parse component `{component}`: {message}")]
    ParseFailed { component: String, message: String },
}

/// Helper used by the derive macro: parse `value` as `T` and insert it.
///
/// Users don't call this directly; it's the body of the `load` fn the
/// macro generates for each `#[derive(Component)]` struct.
pub fn load_component<T>(
    entity: &mut bevy_ecs::world::EntityWorldMut<'_>,
    value: &toml::Value,
) -> Result<(), String>
where
    T: bevy_ecs::component::Component<Mutability = bevy_ecs::component::Mutable>
        + for<'de> Deserialize<'de>,
{
    let parsed: T = value
        .clone()
        .try_into()
        .map_err(|e: toml::de::Error| e.to_string())?;
    entity.insert(parsed);
    Ok(())
}

// ──────────────────────────── Query helpers ────────────────────────────

/// Iterate over every registered component schema.
pub fn iter_schemas() -> impl Iterator<Item = ComponentSchema> {
    inventory::iter::<ComponentRegistration>
        .into_iter()
        .map(|r| (r.build)())
}

/// Find the loader for a TOML section name.
pub fn find_loader(section_name: &str) -> Option<&'static ComponentLoader> {
    inventory::iter::<ComponentLoader>
        .into_iter()
        .find(|l| l.section_name == section_name)
}

/// Collect all registered schemas into a Vec (owned). Convenient for
/// serializing to JSON for the LSP.
#[must_use]
pub fn collect_schemas() -> Vec<ComponentSchema> {
    iter_schemas().collect()
}
