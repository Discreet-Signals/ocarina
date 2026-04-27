//! Shared rendering core for the Ocarina engine.
//!
//! This crate sits **above** the [`ocarina-rhi`] trait and **below**
//! every host that wants to draw a scene (the runtime's windowed
//! mode, the editor's viewport panel, and any future host such as
//! a hot-reload debugger or a profiler overlay). It owns the lit
//! pipeline, the mesh registry, and the canonical UBO formats so
//! every host renders the same scene the same way.
//!
//! No `wgpu` types appear in the public API. All resource creation
//! and command recording route through `ocarina-rhi`. A second
//! backend (Vulkan, Metal, D3D12) drops in by adding another
//! `ocarina-rhi-*` crate; this crate doesn't change.
//!
//! ## Feature gates
//!
//! - `obj` (default): OBJ file loading via `tobj` for user meshes
//!   referenced by `Mesh.source`.
//! - `editor` (off by default): wireframe alt pipeline, unlit
//!   shader variant, [`RenderConfig`] toggles, custom-shader hot
//!   path. The runtime binary builds with this off so none of that
//!   code, data, or branching reaches the shipped binary.

pub mod builtins;
pub mod default;
pub mod mesh;
pub mod shader;
pub mod sky;
pub mod texture;

#[cfg(feature = "editor")]
pub mod editor;

pub use default::{
    build_lights_uniform, build_material_textures_bg, build_scene_pipeline_against,
    build_simple_pipeline_against, collect_lights_from_world, CameraUniform, DefaultPipeline,
    InstanceUniform, LightInstance, LightKind, LightUniform, LightsUniform, MaterialTextureKey,
    MaterialUniform, INSTANCE_STRIDE, LIGHTS_UNIFORM_SIZE, MATERIAL_STRIDE, MATERIAL_UNIFORM_SIZE,
    MAX_INSTANCES, MAX_LIGHTS,
};
pub use mesh::{
    MeshAsset, MeshHandle, MeshRegistry, Vertex, ENGINE_CUBE, ENGINE_PREFIX, ENGINE_SPHERE,
};
pub use shader::{
    RenderSettings, ShaderRegistry, ShaderSource, ENGINE_SHADER_DEFAULT, ENGINE_SHADER_PREFIX,
};
pub use sky::{build_sky_pipeline, SkyPipeline, SkyUniform, SKY_UNIFORM_SIZE};
pub use texture::{
    ColorSpace, TextureRegistry, ENGINE_TEXTURE_BLACK, ENGINE_TEXTURE_NORMAL,
    ENGINE_TEXTURE_PREFIX, ENGINE_TEXTURE_WHITE,
};

#[cfg(feature = "editor")]
pub use editor::RenderConfig;
