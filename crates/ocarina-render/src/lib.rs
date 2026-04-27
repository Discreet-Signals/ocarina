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
pub mod lit;
pub mod mesh;

#[cfg(feature = "editor")]
pub mod editor;

pub use lit::{CameraUniform, InstanceUniform, LitPipeline, INSTANCE_STRIDE, MAX_INSTANCES};
pub use mesh::{
    MeshAsset, MeshHandle, MeshRegistry, Vertex, ENGINE_CUBE, ENGINE_PREFIX, ENGINE_SPHERE,
};

#[cfg(feature = "editor")]
pub use editor::RenderConfig;
