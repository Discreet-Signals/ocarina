//! Mesh data, registry, and asset loading.
//!
//! One mesh component, one mesh path. Both engine-shipped meshes
//! (`engine/cube`, `engine/sphere`) and project-relative user
//! meshes (`assets/wood.obj`) are looked up through the same
//! [`MeshRegistry`] keyed by [`AssetPath`]. There is no special
//! "built-in" enum or escape hatch on the component side.
//!
//! The `engine/` path prefix is reserved. The registry pre-loads
//! engine meshes from data baked into the binary at startup, so
//! they appear as ordinary `MeshHandle`s like any user-loaded OBJ.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use ocarina_core::AssetPath;
use ocarina_rhi::{
    BufferDesc, BufferHandle, BufferUsage, MemoryLocation, Rhi, RhiResult,
};

// ─────────────────────────── Vertex format ───────────────────────────

/// Interleaved vertex used by every built-in pipeline. Position +
/// normal + uv. 32 bytes; matches the layout the lit shader expects.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

// ─────────────────────────── Mesh asset ───────────────────────────

/// CPU-side mesh data. The registry uploads this to GPU buffers on
/// first use; subsequent renders pull the GPU handle from the cache.
#[derive(Debug, Clone)]
pub struct MeshAsset {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

impl MeshAsset {
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.aabb_min + self.aabb_max) * 0.5
    }
    #[must_use]
    pub fn radius(&self) -> f32 {
        ((self.aabb_max - self.aabb_min) * 0.5).length()
    }
}

/// Handle returned from [`MeshRegistry::ensure`]. Points at GPU
/// resources; cheap to clone and store across frames.
#[derive(Debug, Clone, Copy)]
pub struct MeshHandle {
    pub vertex_buffer: BufferHandle,
    pub index_buffer: BufferHandle,
    pub index_count: u32,
}

// ─────────────────────────── Built-in catalogue ───────────────────────────

/// Reserved path prefix for engine-shipped meshes. The registry
/// recognises this and serves from embedded data instead of hitting
/// the project filesystem.
pub const ENGINE_PREFIX: &str = "engine/";

/// Engine path of the built-in unit cube.
pub const ENGINE_CUBE: &str = "engine/cube";
/// Engine path of the built-in unit sphere.
pub const ENGINE_SPHERE: &str = "engine/sphere";

// ─────────────────────────── Registry ───────────────────────────

/// Owns the GPU buffers backing every mesh in use. Engine meshes
/// are uploaded at construction; user meshes load and cache the
/// first time an entity references them.
pub struct MeshRegistry {
    /// Project root used to resolve user-asset paths. `None` means
    /// only engine meshes are available; user-mesh requests fail.
    project_root: Option<std::path::PathBuf>,
    cache: HashMap<AssetPath, MeshHandle>,
    /// Failures we've already logged so we don't spam every frame.
    failed: HashMap<AssetPath, ()>,
}

impl MeshRegistry {
    /// Construct an empty registry. Call [`MeshRegistry::register_builtins`]
    /// once you have an `Rhi` to upload to.
    #[must_use]
    pub fn new() -> Self {
        Self {
            project_root: None,
            cache: HashMap::new(),
            failed: HashMap::new(),
        }
    }

    pub fn set_project_root(&mut self, root: std::path::PathBuf) {
        self.project_root = Some(root);
    }

    /// Upload the engine's built-in meshes so subsequent
    /// `ensure(AssetPath::new("engine/..."))` calls hit the cache.
    pub fn register_builtins(&mut self, rhi: &mut dyn Rhi) -> RhiResult<()> {
        let cube = crate::builtins::build_unit_cube();
        let h = upload_mesh(rhi, ENGINE_CUBE, &cube)?;
        self.cache.insert(AssetPath::new(ENGINE_CUBE), h);

        let sphere = crate::builtins::build_unit_sphere(2);
        let h = upload_mesh(rhi, ENGINE_SPHERE, &sphere)?;
        self.cache.insert(AssetPath::new(ENGINE_SPHERE), h);

        Ok(())
    }

    /// Resolve an asset path to a GPU mesh, loading from disk on
    /// first ask for project meshes. Returns `None` for an empty
    /// path or a load failure (logged once); the renderer treats
    /// `None` as "draw nothing for this entity."
    pub fn ensure(&mut self, rhi: &mut dyn Rhi, path: &AssetPath) -> Option<MeshHandle> {
        if path.is_empty() {
            return None;
        }
        if let Some(h) = self.cache.get(path) {
            return Some(*h);
        }
        if self.failed.contains_key(path) {
            return None;
        }
        // The `engine/` namespace is reserved. If we reach here for
        // a path under it, the user typed an unknown built-in name.
        if path.as_str().starts_with(ENGINE_PREFIX) {
            log::warn!(
                "ocarina-render: unknown engine mesh `{}` (known: `{ENGINE_CUBE}`, `{ENGINE_SPHERE}`)",
                path.as_str(),
            );
            self.failed.insert(path.clone(), ());
            return None;
        }
        match self.load_asset(rhi, path) {
            Ok(handle) => {
                self.cache.insert(path.clone(), handle);
                Some(handle)
            }
            Err(e) => {
                log::warn!(
                    "ocarina-render: failed to load mesh `{}`: {e}",
                    path.as_str()
                );
                self.failed.insert(path.clone(), ());
                None
            }
        }
    }

    fn load_asset(
        &self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
    ) -> Result<MeshHandle, MeshLoadError> {
        let root = self
            .project_root
            .as_ref()
            .ok_or(MeshLoadError::NoProjectRoot)?;
        let absolute = path.resolve(root);
        let ext = absolute
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            #[cfg(feature = "obj")]
            Some("obj") => {
                let asset = load_obj(&absolute)?;
                upload_mesh(rhi, &absolute.display().to_string(), &asset)
                    .map_err(|e| MeshLoadError::Upload(e.to_string()))
            }
            Some(other) => Err(MeshLoadError::UnsupportedExtension(other.to_string())),
            None => Err(MeshLoadError::UnsupportedExtension("(none)".into())),
        }
    }
}

impl Default for MeshRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────── Upload helper ───────────────────────────

fn upload_mesh(rhi: &mut dyn Rhi, label: &str, mesh: &MeshAsset) -> RhiResult<MeshHandle> {
    let vb_size = (mesh.vertices.len() * Vertex::SIZE) as u64;
    let vertex_buffer = rhi.create_buffer(&BufferDesc {
        label: Some(label),
        size: vb_size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));

    let ib_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
    let index_buffer = rhi.create_buffer(&BufferDesc {
        label: Some(label),
        size: ib_size,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(index_buffer, 0, bytemuck::cast_slice(&mesh.indices));

    Ok(MeshHandle {
        vertex_buffer,
        index_buffer,
        index_count: mesh.indices.len() as u32,
    })
}

// ─────────────────────────── OBJ loader ───────────────────────────

#[derive(Debug)]
pub enum MeshLoadError {
    NoProjectRoot,
    UnsupportedExtension(String),
    Io(std::io::Error),
    Parse(String),
    Upload(String),
}

impl std::fmt::Display for MeshLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProjectRoot => write!(f, "registry has no project root set"),
            Self::UnsupportedExtension(e) => write!(f, "unsupported extension `{e}`"),
            Self::Io(e) => write!(f, "i/o: {e}"),
            Self::Parse(e) => write!(f, "parse: {e}"),
            Self::Upload(e) => write!(f, "upload: {e}"),
        }
    }
}

impl std::error::Error for MeshLoadError {}

#[cfg(feature = "obj")]
fn load_obj(path: &std::path::Path) -> Result<MeshAsset, MeshLoadError> {
    let opts = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ignore_lines: true,
        ignore_points: true,
    };
    let (models, _materials) = tobj::load_obj(path, &opts)
        .map_err(|e| MeshLoadError::Parse(e.to_string()))?;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

    for model in models {
        let mesh = &model.mesh;
        let base = vertices.len() as u32;
        let position_count = mesh.positions.len() / 3;
        let has_normals = !mesh.normals.is_empty()
            && mesh.normals.len() == mesh.positions.len();
        let has_uvs = !mesh.texcoords.is_empty();

        for i in 0..position_count {
            let p = Vec3::new(
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            );
            aabb_min = aabb_min.min(p);
            aabb_max = aabb_max.max(p);
            let n = if has_normals {
                [
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]
            } else {
                [0.0, 1.0, 0.0]
            };
            let uv = if has_uvs && i * 2 + 1 < mesh.texcoords.len() {
                [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };
            vertices.push(Vertex {
                position: [p.x, p.y, p.z],
                normal: n,
                uv,
            });
        }

        for &idx in &mesh.indices {
            indices.push(base + idx);
        }

        // OBJ without normals: derive flat per-face normals from
        // triangle winding. The last face writing each vertex wins,
        // which is a simplification but gives a sane look for
        // un-normaled assets.
        if !has_normals {
            let tri_count = (indices.len() - (base as usize)) / 3;
            let start = (indices.len() - tri_count * 3) as usize;
            for tri in 0..tri_count {
                let i0 = indices[start + tri * 3] as usize;
                let i1 = indices[start + tri * 3 + 1] as usize;
                let i2 = indices[start + tri * 3 + 2] as usize;
                let p0 = Vec3::from_array(vertices[i0].position);
                let p1 = Vec3::from_array(vertices[i1].position);
                let p2 = Vec3::from_array(vertices[i2].position);
                let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                let arr = [n.x, n.y, n.z];
                vertices[i0].normal = arr;
                vertices[i1].normal = arr;
                vertices[i2].normal = arr;
            }
        }
    }

    if vertices.is_empty() {
        aabb_min = Vec3::ZERO;
        aabb_max = Vec3::ZERO;
    }

    Ok(MeshAsset {
        vertices,
        indices,
        aabb_min,
        aabb_max,
    })
}

#[cfg(not(feature = "obj"))]
fn load_obj(_path: &std::path::Path) -> Result<MeshAsset, MeshLoadError> {
    Err(MeshLoadError::UnsupportedExtension(
        "obj feature disabled at compile time".into(),
    ))
}
