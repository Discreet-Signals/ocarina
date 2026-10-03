//! Shader registry + scene-level shader selection.
//!
//! Shaders are first-class assets keyed by [`AssetPath`], the same
//! way meshes and textures are. The engine ships with one built-in
//! scene shader (`engine/shaders/default`); projects can drop a
//! `.wgsl` file anywhere under their root and reference it from
//! [`RenderSettings::default_shader`] to swap the entire scene
//! shading model (PBR → ray-traced → cel → whatever).
//!
//! The `engine/` path prefix is reserved -- the same convention
//! [`MeshRegistry`](crate::mesh::MeshRegistry) and
//! [`TextureRegistry`](crate::texture::TextureRegistry) use.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ocarina_core::AssetPath;

// ─────────────────────────── Built-in catalogue ───────────────────────────

/// Reserved path prefix for engine-shipped shaders.
pub const ENGINE_SHADER_PREFIX: &str = "engine/shaders/";

/// Engine path of the default scene shader (today: minimal Lambert;
/// stage 7: full PBR over the lights buffer).
pub const ENGINE_SHADER_DEFAULT: &str = "engine/shaders/default";

pub(crate) const ENGINE_DEFAULT_WGSL: &str = include_str!("../shaders/default.wgsl");

// ─────────────────────────── Shader source ───────────────────────────

/// A loaded shader, ready to hand to the RHI as
/// [`ShaderBytecode::Wgsl`](ocarina_rhi::ShaderBytecode::Wgsl).
///
/// `Arc`-shared so the registry hands out cheap clones; pipelines
/// hold one for the lifetime of the compiled module.
#[derive(Debug)]
pub struct ShaderSource {
    /// Human-readable label, used in RHI debug labels.
    pub label: String,
    /// WGSL source text. `Cow` because engine shaders are baked-in
    /// `&'static str` while user shaders are `String` read from disk.
    pub wgsl: Cow<'static, str>,
}

// ─────────────────────────── Registry ───────────────────────────

/// Owns every shader source the renderer might need this frame.
/// Engine shaders are pre-loaded at construction; user shaders are
/// read from disk and cached on first reference.
///
/// The registry is *just storage*. Pipeline compilation still
/// happens in [`DefaultPipeline`](crate::default::DefaultPipeline)
/// (and the renderer's own pipeline cache, stage 8) -- swapping a
/// shader means destroying the old pipeline and building a new one
/// against the new source.
pub struct ShaderRegistry {
    /// Wrapped in `Arc` so the per-frame `set_project_root` call from
    /// the viewport is a refcount bump, not a deep `PathBuf` clone.
    project_root: Option<Arc<std::path::PathBuf>>,
    cache: HashMap<AssetPath, Arc<ShaderSource>>,
    /// Failures we've already logged so we don't spam every frame.
    failed: HashSet<AssetPath>,
}

impl ShaderRegistry {
    /// Construct an empty registry, then pre-cache the engine
    /// built-ins so subsequent lookups for them hit the cache.
    #[must_use]
    pub fn new() -> Self {
        let mut cache = HashMap::new();
        cache.insert(
            AssetPath::new(ENGINE_SHADER_DEFAULT),
            Arc::new(ShaderSource {
                label: "engine/shaders/default".into(),
                wgsl: Cow::Borrowed(ENGINE_DEFAULT_WGSL),
            }),
        );
        Self {
            project_root: None,
            cache,
            failed: HashSet::new(),
        }
    }

    pub fn set_project_root(&mut self, root: Arc<std::path::PathBuf>) {
        self.project_root = Some(root);
    }

    /// Resolve `path` to its absolute on-disk location using the
    /// configured project root. Returns `None` if the registry has
    /// no project root set or the path is empty / engine-namespaced
    /// (engine shaders are baked-in and don't have a disk file).
    /// Used by callers that need to stat the source file (e.g. for
    /// mtime-based hot-reload).
    #[must_use]
    pub fn resolve_path(&self, path: &AssetPath) -> Option<std::path::PathBuf> {
        if path.is_empty() || path.as_str().starts_with(ENGINE_SHADER_PREFIX) {
            return None;
        }
        self.project_root.as_ref().map(|root| path.resolve(root.as_path()))
    }

    /// Drop the cached source for `path` so the next call to
    /// [`Self::ensure`] re-reads the file from disk. Also clears any
    /// remembered failure for that path. Used by hot-reload -- the
    /// source file's mtime changed, so any compiled pipeline tied to
    /// the old text needs to come down too (the caller is responsible
    /// for that; the registry is just the source-text cache).
    pub fn invalidate(&mut self, path: &AssetPath) {
        // Engine-namespaced shaders are built-in via `include_str!`,
        // not loaded from disk. Removing them from the cache would
        // strand the `ensure_or_default` fallback (which expects the
        // engine default to always be cached) and panic on the next
        // lookup. Hot-reload doesn't apply to baked-in shaders, so
        // skip them here.
        if path.is_empty() || path.as_str().starts_with(ENGINE_SHADER_PREFIX) {
            return;
        }
        self.cache.remove(path);
        self.failed.remove(path);
    }

    /// Resolve an asset path to a shader source. Returns `None` for
    /// an empty path or a load failure (logged once); callers can
    /// fall back to [`ENGINE_SHADER_DEFAULT`] when they need a
    /// guaranteed handle.
    pub fn ensure(&mut self, path: &AssetPath) -> Option<Arc<ShaderSource>> {
        if path.is_empty() {
            return None;
        }
        if let Some(s) = self.cache.get(path) {
            return Some(s.clone());
        }
        if self.failed.contains(path) {
            return None;
        }
        // Engine namespace is reserved; an unknown engine path is a
        // user typo, not a missing asset.
        if path.as_str().starts_with(ENGINE_SHADER_PREFIX) {
            log::warn!(
                "ocarina-render: unknown engine shader `{}` (known: `{ENGINE_SHADER_DEFAULT}`)",
                path.as_str(),
            );
            self.failed.insert(path.clone());
            return None;
        }
        match self.load_asset(path) {
            Ok(source) => {
                let arc = Arc::new(source);
                self.cache.insert(path.clone(), arc.clone());
                Some(arc)
            }
            Err(e) => {
                log::warn!(
                    "ocarina-render: failed to load shader `{}`: {e}",
                    path.as_str()
                );
                self.failed.insert(path.clone());
                None
            }
        }
    }

    /// Convenience: look up `path`, falling back to the built-in
    /// default scene shader on miss. Always returns a valid shader
    /// source.
    pub fn ensure_or_default(&mut self, path: &AssetPath) -> Arc<ShaderSource> {
        if let Some(s) = self.ensure(path) {
            return s;
        }
        self.cache
            .get(&AssetPath::new(ENGINE_SHADER_DEFAULT))
            .cloned()
            .expect("default scene shader pre-cached at construction")
    }

    fn load_asset(&self, path: &AssetPath) -> Result<ShaderSource, ShaderLoadError> {
        let root = self
            .project_root
            .as_ref()
            .ok_or(ShaderLoadError::NoProjectRoot)?;
        let absolute = path.resolve(root.as_path());
        let ext = absolute
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("wgsl") => {
                let wgsl = std::fs::read_to_string(&absolute).map_err(ShaderLoadError::Io)?;
                Ok(ShaderSource {
                    label: absolute.display().to_string(),
                    wgsl: Cow::Owned(wgsl),
                })
            }
            // HLSL → SPIR-V via ocarina-shader is a future addition;
            // for now we only accept WGSL on the disk path so users
            // get a clear error rather than a silent miss.
            Some(other) => Err(ShaderLoadError::UnsupportedExtension(other.to_string())),
            None => Err(ShaderLoadError::UnsupportedExtension("(none)".into())),
        }
    }
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum ShaderLoadError {
    NoProjectRoot,
    UnsupportedExtension(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ShaderLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProjectRoot => write!(f, "registry has no project root set"),
            Self::UnsupportedExtension(e) => write!(f, "unsupported extension `{e}` (only WGSL supported on disk)"),
            Self::Io(e) => write!(f, "i/o: {e}"),
        }
    }
}

impl std::error::Error for ShaderLoadError {}

// ─────────────────────────── Scene render settings ───────────────────────────

/// World-level render settings. Inserted as a `bevy_ecs` resource
/// and queried by the renderer each frame.
///
/// The default selects the engine's built-in scene shader. Game
/// scripts swap this resource (or mutate its fields) to change the
/// scene-wide shader at runtime. Per-camera post-process chains and
/// per-object shader overrides land later as separate components.
#[derive(bevy_ecs::resource::Resource, Debug, Clone)]
pub struct RenderSettings {
    /// Which shader the scene's main render pass uses. Resolved
    /// through [`ShaderRegistry::ensure_or_default`], so an unknown
    /// path silently falls back to the built-in default and the
    /// renderer keeps drawing.
    ///
    /// Wrapped in `Arc` so per-frame consumers (the runtime / viewport
    /// render passes that pull this off the world to drive
    /// `pipeline_for`) refcount-bump rather than deep-cloning a
    /// `String` every frame.
    pub default_shader: Arc<AssetPath>,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            default_shader: engine_default_shader_path(),
        }
    }
}

/// Process-wide shared `Arc<AssetPath>` for the engine default shader.
/// Materialised once and handed back on every fallback so the per-frame
/// "no `RenderSettings` resource present" path doesn't allocate a fresh
/// `AssetPath` each frame.
#[must_use]
pub fn engine_default_shader_path() -> Arc<AssetPath> {
    static CELL: std::sync::OnceLock<Arc<AssetPath>> = std::sync::OnceLock::new();
    CELL.get_or_init(|| Arc::new(AssetPath::new(ENGINE_SHADER_DEFAULT)))
        .clone()
}
