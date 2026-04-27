//! Texture data, registry, and asset loading.
//!
//! Parallel to [`MeshRegistry`](crate::mesh::MeshRegistry):
//! [`AssetPath`]-keyed cache of GPU textures, engine builtins
//! (`engine/textures/white`, `engine/textures/black`,
//! `engine/textures/normal`) baked into the binary and pre-uploaded
//! at startup, project-relative user textures (PNG / JPEG via the
//! `image` crate) loaded and cached on first reference.
//!
//! Color-space matters for correctness: albedo / emission textures
//! are stored sRGB-encoded and want hardware sRGB → linear decoding
//! on sample, while normal / metallic-roughness textures store
//! linear data (the byte values *are* the data) and must NOT be
//! sRGB-decoded. The registry serves both formats from the same
//! source path -- caching by `(path, ColorSpace)` -- so a single
//! file can be bound either way without re-uploading.
//!
//! The `engine/` path prefix is reserved -- the same convention
//! [`MeshRegistry`](crate::mesh::MeshRegistry) uses.

use std::collections::{HashMap, HashSet};

use ocarina_core::AssetPath;
use ocarina_rhi::{
    AddressMode, FilterMode, MemoryLocation, Rhi, RhiResult, SamplerDesc, SamplerHandle,
    TextureDesc, TextureDimension, TextureFormat, TextureHandle, TextureRegion, TextureUsage,
};

// ─────────────────────────── Color space ───────────────────────────

/// Colour-space hint for `TextureRegistry::ensure`. Drives whether
/// the GPU texture is created as `Rgba8UnormSrgb` (hardware
/// gamma-decode on sample) or `Rgba8Unorm` (raw linear bytes).
///
/// Pick `Srgb` for albedo / emission textures (anything authored as
/// "what you see on screen" colour). Pick `Linear` for normal /
/// metallic-roughness / data textures (anything where the bytes are
/// raw values, not gamma-encoded colour).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

impl ColorSpace {
    fn texture_format(self) -> TextureFormat {
        match self {
            Self::Srgb => TextureFormat::Rgba8UnormSrgb,
            Self::Linear => TextureFormat::Rgba8Unorm,
        }
    }
}

// ─────────────────────────── Built-in catalogue ───────────────────────────

/// Reserved path prefix for engine-shipped textures.
pub const ENGINE_TEXTURE_PREFIX: &str = "engine/textures/";

/// Engine path of the 1×1 opaque white texture.
pub const ENGINE_TEXTURE_WHITE: &str = "engine/textures/white";
/// Engine path of the 1×1 opaque black texture.
pub const ENGINE_TEXTURE_BLACK: &str = "engine/textures/black";
/// Engine path of the 1×1 flat-normal texture (RGB = 128, 128, 255 →
/// tangent-space normal `(0, 0, 1)` after the standard
/// `n = 2 * sample - 1` decode). Always sampled `Linear`.
pub const ENGINE_TEXTURE_NORMAL: &str = "engine/textures/normal";

// ─────────────────────────── Registry ───────────────────────────

/// Owns the GPU textures backing every material in use, plus a
/// shared linear sampler. Engine textures are uploaded at
/// construction; user textures load and cache the first time a
/// material references them.
pub struct TextureRegistry {
    project_root: Option<std::path::PathBuf>,
    cache: HashMap<(AssetPath, ColorSpace), TextureHandle>,
    /// Failures we've already logged so we don't spam every frame.
    failed: HashSet<(AssetPath, ColorSpace)>,
    /// Linear sampler shared by every material draw. Wrap-repeat by
    /// default (the standard convention for tileable PBR maps);
    /// users wanting clamp-to-edge can swap the sampler in stage 8.
    sampler: SamplerHandle,
    /// 1×1 white textures pre-cached for both color spaces. Returned
    /// by [`ensure_or_white`](Self::ensure_or_white) when a material
    /// has no user texture in that slot, so the renderer's bind
    /// group layout stays uniform across draws.
    white_srgb: TextureHandle,
    white_linear: TextureHandle,
    /// 1×1 black textures, used as the default for emission and
    /// metallic-roughness slots.
    black_srgb: TextureHandle,
    black_linear: TextureHandle,
    /// 1×1 flat-normal texture, used as the default normal map.
    flat_normal: TextureHandle,
}

impl TextureRegistry {
    /// Construct a new registry, creating the shared sampler and
    /// the built-in 1×1 textures up-front. After this returns, every
    /// `ensure_or_*` call has a valid handle to fall back to.
    pub fn new(rhi: &mut dyn Rhi) -> RhiResult<Self> {
        let sampler = rhi.create_sampler(&SamplerDesc {
            label: Some("render.texture.sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..SamplerDesc::default()
        })?;

        let white_srgb = upload_1x1(
            rhi,
            "render.texture.white.srgb",
            [255, 255, 255, 255],
            TextureFormat::Rgba8UnormSrgb,
        )?;
        let white_linear = upload_1x1(
            rhi,
            "render.texture.white.linear",
            [255, 255, 255, 255],
            TextureFormat::Rgba8Unorm,
        )?;
        let black_srgb = upload_1x1(
            rhi,
            "render.texture.black.srgb",
            [0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
        )?;
        let black_linear = upload_1x1(
            rhi,
            "render.texture.black.linear",
            [0, 0, 0, 255],
            TextureFormat::Rgba8Unorm,
        )?;
        let flat_normal = upload_1x1(
            rhi,
            "render.texture.flat_normal",
            [128, 128, 255, 255],
            TextureFormat::Rgba8Unorm,
        )?;

        let mut cache: HashMap<(AssetPath, ColorSpace), TextureHandle> = HashMap::new();
        cache.insert(
            (AssetPath::new(ENGINE_TEXTURE_WHITE), ColorSpace::Srgb),
            white_srgb,
        );
        cache.insert(
            (AssetPath::new(ENGINE_TEXTURE_WHITE), ColorSpace::Linear),
            white_linear,
        );
        cache.insert(
            (AssetPath::new(ENGINE_TEXTURE_BLACK), ColorSpace::Srgb),
            black_srgb,
        );
        cache.insert(
            (AssetPath::new(ENGINE_TEXTURE_BLACK), ColorSpace::Linear),
            black_linear,
        );
        cache.insert(
            (AssetPath::new(ENGINE_TEXTURE_NORMAL), ColorSpace::Linear),
            flat_normal,
        );

        Ok(Self {
            project_root: None,
            cache,
            failed: HashSet::new(),
            sampler,
            white_srgb,
            white_linear,
            black_srgb,
            black_linear,
            flat_normal,
        })
    }

    pub fn set_project_root(&mut self, root: std::path::PathBuf) {
        self.project_root = Some(root);
    }

    /// Shared sampler used for every material binding.
    #[must_use]
    pub fn sampler(&self) -> SamplerHandle {
        self.sampler
    }

    /// Resolve an asset path to a GPU texture, loading from disk on
    /// first ask. Returns `None` for an empty path or a load failure
    /// (logged once); callers wanting a guaranteed handle should use
    /// [`ensure_or_white`](Self::ensure_or_white) and friends.
    pub fn ensure(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        space: ColorSpace,
    ) -> Option<TextureHandle> {
        if path.is_empty() {
            return None;
        }
        let key = (path.clone(), space);
        if let Some(h) = self.cache.get(&key) {
            return Some(*h);
        }
        if self.failed.contains(&key) {
            return None;
        }
        // Engine namespace is reserved; an unknown engine path is a
        // user typo, not a missing asset.
        if path.as_str().starts_with(ENGINE_TEXTURE_PREFIX) {
            log::warn!(
                "ocarina-render: unknown engine texture `{}` (known: `{ENGINE_TEXTURE_WHITE}`, \
                 `{ENGINE_TEXTURE_BLACK}`, `{ENGINE_TEXTURE_NORMAL}`)",
                path.as_str(),
            );
            self.failed.insert(key);
            return None;
        }
        match self.load_asset(rhi, path, space) {
            Ok(handle) => {
                self.cache.insert(key, handle);
                Some(handle)
            }
            Err(e) => {
                log::warn!(
                    "ocarina-render: failed to load texture `{}`: {e}",
                    path.as_str()
                );
                self.failed.insert(key);
                None
            }
        }
    }

    /// Load `path` (or fall back to the 1×1 white default) in the
    /// given color space. Always returns a valid handle, so the
    /// renderer's per-material bind group layout stays uniform
    /// across draws.
    pub fn ensure_or_white(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        space: ColorSpace,
    ) -> TextureHandle {
        self.ensure(rhi, path, space).unwrap_or(match space {
            ColorSpace::Srgb => self.white_srgb,
            ColorSpace::Linear => self.white_linear,
        })
    }

    /// Same as [`ensure_or_white`](Self::ensure_or_white) but falls
    /// back to the 1×1 black default. Right choice for emission and
    /// metallic-roughness slots, where "no texture" should mean "no
    /// contribution" not "full white."
    pub fn ensure_or_black(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        space: ColorSpace,
    ) -> TextureHandle {
        self.ensure(rhi, path, space).unwrap_or(match space {
            ColorSpace::Srgb => self.black_srgb,
            ColorSpace::Linear => self.black_linear,
        })
    }

    /// Load a normal map (always linear) or fall back to the 1×1
    /// flat-normal default.
    pub fn ensure_or_flat_normal(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
    ) -> TextureHandle {
        self.ensure(rhi, path, ColorSpace::Linear)
            .unwrap_or(self.flat_normal)
    }

    fn load_asset(
        &self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        space: ColorSpace,
    ) -> Result<TextureHandle, TextureLoadError> {
        let root = self
            .project_root
            .as_ref()
            .ok_or(TextureLoadError::NoProjectRoot)?;
        let absolute = path.resolve(root);
        let label = absolute.display().to_string();
        let pixels = decode_image(&absolute)?;
        upload_rgba8(rhi, &label, &pixels, space.texture_format())
            .map_err(|e| TextureLoadError::Upload(e.to_string()))
    }
}

// ─────────────────────────── Upload helpers ───────────────────────────

fn upload_1x1(
    rhi: &mut dyn Rhi,
    label: &str,
    rgba: [u8; 4],
    format: TextureFormat,
) -> RhiResult<TextureHandle> {
    let handle = rhi.create_texture(&TextureDesc {
        label: Some(label),
        dimension: TextureDimension::D2 {
            width: 1,
            height: 1,
            array_layers: 1,
        },
        format,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        mip_levels: 1,
        sample_count: 1,
    })?;
    rhi.write_texture(
        handle,
        &TextureRegion {
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
            extent: [1, 1, 1],
        },
        &rgba,
        4,
        1,
    );
    Ok(handle)
}

fn upload_rgba8(
    rhi: &mut dyn Rhi,
    label: &str,
    pixels: &DecodedImage,
    format: TextureFormat,
) -> RhiResult<TextureHandle> {
    let handle = rhi.create_texture(&TextureDesc {
        label: Some(label),
        dimension: TextureDimension::D2 {
            width: pixels.width,
            height: pixels.height,
            array_layers: 1,
        },
        format,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        mip_levels: 1,
        sample_count: 1,
    })?;
    rhi.write_texture(
        handle,
        &TextureRegion {
            mip_level: 0,
            array_layer: 0,
            origin: [0, 0, 0],
            extent: [pixels.width, pixels.height, 1],
        },
        &pixels.bytes,
        pixels.width * 4,
        pixels.height,
    );
    let _ = MemoryLocation::Device; // touch import path so it isn't pruned
    Ok(handle)
}

// ─────────────────────────── PNG / JPEG decoder ───────────────────────────

#[derive(Debug)]
pub enum TextureLoadError {
    NoProjectRoot,
    UnsupportedExtension(String),
    Decode(String),
    Upload(String),
}

impl std::fmt::Display for TextureLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProjectRoot => write!(f, "registry has no project root set"),
            Self::UnsupportedExtension(e) => write!(f, "unsupported extension `{e}`"),
            Self::Decode(e) => write!(f, "decode: {e}"),
            Self::Upload(e) => write!(f, "upload: {e}"),
        }
    }
}

impl std::error::Error for TextureLoadError {}

struct DecodedImage {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

#[cfg(feature = "texture")]
fn decode_image(path: &std::path::Path) -> Result<DecodedImage, TextureLoadError> {
    let img = image::open(path)
        .map_err(|e| TextureLoadError::Decode(e.to_string()))?
        .to_rgba8();
    let (width, height) = img.dimensions();
    Ok(DecodedImage {
        width,
        height,
        bytes: img.into_raw(),
    })
}

#[cfg(not(feature = "texture"))]
fn decode_image(_path: &std::path::Path) -> Result<DecodedImage, TextureLoadError> {
    Err(TextureLoadError::UnsupportedExtension(
        "texture feature disabled at compile time".into(),
    ))
}
