//! Buffer, texture, and sampler descriptors. Descs are borrowing, short-lived,
//! and lifetime-tagged so backends can borrow labels and slices into their
//! creation paths without forcing allocations at the call site.

use bitflags::bitflags;

use crate::format::TextureFormat;

// ────────────────────────── Buffer ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: MemoryLocation,
}

bitflags! {
    /// How a buffer will be used. Backends use this to pick memory types and
    /// validate bind groups. OR the flags that apply: `VERTEX | COPY_DST`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        const VERTEX   = 1 << 0;
        const INDEX    = 1 << 1;
        const UNIFORM  = 1 << 2;
        const STORAGE  = 1 << 3;
        const INDIRECT = 1 << 4;
        const COPY_SRC = 1 << 5;
        const COPY_DST = 1 << 6;
        /// Mappable for CPU read-back after GPU writes (e.g. readback buffers).
        const MAP_READ  = 1 << 7;
        /// Mappable for CPU writes before GPU reads (e.g. staging uploads).
        const MAP_WRITE = 1 << 8;
    }
}

/// Where a resource lives physically. `Device` is GPU-local (fastest for GPU
/// reads); `Upload` is CPU-visible host memory (per-frame dynamic data);
/// `Readback` is GPU-writable host memory (for CPU read-back).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    Device,
    Upload,
    Readback,
}

// ────────────────────────── Texture ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TextureDesc<'a> {
    pub label: Option<&'a str>,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_levels: u32,
    pub sample_count: u32,
}

/// Texture shape + size. `D2` covers both 2D and 2D array via `array_layers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDimension {
    D1 {
        width: u32,
    },
    D2 {
        width: u32,
        height: u32,
        array_layers: u32,
    },
    D3 {
        width: u32,
        height: u32,
        depth: u32,
    },
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        const COPY_SRC          = 1 << 0;
        const COPY_DST          = 1 << 1;
        const TEXTURE_BINDING   = 1 << 2;
        const STORAGE_BINDING   = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

/// A sub-region of a texture, used for partial uploads and copies.
#[derive(Debug, Clone, Copy)]
pub struct TextureRegion {
    pub mip_level: u32,
    pub array_layer: u32,
    pub origin: [u32; 3],
    pub extent: [u32; 3],
}

// ────────────────────────── Sampler ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct SamplerDesc<'a> {
    pub label: Option<&'a str>,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min: f32,
    pub lod_max: f32,
    pub compare: Option<CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color: Option<BorderColor>,
}

impl Default for SamplerDesc<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min: 0.0,
            lod_max: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}
