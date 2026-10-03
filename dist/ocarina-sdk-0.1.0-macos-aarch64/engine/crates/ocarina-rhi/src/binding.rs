//! Bind group layouts and bind groups.
//!
//! Bind group layouts are first-class objects (required for mapping to VK
//! descriptor set layouts and D3D12 root signatures). Pipelines list the
//! layouts they expect; bind groups are created against a specific layout
//! and attached at draw time.

use crate::format::TextureFormat;
use crate::handle::{BindGroupLayoutHandle, BufferHandle, SamplerHandle, TextureHandle};
use crate::shader::ShaderStageFlags;

// ────────────────────────── Layout ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BindGroupLayoutDesc<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupLayoutEntry],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStageFlags,
    pub binding_type: BindingType,
    /// `None` = single binding; `Some(n)` = array of n (bindless if
    /// `Features::BINDLESS_ARRAYS` and n is unbounded via a large value).
    pub count: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
    UniformBuffer {
        has_dynamic_offset: bool,
        min_binding_size: Option<u64>,
    },
    StorageBuffer {
        read_only: bool,
        has_dynamic_offset: bool,
        min_binding_size: Option<u64>,
    },
    Sampler(SamplerBindingType),
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

// ────────────────────────── Bind group ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BindGroupDesc<'a> {
    pub label: Option<&'a str>,
    pub layout: BindGroupLayoutHandle,
    pub entries: &'a [BindGroupEntry],
}

#[derive(Debug, Clone, Copy)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Debug, Clone, Copy)]
pub enum BindingResource {
    Buffer {
        buffer: BufferHandle,
        offset: u64,
        /// `None` = remainder of the buffer from `offset`.
        size: Option<u64>,
    },
    Texture(TextureHandle),
    Sampler(SamplerHandle),
}
