//! Capabilities a backend advertises. Engine code queries these before using
//! optional features; adding a new feature later is a new bit in [`Features`]
//! plus a new entry point on [`crate::Rhi`], not a trait-breaking change.

use bitflags::bitflags;

#[derive(Debug, Clone)]
pub struct Capabilities {
    pub backend_name: &'static str,
    pub device_name: String,
    pub adapter_kind: AdapterKind,
    pub features: Features,
    pub limits: Limits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Unknown,
}

bitflags! {
    /// Optional GPU feature bits. Backends advertise which they support;
    /// engine code gates usage behind capability checks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Features: u64 {
        const MESH_SHADERS              = 1 << 0;
        const RAY_TRACING               = 1 << 1;
        const BINDLESS_ARRAYS           = 1 << 2;
        const TIMESTAMP_QUERIES         = 1 << 3;
        const PIPELINE_STATISTICS       = 1 << 4;
        const DEPTH_CLIP_CONTROL        = 1 << 5;
        const INDIRECT_FIRST_INSTANCE   = 1 << 6;
        const MULTI_DRAW_INDIRECT       = 1 << 7;
        const SHADER_F16                = 1 << 8;
        const SHADER_F64                = 1 << 9;
        const SUBGROUP_OPS              = 1 << 10;
        const PUSH_CONSTANTS            = 1 << 11;
        const TEXTURE_COMPRESSION_BC    = 1 << 12;
        const TEXTURE_COMPRESSION_ASTC  = 1 << 13;
        const DEPTH_CLAMPING            = 1 << 14;
        const POLYGON_MODE_LINE         = 1 << 15;
        const POLYGON_MODE_POINT        = 1 << 16;
    }
}

/// Per-backend limits. Follows the WebGPU shape closely (because wgpu is
/// backend #1) but the fields are generic enough to be honest across VK,
/// D3D12, and Metal. Engine code should query these rather than hard-coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_vertex_buffers: u32,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffer_array_stride: u32,
    pub max_push_constant_size: u32,
    pub max_color_attachments: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl Limits {
    /// Conservative limits that any serious backend should meet. Good default
    /// for the null backend and for engine code that doesn't want to branch
    /// on hardware variation.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_texture_dimension_1d: 2048,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 640,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 64 * 1024,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_push_constant_size: 128,
            max_color_attachments: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}
