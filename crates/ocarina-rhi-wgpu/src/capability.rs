//! Translate wgpu capabilities into the RHI's [`Capabilities`] shape.

use ocarina_rhi::{AdapterKind, Capabilities, Features, Limits};

pub fn capabilities_from_adapter(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Capabilities {
    let info = adapter.get_info();
    let wgpu_features = device.features();
    let wgpu_limits = device.limits();

    Capabilities {
        backend_name: super::BACKEND_NAME,
        device_name: info.name.clone(),
        adapter_kind: adapter_kind(info.device_type),
        features: features_from(wgpu_features),
        limits: limits_from(&wgpu_limits),
    }
}

fn adapter_kind(dt: wgpu::DeviceType) -> AdapterKind {
    match dt {
        wgpu::DeviceType::DiscreteGpu => AdapterKind::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu => AdapterKind::IntegratedGpu,
        wgpu::DeviceType::VirtualGpu => AdapterKind::VirtualGpu,
        wgpu::DeviceType::Cpu => AdapterKind::Cpu,
        wgpu::DeviceType::Other => AdapterKind::Unknown,
    }
}

fn features_from(f: wgpu::Features) -> Features {
    let mut out = Features::empty();
    // EXPERIMENTAL_MESH_SHADER isn't in stable wgpu yet; capability bit reserved.
    if f.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        out |= Features::RAY_TRACING;
    }
    if f.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
        && f.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    {
        out |= Features::BINDLESS_ARRAYS;
    }
    if f.contains(wgpu::Features::TIMESTAMP_QUERY) {
        out |= Features::TIMESTAMP_QUERIES;
    }
    if f.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY) {
        out |= Features::PIPELINE_STATISTICS;
    }
    if f.contains(wgpu::Features::DEPTH_CLIP_CONTROL) {
        out |= Features::DEPTH_CLIP_CONTROL;
    }
    if f.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        out |= Features::INDIRECT_FIRST_INSTANCE;
    }
    if f.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        out |= Features::MULTI_DRAW_INDIRECT;
    }
    if f.contains(wgpu::Features::SHADER_F16) {
        out |= Features::SHADER_F16;
    }
    if f.contains(wgpu::Features::SHADER_F64) {
        out |= Features::SHADER_F64;
    }
    if f.contains(wgpu::Features::IMMEDIATES) {
        out |= Features::PUSH_CONSTANTS;
    }
    if f.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        out |= Features::TEXTURE_COMPRESSION_BC;
    }
    if f.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        out |= Features::TEXTURE_COMPRESSION_ASTC;
    }
    if f.contains(wgpu::Features::POLYGON_MODE_LINE) {
        out |= Features::POLYGON_MODE_LINE;
    }
    if f.contains(wgpu::Features::POLYGON_MODE_POINT) {
        out |= Features::POLYGON_MODE_POINT;
    }
    out
}

fn limits_from(l: &wgpu::Limits) -> Limits {
    Limits {
        max_texture_dimension_1d: l.max_texture_dimension_1d,
        max_texture_dimension_2d: l.max_texture_dimension_2d,
        max_texture_dimension_3d: l.max_texture_dimension_3d,
        max_texture_array_layers: l.max_texture_array_layers,
        max_bind_groups: l.max_bind_groups,
        max_bindings_per_bind_group: l.max_bindings_per_bind_group,
        max_dynamic_uniform_buffers_per_pipeline_layout: l
            .max_dynamic_uniform_buffers_per_pipeline_layout,
        max_dynamic_storage_buffers_per_pipeline_layout: l
            .max_dynamic_storage_buffers_per_pipeline_layout,
        max_sampled_textures_per_shader_stage: l.max_sampled_textures_per_shader_stage,
        max_samplers_per_shader_stage: l.max_samplers_per_shader_stage,
        max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage,
        max_storage_textures_per_shader_stage: l.max_storage_textures_per_shader_stage,
        max_uniform_buffers_per_shader_stage: l.max_uniform_buffers_per_shader_stage,
        max_uniform_buffer_binding_size: l.max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
        max_vertex_buffers: l.max_vertex_buffers,
        max_vertex_attributes: l.max_vertex_attributes,
        max_vertex_buffer_array_stride: l.max_vertex_buffer_array_stride,
        max_push_constant_size: l.max_immediate_size,
        max_color_attachments: l.max_color_attachments,
        max_compute_workgroup_size_x: l.max_compute_workgroup_size_x,
        max_compute_workgroup_size_y: l.max_compute_workgroup_size_y,
        max_compute_workgroup_size_z: l.max_compute_workgroup_size_z,
        max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
    }
}
