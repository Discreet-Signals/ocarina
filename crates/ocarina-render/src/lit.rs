//! Lit pipeline + UBO formats. The shared "draw a mesh with simple
//! lighting" path used by every host.
//!
//! The factory builds:
//!  - one camera UBO (group 0, binding 0; not dynamic-offset).
//!  - one instance UBO (group 1, binding 0; dynamic-offset, single
//!    buffer of `INSTANCE_STRIDE * MAX_INSTANCES` bytes).
//!  - one graphics pipeline using [`Vertex`](crate::mesh::Vertex).
//!
//! Hosts query handles back from the returned [`LitPipeline`] and
//! either record commands via the RHI's `CommandList` (runtime) or
//! drop to wgpu via host-specific accessors (editor / Floem).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindGroupLayoutHandle, BindingResource, BindingType, BufferDesc, BufferHandle, BufferUsage,
    ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentStage,
    GraphicsPipelineDesc, MemoryLocation, MultisampleState, PipelineDesc, PipelineHandle,
    PipelineKind, PrimitiveState, Rhi, RhiResult, ShaderBytecode, ShaderDesc, ShaderHandle,
    ShaderStage, ShaderStageFlags, TextureFormat, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexStage, VertexStepMode,
};

use crate::mesh::Vertex;

// ────────────────────────── UBO formats ──────────────────────────

/// Camera UBO at (0, 0). 80 bytes; 16-byte aligned by std140 rules.
///
/// `viewport_size` is included for layout compatibility with the
/// editor's 2D gizmo shader (it samples pixel-to-NDC conversion off
/// this slot). The lit shader doesn't read it; keeping it here means
/// every consumer of the camera UBO uses the same struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CameraUniform {
    pub view_proj: Mat4,
    pub light_dir: Vec4,
    pub base_color: Vec4,
    pub viewport_size: Vec4,
}

impl CameraUniform {
    /// Standard daylight setup: light from above-and-right, neutral
    /// white base color, viewport size left at zero (only the editor
    /// gizmo shader reads it).
    #[must_use]
    pub fn standard(view_proj: Mat4) -> Self {
        Self {
            view_proj,
            light_dir: Vec3::new(0.35, 1.0, 0.85).normalize().extend(0.0),
            base_color: Vec4::ONE,
            viewport_size: Vec4::ZERO,
        }
    }
}

pub const CAMERA_UNIFORM_SIZE: u64 = std::mem::size_of::<CameraUniform>() as u64;

/// Per-instance UBO at (1, 0). 80 bytes of payload, padded by the
/// stride to satisfy wgpu's `min_uniform_buffer_offset_alignment`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct InstanceUniform {
    pub model: Mat4,
    pub tint: Vec4,
}

pub const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;

/// Stride between consecutive entries in the instance buffer. 256 is
/// the alignment wgpu's default limits report; bumping past that
/// would be a no-op for correctness and a waste of memory.
pub const INSTANCE_STRIDE: u64 = 256;

/// Cap on per-frame instances. Way more than current scenes need;
/// raise once we have evidence it's load-bearing.
pub const MAX_INSTANCES: u32 = 1024;

// ────────────────────────── Pipeline ──────────────────────────

const LIT_WGSL: &str = include_str!("../shaders/lit.wgsl");

/// All RHI handles owned by the lit pipeline factory. Hosts hold
/// one of these for the lifetime of the renderer.
pub struct LitPipeline {
    pub pipeline: PipelineHandle,
    pub camera_buffer: BufferHandle,
    pub camera_bgl: BindGroupLayoutHandle,
    pub camera_bg: BindGroupHandle,
    pub instance_buffer: BufferHandle,
    pub instance_bgl: BindGroupLayoutHandle,
    pub instance_bg: BindGroupHandle,
}

/// Build the lit pipeline + UBOs against `rhi`. `target_format` is
/// the swapchain / panel-target color format the pipeline will be
/// drawn into. `depth_format` is the renderer's depth attachment.
///
/// Built shaders are destroyed before return — the pipeline retains
/// its compiled module internally.
pub fn build_lit_pipeline(
    rhi: &mut dyn Rhi,
    target_format: TextureFormat,
    depth_format: TextureFormat,
) -> RhiResult<LitPipeline> {
    let camera_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.camera.bgl"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::ALL_GRAPHICS,
            binding_type: BindingType::UniformBuffer {
                has_dynamic_offset: false,
                min_binding_size: Some(CAMERA_UNIFORM_SIZE),
            },
            count: None,
        }],
    })?;
    let camera_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.camera.ubo"),
        size: CAMERA_UNIFORM_SIZE,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let camera_bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.camera.bg"),
        layout: camera_bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: camera_buffer,
                offset: 0,
                size: Some(CAMERA_UNIFORM_SIZE),
            },
        }],
    })?;

    let instance_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.instance.bgl"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::ALL_GRAPHICS,
            binding_type: BindingType::UniformBuffer {
                has_dynamic_offset: true,
                min_binding_size: Some(INSTANCE_UNIFORM_SIZE),
            },
            count: None,
        }],
    })?;
    let instance_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.instance.ubo"),
        size: INSTANCE_STRIDE * u64::from(MAX_INSTANCES),
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let instance_bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.instance.bg"),
        layout: instance_bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: instance_buffer,
                offset: 0,
                size: Some(INSTANCE_UNIFORM_SIZE),
            },
        }],
    })?;

    let vs = rhi.create_shader(&ShaderDesc {
        label: Some("render.lit.wgsl:vs_main"),
        stage: ShaderStage::Vertex,
        entry_point: "vs_main",
        bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
    })?;
    let fs = rhi.create_shader(&ShaderDesc {
        label: Some("render.lit.wgsl:fs_main"),
        stage: ShaderStage::Fragment,
        entry_point: "fs_main",
        bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
    })?;

    let pipeline = build_lit_graphics_pipeline(
        rhi,
        vs,
        fs,
        camera_bgl,
        instance_bgl,
        target_format,
        depth_format,
        PrimitiveState {
            cull_mode: None,
            ..PrimitiveState::default()
        },
    )?;

    rhi.destroy_shader(vs);
    rhi.destroy_shader(fs);

    Ok(LitPipeline {
        pipeline,
        camera_buffer,
        camera_bgl,
        camera_bg,
        instance_buffer,
        instance_bgl,
        instance_bg,
    })
}

/// Pipeline factory shared by `build_lit_pipeline` and the editor's
/// wireframe alt pipeline (which differs only in `primitive`).
pub(crate) fn build_lit_graphics_pipeline(
    rhi: &mut dyn Rhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
    target_format: TextureFormat,
    depth_format: TextureFormat,
    primitive: PrimitiveState,
) -> RhiResult<PipelineHandle> {
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 24,
            shader_location: 2,
        },
    ];
    let vertex_buffers = [VertexBufferLayout {
        array_stride: Vertex::SIZE as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &attributes,
    }];
    let color_targets = [Some(ColorTargetState {
        format: target_format,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];

    rhi.create_pipeline(&PipelineDesc {
        label: Some("render.lit"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets: &color_targets,
            }),
            primitive,
            depth_stencil: Some(DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                bias: Default::default(),
            }),
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[camera_bgl, instance_bgl],
        push_constant_ranges: &[],
    })
}
