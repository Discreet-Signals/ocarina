//! Sky pipeline + UBO.
//!
//! Renders a 3-stop gradient skybox as a fullscreen quad before the
//! lit pass. The lit pass switches its `LoadOp` from `Clear` to
//! `Load` whenever a sky pass ran first (the renderer hosts manage
//! that bookkeeping), so the sky becomes the "background" for the
//! lit scene. Without a `Sky` component in the world, the host
//! falls back to clearing to a fixed editor-grey colour.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindGroupLayoutHandle, BindingResource, BindingType, BufferDesc, BufferHandle, BufferUsage,
    ColorTargetState, ColorWrites, FragmentStage, GraphicsPipelineDesc, MemoryLocation,
    MultisampleState, PipelineDesc, PipelineHandle, PipelineKind, PrimitiveState, Rhi, RhiResult,
    ShaderBytecode, ShaderDesc, ShaderStage, ShaderStageFlags, TextureFormat, VertexStage,
};

/// Sky UBO. 144 bytes, std140-aligned. Single bind group at
/// `@group(0) @binding(0)` for the sky pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SkyUniform {
    pub inv_view_proj: Mat4,
    pub eye_pos: Vec4,
    pub top_color: Vec4,
    pub horizon_color: Vec4,
    pub bottom_color: Vec4,
    /// x = horizon_falloff, y = intensity, z/w = padding.
    pub params: Vec4,
}

impl SkyUniform {
    /// Build from the live `Sky` component + the camera's view-proj.
    /// `view_proj` is the same matrix uploaded in `CameraUniform`;
    /// we invert it on the CPU here so the shader doesn't have to.
    #[must_use]
    pub fn build(
        sky: &ocarina_core::components::Sky,
        view_proj: Mat4,
        eye_pos: Vec3,
    ) -> Self {
        Self {
            inv_view_proj: view_proj.inverse(),
            eye_pos: eye_pos.extend(0.0),
            top_color: sky.top_color.extend(1.0),
            horizon_color: sky.horizon_color.extend(1.0),
            bottom_color: sky.bottom_color.extend(1.0),
            params: Vec4::new(sky.horizon_falloff, sky.intensity, 0.0, 0.0),
        }
    }
}

pub const SKY_UNIFORM_SIZE: u64 = std::mem::size_of::<SkyUniform>() as u64;

const SKY_WGSL: &str = include_str!("../shaders/sky.wgsl");

/// All RHI handles for the sky pipeline. One sky buffer, one bind
/// group, one pipeline. Hosts hold one of these for the lifetime of
/// the renderer.
pub struct SkyPipeline {
    pub pipeline: PipelineHandle,
    pub buffer: BufferHandle,
    pub bgl: BindGroupLayoutHandle,
    pub bg: BindGroupHandle,
}

pub fn build_sky_pipeline(
    rhi: &mut dyn Rhi,
    target_format: TextureFormat,
    depth_format: TextureFormat,
) -> RhiResult<SkyPipeline> {
    let bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.sky.bgl"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::FRAGMENT | ShaderStageFlags::VERTEX,
            binding_type: BindingType::UniformBuffer {
                has_dynamic_offset: false,
                min_binding_size: Some(SKY_UNIFORM_SIZE),
            },
            count: None,
        }],
    })?;
    let buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.sky.ubo"),
        size: SKY_UNIFORM_SIZE,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.sky.bg"),
        layout: bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer,
                offset: 0,
                size: Some(SKY_UNIFORM_SIZE),
            },
        }],
    })?;

    let vs = rhi.create_shader(&ShaderDesc {
        label: Some("render.sky.wgsl:vs_main"),
        stage: ShaderStage::Vertex,
        entry_point: "vs_main",
        bytecode: ShaderBytecode::Wgsl(SKY_WGSL),
    })?;
    let fs = rhi.create_shader(&ShaderDesc {
        label: Some("render.sky.wgsl:fs_main"),
        stage: ShaderStage::Fragment,
        entry_point: "fs_main",
        bytecode: ShaderBytecode::Wgsl(SKY_WGSL),
    })?;

    // Opaque sky: full overwrite of any existing pixels in the
    // colour attachment, no blending. Sky pass runs first; the lit
    // pass that follows uses `LoadOp::Load` to paint geometry over
    // the sky background.
    let color_targets = [Some(ColorTargetState {
        format: target_format,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];

    let pipeline = rhi.create_pipeline(&PipelineDesc {
        label: Some("render.sky"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &[],
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets: &color_targets,
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
            // No depth -- sky pass runs first, no depth attachment.
            // Subsequent lit pass clears depth and renders normally
            // on top of whatever sky wrote into the colour buffer.
            depth_stencil: None,
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });

    rhi.destroy_shader(vs);
    rhi.destroy_shader(fs);

    let _ = depth_format; // depth not used by sky pipeline; arg kept for API consistency.

    let pipeline = pipeline?;

    Ok(SkyPipeline {
        pipeline,
        buffer,
        bgl,
        bg,
    })
}
