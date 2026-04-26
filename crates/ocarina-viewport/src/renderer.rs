//! `ViewportRenderer` — the renderer core for both the editor's
//! viewport panel and (future) the shipped runtime's player window.
//!
//! Reusable design: takes a flat list of [`EntityInstance`]s, a
//! [`ViewportCamera`], and a [`RenderMode`], and renders into an
//! arbitrary [`WgpuPassCtx::view`] at the supplied rect using
//! `LoadOp::Load` (so a UI compositor can paint underneath without
//! being clobbered).
//!
//! The renderer owns its own depth target (recreated when the panel
//! resizes) but does **not** own a swapchain — it always renders into
//! the caller-supplied target view. That's what lets the same code
//! serve both the editor (target = editor swapchain) and the runtime
//! (target = game window swapchain).
//!
//! Held entirely on the RHI (`ocarina-rhi`), so a future Vulkan /
//! Metal / D3D12 backend slots in without changing this file.

use bevy_ecs::entity::Entity;
use bytemuck::{Pod, Zeroable};
use floem_renderer::WgpuPassCtx;
use glam::{Mat4, Vec3, Vec4};

use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindGroupLayoutHandle, BindingResource, BindingType, BufferDesc, BufferHandle, BufferUsage,
    ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentStage,
    GraphicsPipelineDesc, MemoryLocation, MultisampleState, PipelineDesc, PipelineHandle,
    PipelineKind, PrimitiveState, Rhi, RhiError, RhiResult, ShaderBytecode, ShaderDesc,
    ShaderHandle, ShaderStage, ShaderStageFlags, TextureDesc, TextureDimension, TextureFormat,
    TextureHandle, TextureUsage, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStage,
    VertexStepMode,
};
use ocarina_rhi_wgpu::WgpuRhi;

use crate::mesh::{self, Vertex};

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;
const LIT_WGSL: &str = include_str!("../shaders/lit.wgsl");
const OUTLINE_WGSL: &str = include_str!("../shaders/outline.wgsl");

/// std140-laid-out camera UBO consumed by `lit.wgsl` at bind group 0.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
    light_dir: Vec4,
    base_color: Vec4,
}
const CAMERA_UNIFORM_SIZE: u64 = std::mem::size_of::<CameraUniform>() as u64;

/// Per-instance UBO at bind group 1, dynamic-offset-stepped through a
/// single buffer of length `INSTANCE_STRIDE * MAX_INSTANCES`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    model: Mat4,
    tint: Vec4,
}
const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;
const INSTANCE_STRIDE: u64 = 256;
const MAX_INSTANCES: u32 = 1024;

/// Camera state ready to upload to the GPU. Computed by the caller —
/// the renderer doesn't know about orbit cameras (editor) vs gameplay
/// cameras (runtime).
#[derive(Clone, Copy)]
pub struct ViewportCamera {
    pub view_proj: Mat4,
    pub light_dir: Vec3,
    /// Multiplied into per-instance tint by the fragment shader. Use
    /// `Vec4::ONE` for "no tint, just per-instance".
    pub base_color: Vec4,
}

impl ViewportCamera {
    /// Standard daylight setup: light from above-and-behind-right,
    /// neutral white base color.
    #[must_use]
    pub fn standard(view_proj: Mat4) -> Self {
        Self {
            view_proj,
            light_dir: Vec3::new(0.35, 1.0, 0.85).normalize(),
            base_color: Vec4::ONE,
        }
    }
}

/// What the renderer is rendering for. Editor = preview with selection
/// highlight (and, later, gizmos / wireframes). Play = pure scene
/// render, no editor overlays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Edit { selected_entity: Option<Entity> },
    Play,
}

/// One renderable entity. The caller is responsible for resolving
/// world transforms (folding parent transforms in the editor case;
/// reading from the runtime's transform system in the play case).
#[derive(Clone, Copy)]
pub struct EntityInstance {
    pub entity: Entity,
    pub world_xform: Mat4,
    /// Multiplied with the camera's `base_color` and the `selected`
    /// boost (in edit mode). Pass `Vec4::new(0.78, 0.82, 0.90, 1.0)`
    /// for the default "neutral cool grey" Ocarina aesthetic.
    pub tint: Vec4,
}

/// All persistent GPU state for the viewport renderer.
pub struct ViewportRenderer {
    rhi: WgpuRhi,

    pipeline: PipelineHandle,
    /// Inverted-hull pipeline: redraws the selected mesh after the lit
    /// pass with cull_mode = Front and a normal-direction vertex push
    /// to produce a screen-space silhouette outline.
    outline_pipeline: PipelineHandle,
    camera_bg: BindGroupHandle,
    instance_bg: BindGroupHandle,
    camera_buffer: BufferHandle,
    instance_buffer: BufferHandle,
    cube_vb: BufferHandle,
    cube_ib: BufferHandle,
    cube_index_count: u32,

    target_format: TextureFormat,

    depth: Option<TextureHandle>,
    depth_size: (u32, u32),
}

impl ViewportRenderer {
    /// Build a renderer that shares wgpu resources with an existing
    /// device + queue + adapter — the Floem editor case. The render
    /// pipeline targets `format` (the swapchain's backbuffer format).
    pub fn from_existing(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> RhiResult<Self> {
        let target_format = wgpu_format_to_rhi(format)?;
        let mut rhi = WgpuRhi::from_existing(device, queue, adapter);

        // Bind group 0: camera UBO.
        let camera_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("camera.bgl"),
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
            label: Some("camera.ubo"),
            size: CAMERA_UNIFORM_SIZE,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let camera_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("camera.bg"),
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

        // Bind group 1: per-instance UBO with dynamic offset.
        let instance_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("instance.bgl"),
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
            label: Some("instance.ubo"),
            size: INSTANCE_STRIDE * u64::from(MAX_INSTANCES),
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let instance_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("instance.bg"),
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

        // Cube mesh (placeholder geometry — every entity with Transform
        // currently renders as a unit cube; replaced when mesh assets
        // come online in Phase 6).
        let cube = mesh::build_unit_cube();
        let (cube_vb, cube_ib, cube_index_count) = upload_mesh(&mut rhi, "cube", &cube)?;

        // Lit shader pipeline.
        let vs = rhi.create_shader(&ShaderDesc {
            label: Some("lit.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
        })?;
        let fs = rhi.create_shader(&ShaderDesc {
            label: Some("lit.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
        })?;
        let pipeline =
            build_pipeline(&mut rhi, vs, fs, camera_bgl, instance_bgl, target_format)?;
        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);

        // Outline pipeline: same vertex/instance layout, different
        // shader + cull-front to produce a silhouette outline of the
        // selected mesh.
        let outline_vs = rhi.create_shader(&ShaderDesc {
            label: Some("outline.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_WGSL),
        })?;
        let outline_fs = rhi.create_shader(&ShaderDesc {
            label: Some("outline.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_WGSL),
        })?;
        let outline_pipeline = build_outline_pipeline(
            &mut rhi,
            outline_vs,
            outline_fs,
            camera_bgl,
            instance_bgl,
            target_format,
        )?;
        rhi.destroy_shader(outline_vs);
        rhi.destroy_shader(outline_fs);

        Ok(Self {
            rhi,
            pipeline,
            outline_pipeline,
            camera_bg,
            instance_bg,
            camera_buffer,
            instance_buffer,
            cube_vb,
            cube_ib,
            cube_index_count,
            target_format,
            depth: None,
            depth_size: (0, 0),
        })
    }

    /// Render `entities` into `ctx.view` at `ctx.rect` (in framebuffer
    /// pixels). Uses `LoadOp::Load` for color so a UI compositor's
    /// underlying paint is preserved; `LoadOp::Clear(1.0)` for depth.
    pub fn render(
        &mut self,
        entities: &[EntityInstance],
        ctx: &WgpuPassCtx<'_>,
        camera: ViewportCamera,
        mode: RenderMode,
    ) {
        let pixel_w = ctx.rect.width().round() as u32;
        let pixel_h = ctx.rect.height().round() as u32;
        if pixel_w == 0 || pixel_h == 0 {
            return;
        }

        // The depth attachment must match the *color* attachment's
        // extent — wgpu enforces "all attachments in a render pass
        // share the same size." Our color attachment is the host
        // swapchain texture (`ctx.view`), full size `ctx.target_size`.
        // Using `ctx.rect` for the depth here would trip the
        // "Attachments have differing sizes" validation. Drawing is
        // still confined to the panel by `set_viewport`/`set_scissor`
        // below.
        let (depth_w, depth_h) = ctx.target_size;
        if depth_w == 0 || depth_h == 0 {
            return;
        }
        if self.depth_size != (depth_w, depth_h) {
            if let Some(d) = self.depth.take() {
                self.rhi.destroy_texture(d);
            }
            match self.rhi.create_texture(&TextureDesc {
                label: Some("viewport.depth"),
                dimension: TextureDimension::D2 {
                    width: depth_w,
                    height: depth_h,
                    array_layers: 1,
                },
                format: DEPTH_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT,
                mip_levels: 1,
                sample_count: 1,
            }) {
                Ok(d) => {
                    self.depth = Some(d);
                    self.depth_size = (depth_w, depth_h);
                }
                Err(e) => {
                    log::warn!("viewport: depth create failed: {e}");
                    return;
                }
            }
        }
        let Some(depth) = self.depth else {
            return;
        };

        // ─── Mutable RHI ops (uploads) up front, before we acquire
        // any immutable borrows for the render pass. ──────────────

        let cam_uniform = CameraUniform {
            view_proj: camera.view_proj,
            light_dir: Vec4::new(camera.light_dir.x, camera.light_dir.y, camera.light_dir.z, 0.0),
            base_color: camera.base_color,
        };
        self.rhi
            .write_buffer(self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        let selected = match mode {
            RenderMode::Edit { selected_entity } => selected_entity,
            RenderMode::Play => None,
        };
        let count = entities.len().min(MAX_INSTANCES as usize);
        if count > 0 {
            let mut instance_bytes: Vec<u8> = vec![0; count * INSTANCE_STRIDE as usize];
            for (i, inst) in entities.iter().take(count).enumerate() {
                let highlight = selected == Some(inst.entity);
                let final_tint = if highlight {
                    inst.tint * 1.4 + Vec4::splat(0.15)
                } else {
                    inst.tint
                };
                let offset = i * INSTANCE_STRIDE as usize;
                let uniform = InstanceUniform {
                    model: inst.world_xform,
                    tint: final_tint,
                };
                instance_bytes[offset..offset + INSTANCE_UNIFORM_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&uniform));
            }
            self.rhi.write_buffer(self.instance_buffer, 0, &instance_bytes);
        }

        // ─── Now resolve all the wgpu objects we need for the
        // render pass. All of these are immutable borrows from the
        // RHI's pools — fine to hold simultaneously. ──────────────

        let Some(depth_view) = self.rhi.texture_view(depth) else {
            return;
        };
        let Some(pipeline) = self.rhi.graphics_pipeline(self.pipeline) else {
            return;
        };
        let Some(outline_pipeline) =
            self.rhi.graphics_pipeline(self.outline_pipeline)
        else {
            return;
        };
        let Some(camera_bg) = self.rhi.bind_group(self.camera_bg) else {
            return;
        };
        let Some(instance_bg) = self.rhi.bind_group(self.instance_bg) else {
            return;
        };
        let Some(cube_vb) = self.rhi.buffer(self.cube_vb) else {
            return;
        };
        let Some(cube_ib) = self.rhi.buffer(self.cube_ib) else {
            return;
        };

        // Find the instance index of the selected entity, if any. The
        // outline pass below redraws just that one through the
        // outline pipeline. `entities` is sorted by Entity in the
        // caller, so this is the same index the main draw used.
        let outline_instance: Option<u32> = match mode {
            RenderMode::Edit { selected_entity: Some(sel) } => entities[..count]
                .iter()
                .position(|inst| inst.entity == sel)
                .map(|i| i as u32),
            _ => None,
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ocarina viewport encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina viewport pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_viewport(
                ctx.rect.x0 as f32,
                ctx.rect.y0 as f32,
                ctx.rect.width() as f32,
                ctx.rect.height() as f32,
                0.0,
                1.0,
            );
            // Clamp scissor to the target so a panel that extends past
            // the swapchain edge (e.g. mid-resize) doesn't trigger
            // wgpu's "scissor outside attachment" validation.
            let sx = ctx.rect.x0.max(0.0) as u32;
            let sy = ctx.rect.y0.max(0.0) as u32;
            let sw = pixel_w.min(depth_w.saturating_sub(sx));
            let sh = pixel_h.min(depth_h.saturating_sub(sy));
            if sw == 0 || sh == 0 {
                return;
            }
            pass.set_scissor_rect(sx, sy, sw, sh);

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera_bg, &[]);
            pass.set_vertex_buffer(0, cube_vb.slice(..));
            pass.set_index_buffer(cube_ib.slice(..), wgpu::IndexFormat::Uint32);

            for i in 0..count {
                let dyn_offset = (i as u64 * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.draw_indexed(0..self.cube_index_count, 0, 0..1);
            }

            // Outline pass: redraw the selected entity through the
            // inverted-hull pipeline. Inherits the depth values written
            // by the main pass above (same render pass), so the hull is
            // depth-tested against the actual mesh and only visible at
            // the silhouette boundary.
            if let Some(outline_idx) = outline_instance {
                let dyn_offset = (u64::from(outline_idx) * INSTANCE_STRIDE) as u32;
                pass.set_pipeline(outline_pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.set_vertex_buffer(0, cube_vb.slice(..));
                pass.set_index_buffer(cube_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.cube_index_count, 0, 0..1);
            }
        }
        ctx.queue.submit([encoder.finish()]);
    }

    /// True if the renderer's pipeline still matches the given target
    /// format. Used by the panel binding to decide whether to rebuild
    /// the renderer when (rarely) the swapchain format changes.
    #[must_use]
    pub fn matches_target_format(&self, format: wgpu::TextureFormat) -> bool {
        wgpu_format_to_rhi(format).map_or(false, |f| f == self.target_format)
    }
}

fn upload_mesh(
    rhi: &mut WgpuRhi,
    label: &str,
    mesh: &mesh::Mesh,
) -> RhiResult<(BufferHandle, BufferHandle, u32)> {
    let vb_size = (mesh.vertices.len() * Vertex::SIZE) as u64;
    let vb = rhi.create_buffer(&BufferDesc {
        label: Some(&format!("{label}.vb")),
        size: vb_size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(vb, 0, bytemuck::cast_slice(&mesh.vertices));

    let ib_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
    let ib = rhi.create_buffer(&BufferDesc {
        label: Some(&format!("{label}.ib")),
        size: ib_size,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(ib, 0, bytemuck::cast_slice(&mesh.indices));

    Ok((vb, ib, mesh.indices.len() as u32))
}

fn build_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
    color_format: TextureFormat,
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
        format: color_format,
        blend: Some(ocarina_rhi::BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("lit"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets: &color_targets,
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
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

/// Outline pipeline. Same vertex layout as `build_pipeline` but
/// `cull_mode = Front` (so the inverted hull's back faces are what
/// actually gets drawn — they form the silhouette around the original
/// mesh) and `depth_write_enabled = false` (the outline shouldn't
/// disturb depth for anything drawn after it).
fn build_outline_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
    color_format: TextureFormat,
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
        format: color_format,
        blend: Some(ocarina_rhi::BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("outline"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets: &color_targets,
            }),
            primitive: PrimitiveState {
                cull_mode: Some(ocarina_rhi::Face::Front),
                ..PrimitiveState::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                // LessEqual: outline pixel passes when at or in front
                // of the existing depth. Where the main mesh wrote
                // depth, the hull's pushed-out z is slightly greater
                // (behind), so it fails — no draw inside the mesh's
                // silhouette. In empty area depth is 1.0 (cleared);
                // hull z < 1.0, so it draws. Net effect: outline
                // visible only at the silhouette edge.
                depth_compare: CompareFunction::LessEqual,
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

fn wgpu_format_to_rhi(fmt: wgpu::TextureFormat) -> RhiResult<TextureFormat> {
    use wgpu::TextureFormat as W;
    Ok(match fmt {
        W::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        W::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        W::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        W::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        other => {
            return Err(RhiError::Backend(format!(
                "unsupported swapchain texture format for viewport renderer: {other:?}"
            )));
        }
    })
}
