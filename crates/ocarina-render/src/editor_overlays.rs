//! Editor-only overlay passes: selection mask + outline + 2D gizmos.
//!
//! Compiled in only when the `editor` feature is on -- the runtime
//! binary contains none of this code, none of these struct fields,
//! none of the pipelines. The whole module is `cfg`-gated out of
//! the runtime path the same way [`crate::editor`] is.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindingResource, BindingType, BlendState, BufferDesc, BufferHandle, BufferUsage,
    ColorAttachment, ColorTargetState, ColorWrites, CommandList, CompareFunction,
    DepthStencilAttachment, DepthStencilState, FragmentStage, GraphicsPipelineDesc, IndexFormat,
    LoadOp, MemoryLocation, MultisampleState, PipelineDesc, PipelineHandle, PipelineKind,
    PolygonMode, PrimitiveState, Rect, Rhi, RhiResult, SamplerBindingType, SamplerDesc,
    SamplerHandle, ShaderBytecode, ShaderDesc, ShaderStage, ShaderStageFlags, StoreOp, TextureDesc,
    TextureDimension, TextureFormat, TextureHandle, TextureSampleType, TextureUsage,
    TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStage,
    VertexStepMode, Viewport,
};

use crate::default::{DefaultPipeline, INSTANCE_STRIDE};
use crate::mesh::Vertex;

const MASK_FORMAT: TextureFormat = TextureFormat::R8Unorm;
const MASK_WGSL: &str = include_str!("../shaders/mask.wgsl");
const OUTLINE_POST_WGSL: &str = include_str!("../shaders/outline_post.wgsl");
const GIZMO_2D_WGSL: &str = include_str!("../shaders/gizmo_2d.wgsl");

/// Maximum gizmo handles drawn per frame: 3 translate axes + 3
/// rotate rings. Sized into the gizmo instance buffer below.
const GIZMO_SLOT_COUNT: u64 = 6;

// ─── Public types ──────────────────────────────────────────────

/// Three world-space axes the translate / rotate gizmos draw along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Editor request for translate + rotate gizmo handles, drawn at
/// the current selection's world origin. The renderer expands this
/// into 6 instances (3 translate axes + 3 rotate rings).
#[derive(Clone, Copy, Debug)]
pub struct GizmoOverlay {
    pub origin: glam::Vec3,
    /// Layout extent in world units. Pixel thickness is constant
    /// regardless of camera distance.
    pub scale_world: f32,
    pub active_axis: Option<Axis>,
}

/// One selected mesh to render into the mask. The editor host
/// resolves the selected entity's mesh + per-entity instance slot
/// once and passes this in.
#[derive(Clone, Copy)]
pub struct SelectedDraw {
    pub instance_slot: u32,
    pub vertex_buffer: BufferHandle,
    pub index_buffer: BufferHandle,
    pub index_count: u32,
}

// ─── Persistent overlay state ───────────────────────────────────

pub struct EditorOverlays {
    mask_pipeline: PipelineHandle,
    outline_pipeline: PipelineHandle,
    gizmo_pipeline: PipelineHandle,

    outline_bgl: ocarina_rhi::BindGroupLayoutHandle,
    mask_sampler: SamplerHandle,

    mask_texture: Option<TextureHandle>,
    mask_size: (u32, u32),
    outline_bg: Option<BindGroupHandle>,

    /// Per-frame UBO storage for the 6 gizmo handles.
    gizmo_instance_buffer: BufferHandle,
    /// Bind group binding `gizmo_instance_buffer` against
    /// `default_pipeline.simple_instance_bgl` with a dynamic offset
    /// the record loop uses to pick which gizmo slot a draw reads.
    gizmo_instance_bg: BindGroupHandle,

    translate_axis_vb: BufferHandle,
    translate_axis_ib: BufferHandle,
    translate_axis_index_count: u32,
    rotate_ring_vb: BufferHandle,
    rotate_ring_ib: BufferHandle,
    rotate_ring_index_count: u32,
}

impl EditorOverlays {
    pub fn build(
        rhi: &mut dyn Rhi,
        default_pipeline: &DefaultPipeline,
        target_format: TextureFormat,
        depth_format: TextureFormat,
    ) -> RhiResult<Self> {
        // ── Mask pipeline ──
        let mask_vs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.mask:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(MASK_WGSL),
        })?;
        let mask_fs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.mask:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(MASK_WGSL),
        })?;
        let lit_attributes = [
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
        let lit_vbufs = [VertexBufferLayout {
            array_stride: Vertex::SIZE as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &lit_attributes,
        }];
        let mask_targets = [Some(ColorTargetState {
            format: MASK_FORMAT,
            blend: None,
            write_mask: ColorWrites::ALL,
        })];
        let mask_pipeline = rhi.create_pipeline(&PipelineDesc {
            label: Some("editor.mask"),
            kind: PipelineKind::Graphics(GraphicsPipelineDesc {
                vertex: VertexStage {
                    shader: mask_vs,
                    buffers: &lit_vbufs,
                },
                fragment: Some(FragmentStage {
                    shader: mask_fs,
                    targets: &mask_targets,
                }),
                primitive: PrimitiveState {
                    cull_mode: None,
                    ..PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
            }),
            bind_group_layouts: &[
                default_pipeline.frame_bgl,
                default_pipeline.simple_instance_bgl,
            ],
            push_constant_ranges: &[],
        })?;
        rhi.destroy_shader(mask_vs);
        rhi.destroy_shader(mask_fs);

        // ── Outline post-process pipeline ──
        let outline_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("editor.outline_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })?;
        let mask_sampler = rhi.create_sampler(&SamplerDesc {
            label: Some("editor.mask_sampler"),
            mag_filter: ocarina_rhi::FilterMode::Nearest,
            min_filter: ocarina_rhi::FilterMode::Nearest,
            mipmap_filter: ocarina_rhi::FilterMode::Nearest,
            ..SamplerDesc::default()
        })?;
        let outline_vs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.outline_post:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_POST_WGSL),
        })?;
        let outline_fs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.outline_post:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_POST_WGSL),
        })?;
        let outline_targets = [Some(ColorTargetState {
            format: target_format,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        })];
        let outline_pipeline = rhi.create_pipeline(&PipelineDesc {
            label: Some("editor.outline_post"),
            kind: PipelineKind::Graphics(GraphicsPipelineDesc {
                vertex: VertexStage {
                    shader: outline_vs,
                    buffers: &[],
                },
                fragment: Some(FragmentStage {
                    shader: outline_fs,
                    targets: &outline_targets,
                }),
                primitive: PrimitiveState {
                    cull_mode: None,
                    ..PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
            }),
            bind_group_layouts: &[outline_bgl],
            push_constant_ranges: &[],
        })?;
        rhi.destroy_shader(outline_vs);
        rhi.destroy_shader(outline_fs);

        // ── Gizmo pipeline ──
        let gizmo_vs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.gizmo_2d:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(GIZMO_2D_WGSL),
        })?;
        let gizmo_fs = rhi.create_shader(&ShaderDesc {
            label: Some("editor.gizmo_2d:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(GIZMO_2D_WGSL),
        })?;
        let gizmo_attributes = [
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
        let gizmo_vbufs = [VertexBufferLayout {
            array_stride: GizmoVertex::SIZE as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &gizmo_attributes,
        }];
        let gizmo_targets = [Some(ColorTargetState {
            format: target_format,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        })];
        let gizmo_pipeline = rhi.create_pipeline(&PipelineDesc {
            label: Some("editor.gizmo_2d"),
            kind: PipelineKind::Graphics(GraphicsPipelineDesc {
                vertex: VertexStage {
                    shader: gizmo_vs,
                    buffers: &gizmo_vbufs,
                },
                fragment: Some(FragmentStage {
                    shader: gizmo_fs,
                    targets: &gizmo_targets,
                }),
                primitive: PrimitiveState {
                    cull_mode: None,
                    polygon_mode: PolygonMode::Fill,
                    ..PrimitiveState::default()
                },
                // depth_compare = Always + no depth write so handles
                // render on top of any 3D occlusion.
                depth_stencil: Some(DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil_front: Default::default(),
                    stencil_back: Default::default(),
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                    bias: Default::default(),
                }),
                multisample: MultisampleState::default(),
            }),
            bind_group_layouts: &[
                default_pipeline.frame_bgl,
                default_pipeline.simple_instance_bgl,
            ],
            push_constant_ranges: &[],
        })?;
        rhi.destroy_shader(gizmo_vs);
        rhi.destroy_shader(gizmo_fs);

        // ── Gizmo instance UBO + bind group ──
        let gizmo_instance_buffer = rhi.create_buffer(&BufferDesc {
            label: Some("editor.gizmo_instance"),
            size: GIZMO_SLOT_COUNT * INSTANCE_STRIDE,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let gizmo_instance_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("editor.gizmo_instance.bg"),
            layout: default_pipeline.simple_instance_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: gizmo_instance_buffer,
                    offset: 0,
                    size: Some(crate::default::INSTANCE_UNIFORM_SIZE),
                },
            }],
        })?;

        // ── Gizmo mesh upload ──
        let translate = build_translate_axis(1.0);
        let (translate_axis_vb, translate_axis_ib, translate_axis_index_count) =
            upload_gizmo_mesh(rhi, "editor.translate_axis", &translate)?;
        let ring = build_rotate_ring(1.0);
        let (rotate_ring_vb, rotate_ring_ib, rotate_ring_index_count) =
            upload_gizmo_mesh(rhi, "editor.rotate_ring", &ring)?;

        Ok(Self {
            mask_pipeline,
            outline_pipeline,
            gizmo_pipeline,
            outline_bgl,
            mask_sampler,
            mask_texture: None,
            mask_size: (0, 0),
            outline_bg: None,
            gizmo_instance_buffer,
            gizmo_instance_bg,
            translate_axis_vb,
            translate_axis_ib,
            translate_axis_index_count,
            rotate_ring_vb,
            rotate_ring_ib,
            rotate_ring_index_count,
        })
    }

    /// Resize the mask attachment alongside `SceneRenderer`'s depth +
    /// scene_color allocations.
    pub fn ensure_mask(&mut self, rhi: &mut dyn Rhi, size: (u32, u32)) {
        if size == self.mask_size && self.mask_texture.is_some() {
            return;
        }
        if let Some(bg) = self.outline_bg.take() {
            rhi.destroy_bind_group(bg);
        }
        if let Some(t) = self.mask_texture.take() {
            rhi.destroy_texture(t);
        }
        if size.0 == 0 || size.1 == 0 {
            self.mask_size = (0, 0);
            return;
        }
        match rhi.create_texture(&TextureDesc {
            label: Some("editor.mask_texture"),
            dimension: TextureDimension::D2 {
                width: size.0,
                height: size.1,
                array_layers: 1,
            },
            format: MASK_FORMAT,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            mip_levels: 1,
            sample_count: 1,
        }) {
            Ok(t) => {
                self.mask_texture = Some(t);
                self.mask_size = size;
                match rhi.create_bind_group(&BindGroupDesc {
                    label: Some("editor.outline_bg"),
                    layout: self.outline_bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Texture(t),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Sampler(self.mask_sampler),
                        },
                    ],
                }) {
                    Ok(bg) => self.outline_bg = Some(bg),
                    Err(e) => log::warn!("editor: outline bg create failed: {e}"),
                }
            }
            Err(e) => {
                log::warn!("editor: mask texture create failed: {e}");
                self.mask_size = (0, 0);
            }
        }
    }

    /// Record mask + outline + gizmo passes into `cl`. The caller
    /// supplies the scene_color attachment (lit-pass output) the
    /// outline + gizmo passes write into, plus the depth attachment
    /// the gizmo pass borrows for its `Always` depth-compare.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi: &mut dyn Rhi,
        default_pipeline: &DefaultPipeline,
        scene_color: TextureHandle,
        depth: TextureHandle,
        viewport: Viewport,
        scissor: Rect,
        selected_draws: &[SelectedDraw],
        gizmo: Option<&GizmoOverlay>,
        cl: &mut CommandList,
    ) {
        // ── Mask pass: render selected meshes' silhouettes.
        if let Some(mask) = self.mask_texture {
            let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
                label: Some("editor.mask"),
                color_attachments: &[Some(ColorAttachment {
                    view: mask,
                    resolve_target: None,
                    load: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                    store: StoreOp::Store,
                })],
                depth_stencil_attachment: None,
            });
            p.set_viewport(viewport);
            p.set_scissor(scissor);
            p.set_pipeline(self.mask_pipeline);
            p.set_bind_group(0, default_pipeline.frame_bg, &[]);
            for d in selected_draws {
                let inst_off = (u64::from(d.instance_slot) * INSTANCE_STRIDE) as u32;
                p.set_bind_group(
                    1,
                    default_pipeline.simple_instance_bg,
                    &[inst_off],
                );
                p.set_vertex_buffer(0, d.vertex_buffer, 0, None);
                p.set_index_buffer(d.index_buffer, 0, None, IndexFormat::Uint32);
                p.draw_indexed(0..d.index_count, 0, 0..1);
            }
        }

        // ── Outline post-process: edge-detect the mask, paint over scene_color.
        if let Some(outline_bg) = self.outline_bg {
            let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
                label: Some("editor.outline_post"),
                color_attachments: &[Some(ColorAttachment {
                    view: scene_color,
                    resolve_target: None,
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                })],
                depth_stencil_attachment: None,
            });
            p.set_viewport(viewport);
            p.set_scissor(scissor);
            p.set_pipeline(self.outline_pipeline);
            p.set_bind_group(0, outline_bg, &[]);
            p.draw(0..6, 0..1);
        }

        // ── Gizmo pass: 2D handles drawn on top with Always depth-compare.
        let Some(gizmo) = gizmo else { return };
        let instances = build_gizmo_instances(gizmo);
        if instances.is_empty() {
            return;
        }
        let mut bytes = vec![0u8; instances.len() * INSTANCE_STRIDE as usize];
        let inst_size = std::mem::size_of::<GizmoInstanceUniform>();
        for (i, inst) in instances.iter().enumerate() {
            let uniform = inst.as_uniform();
            let off = i * INSTANCE_STRIDE as usize;
            bytes[off..off + inst_size].copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        rhi.write_buffer(self.gizmo_instance_buffer, 0, &bytes);

        let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
            label: Some("editor.gizmo_2d"),
            color_attachments: &[Some(ColorAttachment {
                view: scene_color,
                resolve_target: None,
                load: LoadOp::Load,
                store: StoreOp::Store,
            })],
            depth_stencil_attachment: Some(DepthStencilAttachment {
                view: depth,
                depth_load: LoadOp::Load,
                depth_store: StoreOp::Discard,
                stencil_load: LoadOp::Load,
                stencil_store: StoreOp::Discard,
                read_only_depth: true,
                read_only_stencil: true,
            }),
        });
        p.set_viewport(viewport);
        p.set_scissor(scissor);
        p.set_pipeline(self.gizmo_pipeline);
        p.set_bind_group(0, default_pipeline.frame_bg, &[]);
        for (i, inst) in instances.iter().enumerate() {
            let off = (i as u64 * INSTANCE_STRIDE) as u32;
            p.set_bind_group(1, self.gizmo_instance_bg, &[off]);
            let (vb, ib, count) = match inst.kind {
                GizmoMeshKind::TranslateAxis => (
                    self.translate_axis_vb,
                    self.translate_axis_ib,
                    self.translate_axis_index_count,
                ),
                GizmoMeshKind::RotateRing => (
                    self.rotate_ring_vb,
                    self.rotate_ring_ib,
                    self.rotate_ring_index_count,
                ),
            };
            p.set_vertex_buffer(0, vb, 0, None);
            p.set_index_buffer(ib, 0, None, IndexFormat::Uint32);
            p.draw_indexed(0..count, 0, 0..1);
        }
    }
}

// ─── Gizmo instance packing ─────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GizmoInstanceUniform {
    model: Mat4,
    tint: Vec4,
}

#[derive(Clone, Copy, Debug)]
enum GizmoMeshKind {
    TranslateAxis,
    RotateRing,
}

#[derive(Clone, Copy)]
struct GizmoInstance {
    model: Mat4,
    tint: Vec4,
    kind: GizmoMeshKind,
}

impl GizmoInstance {
    fn as_uniform(&self) -> GizmoInstanceUniform {
        GizmoInstanceUniform {
            model: self.model,
            tint: self.tint,
        }
    }
}

fn build_gizmo_instances(g: &GizmoOverlay) -> Vec<GizmoInstance> {
    let s = g.scale_world;
    let origin = Mat4::from_translation(g.origin);
    let active = g.active_axis;
    let tint = |axis: Axis| -> Vec4 {
        let base = match axis {
            Axis::X => Vec4::new(1.0, 0.25, 0.25, 1.0),
            Axis::Y => Vec4::new(0.25, 1.0, 0.25, 1.0),
            Axis::Z => Vec4::new(0.25, 0.4, 1.0, 1.0),
        };
        if active == Some(axis) {
            base + Vec4::new(0.3, 0.3, 0.3, 0.0)
        } else {
            base
        }
    };
    let mk = |axis_rot: Mat4, kind, tint: Vec4| GizmoInstance {
        model: origin * axis_rot * Mat4::from_scale(glam::Vec3::splat(s)),
        tint,
        kind,
    };
    let tx = Mat4::IDENTITY;
    let ty = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let tz = Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2);
    let rx = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let ry = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let rz = Mat4::IDENTITY;
    vec![
        mk(tx, GizmoMeshKind::TranslateAxis, tint(Axis::X)),
        mk(ty, GizmoMeshKind::TranslateAxis, tint(Axis::Y)),
        mk(tz, GizmoMeshKind::TranslateAxis, tint(Axis::Z)),
        mk(rx, GizmoMeshKind::RotateRing, tint(Axis::X)),
        mk(ry, GizmoMeshKind::RotateRing, tint(Axis::Y)),
        mk(rz, GizmoMeshKind::RotateRing, tint(Axis::Z)),
    ]
}

// ─── Gizmo mesh authoring ───────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub neighbor: [f32; 3],
    pub extras: [f32; 2],
}

impl GizmoVertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

pub struct GizmoMesh {
    pub vertices: Vec<GizmoVertex>,
    pub indices: Vec<u32>,
}

const LINE_THICKNESS_PX: f32 = 16.0;
const ARROW_HALF_PX: f32 = 24.0;

pub fn build_translate_axis(length: f32) -> GizmoMesh {
    let line_end = length * 0.85;
    let arrow_tip = length;
    let p0 = [0.0_f32, 0.0, 0.0];
    let p1 = [line_end, 0.0, 0.0];
    let p_tip = [arrow_tip, 0.0, 0.0];
    let line_half = LINE_THICKNESS_PX * 0.5;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(7);
    vertices.push(GizmoVertex {
        position: p0,
        neighbor: p1,
        extras: [-line_half, line_half],
    });
    vertices.push(GizmoVertex {
        position: p0,
        neighbor: p1,
        extras: [line_half, line_half],
    });
    vertices.push(GizmoVertex {
        position: p1,
        neighbor: p0,
        extras: [line_half, line_half],
    });
    vertices.push(GizmoVertex {
        position: p1,
        neighbor: p0,
        extras: [-line_half, line_half],
    });
    vertices.push(GizmoVertex {
        position: p1,
        neighbor: p_tip,
        extras: [-ARROW_HALF_PX, ARROW_HALF_PX],
    });
    vertices.push(GizmoVertex {
        position: p1,
        neighbor: p_tip,
        extras: [ARROW_HALF_PX, ARROW_HALF_PX],
    });
    vertices.push(GizmoVertex {
        position: p_tip,
        neighbor: p1,
        extras: [0.0, 0.0],
    });
    let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3, 4, 5, 6];
    GizmoMesh { vertices, indices }
}

const RING_THICKNESS_PX: f32 = 12.0;

pub fn build_rotate_ring(radius: f32) -> GizmoMesh {
    const SEGMENTS: usize = 64;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(SEGMENTS * 2);
    let mut indices = Vec::<u32>::with_capacity(SEGMENTS * 6);
    let ring_half = RING_THICKNESS_PX * 0.5;
    for i in 0..=SEGMENTS {
        let theta = (i % SEGMENTS) as f32 / (SEGMENTS as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        let p = [c * radius, s * radius, 0.0];
        let t = [-s, c, 0.0];
        let nbr = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
        vertices.push(GizmoVertex {
            position: p,
            neighbor: nbr,
            extras: [-ring_half, ring_half],
        });
        vertices.push(GizmoVertex {
            position: p,
            neighbor: nbr,
            extras: [ring_half, ring_half],
        });
    }
    for i in 0..SEGMENTS {
        let inner_a = (i * 2) as u32;
        let outer_a = (i * 2 + 1) as u32;
        let inner_b = ((i + 1) * 2) as u32;
        let outer_b = ((i + 1) * 2 + 1) as u32;
        indices.extend_from_slice(&[inner_a, outer_a, inner_b, outer_a, outer_b, inner_b]);
    }
    GizmoMesh { vertices, indices }
}

fn upload_gizmo_mesh(
    rhi: &mut dyn Rhi,
    label: &'static str,
    mesh: &GizmoMesh,
) -> RhiResult<(BufferHandle, BufferHandle, u32)> {
    let vb_bytes = bytemuck::cast_slice::<GizmoVertex, u8>(&mesh.vertices);
    let vb = rhi.create_buffer(&BufferDesc {
        label: Some(label),
        size: vb_bytes.len() as u64,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(vb, 0, vb_bytes);
    let ib_bytes = bytemuck::cast_slice::<u32, u8>(&mesh.indices);
    let ib = rhi.create_buffer(&BufferDesc {
        label: Some(label),
        size: ib_bytes.len() as u64,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(ib, 0, ib_bytes);
    Ok((vb, ib, mesh.indices.len() as u32))
}
