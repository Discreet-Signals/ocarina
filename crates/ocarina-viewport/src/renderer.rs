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
use floem_renderer::WgpuPassCtx;
use glam::{Mat4, Vec3, Vec4};

use ocarina_core::AssetPath;
use ocarina_render::editor::{EditorPipelines, RenderConfig};
use ocarina_render::{
    CameraUniform, InstanceUniform, LitPipeline, MeshHandle, MeshRegistry, ENGINE_SPHERE,
    INSTANCE_STRIDE, MAX_INSTANCES,
};
use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindGroupLayoutHandle, BindingResource, BindingType, BlendState, BufferDesc, BufferHandle,
    BufferUsage, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentStage,
    GraphicsPipelineDesc, MemoryLocation, MultisampleState, PipelineDesc, PipelineHandle,
    PipelineKind, PrimitiveState, Rhi, RhiError, RhiResult, SamplerBindingType, SamplerDesc,
    SamplerHandle, ShaderBytecode, ShaderDesc, ShaderHandle, ShaderStage, ShaderStageFlags,
    TextureDesc, TextureDimension, TextureFormat, TextureHandle, TextureSampleType, TextureUsage,
    TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStage,
    VertexStepMode,
};
use ocarina_rhi_wgpu::WgpuRhi;

use crate::mesh;

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;
/// Single-channel format for the selection mask. R8Unorm gives 1 byte
/// per pixel and reads back through a Float texture binding, so the
/// outline post-process can sample it directly.
const MASK_FORMAT: TextureFormat = TextureFormat::R8Unorm;
const MASK_WGSL: &str = include_str!("../shaders/mask.wgsl");
const OUTLINE_POST_WGSL: &str = include_str!("../shaders/outline_post.wgsl");
const GIZMO_2D_WGSL: &str = include_str!("../shaders/gizmo_2d.wgsl");

// CameraUniform / InstanceUniform / INSTANCE_STRIDE / MAX_INSTANCES
// come from `ocarina_render`. The camera UBO carries a
// `viewport_size: Vec4` field used by the 2D gizmo shader to convert
// pixel offsets to NDC; the lit and unlit shaders declare it for
// layout compatibility and ignore it.
const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;

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
#[derive(Clone)]
pub struct EntityInstance {
    pub entity: Entity,
    pub world_xform: Mat4,
    /// Multiplied with the camera's `base_color` and the `selected`
    /// boost (in edit mode). Pass `Vec4::new(0.78, 0.82, 0.90, 1.0)`
    /// for the default "neutral cool grey" Ocarina aesthetic.
    pub tint: Vec4,
    /// Asset path of the entity's mesh. `None` means the entity has
    /// no `Mesh` component (e.g. a camera) and isn't drawn in the
    /// lit pass; the editor may still surface a small selection
    /// marker for it.
    pub mesh: Option<ocarina_core::AssetPath>,
}

/// Editor overlay: translate **and** rotate gizmo handles to draw at
/// the selection's origin. Both sets render every frame so the user
/// can grab either without first switching modes. The handles are
/// 2D screen-space lines/arrows + circles (see `gizmo_2d.wgsl`), so
/// `scale_world` only controls the *layout* extent (how far the
/// arrows reach into the world); thickness is constant pixels.
#[derive(Clone, Copy, Debug)]
pub struct GizmoOverlay {
    pub origin: Vec3,
    pub scale_world: f32,
    /// Axis to highlight (currently dragged); the other handles
    /// render at their normal axis tint.
    pub active_axis: Option<crate::gizmo::Axis>,
}

/// All persistent GPU state for the viewport renderer.
pub struct ViewportRenderer {
    rhi: WgpuRhi,

    /// Lit pipeline + camera/instance UBOs + bind groups, owned by
    /// the shared rendering core. Used for normal lit draws.
    lit: LitPipeline,
    /// Editor-only alt pipelines (wireframe + unlit). Selected via
    /// [`RenderConfig`] passed to `render()`. Compiled in here
    /// because viewport always builds with ocarina-render's
    /// `editor` feature on.
    editor_pipelines: EditorPipelines,
    /// Shared mesh registry. Built-in `engine/cube`, `engine/sphere`
    /// pre-loaded; user OBJs cached on first reference. Same handle
    /// type the runtime uses, same loading path.
    meshes: MeshRegistry,

    /// Mask pipeline. Re-renders the selected mesh through the
    /// `mask.wgsl` shader into a single-channel R8Unorm texture
    /// (1.0 inside the silhouette, 0.0 outside). Reads camera /
    /// instance UBOs from `lit`'s bind groups.
    mask_pipeline: PipelineHandle,
    /// Post-process pipeline. Fullscreen quad, samples the mask +
    /// 4 neighbors, emits outline color where the mask transitions.
    outline_pipeline: PipelineHandle,
    /// Gizmo pipeline. Same lit shader as the main pipeline but
    /// `depth_compare = Always`, `depth_write_enabled = false` so
    /// translate / rotate handles render on top regardless of 3D
    /// occlusion (you should be able to grab a handle even when it
    /// pokes through a mesh).
    gizmo_pipeline: PipelineHandle,
    /// Translate-axis line+arrow mesh in the *gizmo* vertex format
    /// (`mesh::GizmoVertex`). Different vertex layout from the lit
    /// pipeline; needs its own pipeline.
    translate_axis_vb: BufferHandle,
    translate_axis_ib: BufferHandle,
    translate_axis_index_count: u32,
    /// Rotate-axis ring mesh in the gizmo vertex format.
    rotate_ring_vb: BufferHandle,
    rotate_ring_ib: BufferHandle,
    rotate_ring_index_count: u32,

    /// Bind group layout for the outline post-process: binding 0 is
    /// the mask texture, binding 1 is the sampler.
    outline_bgl: BindGroupLayoutHandle,
    /// Sampler used to read the mask texture in the post-process.
    /// Non-filtering: the mask is binary, linear filtering would just
    /// blur the silhouette without helping edge detection.
    mask_sampler: SamplerHandle,
    /// Bind group binding `mask_texture` + `mask_sampler` for the
    /// outline post-process. Recreated whenever `mask_texture` is
    /// recreated (on resize).
    outline_bg: Option<BindGroupHandle>,

    target_format: TextureFormat,

    depth: Option<TextureHandle>,
    depth_size: (u32, u32),

    /// Selection mask texture, sized to the swapchain. Recreated on
    /// resize alongside the depth target. Lives only when there's
    /// something to outline, but easier to keep persistent.
    mask_texture: Option<TextureHandle>,
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

        // Lit pipeline + camera/instance UBOs (shared with the
        // runtime). After this call we never construct lit-side
        // resources locally again.
        let lit = ocarina_render::lit::build_lit_pipeline(
            &mut rhi,
            target_format,
            DEPTH_FORMAT,
        )?;

        // Editor-only alt pipelines (wireframe, unlit). Always built
        // because the viewport always enables ocarina-render's
        // `editor` feature.
        let editor_pipelines =
            EditorPipelines::build(&mut rhi, &lit, target_format, DEPTH_FORMAT)?;

        // Mesh registry. Built-ins (engine/cube, engine/sphere)
        // pre-loaded; user-asset paths cached on first reference.
        let mut meshes = MeshRegistry::new();
        meshes.register_builtins(&mut rhi)?;

        // 2D gizmo geometry: translate-axis line+arrow, rotate-axis
        // ring. Both are authored in a *gizmo* vertex format (see
        // `mesh::GizmoVertex`) that the gizmo shader expands into
        // constant-pixel-thickness lines on screen.
        let translate_mesh = mesh::build_translate_axis(1.0);
        let (translate_axis_vb, translate_axis_ib, translate_axis_index_count) =
            upload_gizmo_mesh(&mut rhi, "translate_axis", &translate_mesh)?;
        let ring_mesh = mesh::build_rotate_ring(1.0);
        let (rotate_ring_vb, rotate_ring_ib, rotate_ring_index_count) =
            upload_gizmo_mesh(&mut rhi, "rotate_ring", &ring_mesh)?;

        // Gizmo overlay pipeline. Different vertex format
        // (`mesh::GizmoVertex`) and shader (`gizmo_2d.wgsl`); handles
        // are screen-space-expanded lines, not 3D geometry. Reuses
        // the lit pipeline's camera + instance bind group layouts.
        let gizmo_vs = rhi.create_shader(&ShaderDesc {
            label: Some("gizmo_2d.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(GIZMO_2D_WGSL),
        })?;
        let gizmo_fs = rhi.create_shader(&ShaderDesc {
            label: Some("gizmo_2d.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(GIZMO_2D_WGSL),
        })?;
        let gizmo_pipeline = build_gizmo_pipeline(
            &mut rhi,
            gizmo_vs,
            gizmo_fs,
            lit.camera_bgl,
            lit.instance_bgl,
            target_format,
        )?;
        rhi.destroy_shader(gizmo_vs);
        rhi.destroy_shader(gizmo_fs);

        // Mask pipeline. Same camera/instance bind groups as the
        // lit pipeline; only the fragment output differs
        // (single-channel 1.0 into the R8Unorm mask texture).
        let mask_vs = rhi.create_shader(&ShaderDesc {
            label: Some("mask.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(MASK_WGSL),
        })?;
        let mask_fs = rhi.create_shader(&ShaderDesc {
            label: Some("mask.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(MASK_WGSL),
        })?;
        let mask_pipeline = build_mask_pipeline(
            &mut rhi,
            mask_vs,
            mask_fs,
            lit.camera_bgl,
            lit.instance_bgl,
        )?;
        rhi.destroy_shader(mask_vs);
        rhi.destroy_shader(mask_fs);

        // Outline post-process. Bind group 0 is mask texture +
        // sampler (no camera/instance UBOs needed; the shader reads
        // from textureDimensions for kernel sizing).
        let outline_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("outline.bgl"),
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
            label: Some("outline.mask_sampler"),
            mag_filter: ocarina_rhi::FilterMode::Nearest,
            min_filter: ocarina_rhi::FilterMode::Nearest,
            mipmap_filter: ocarina_rhi::FilterMode::Nearest,
            ..SamplerDesc::default()
        })?;
        let outline_vs = rhi.create_shader(&ShaderDesc {
            label: Some("outline_post.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_POST_WGSL),
        })?;
        let outline_fs = rhi.create_shader(&ShaderDesc {
            label: Some("outline_post.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(OUTLINE_POST_WGSL),
        })?;
        let outline_pipeline =
            build_outline_post_pipeline(&mut rhi, outline_vs, outline_fs, outline_bgl, target_format)?;
        rhi.destroy_shader(outline_vs);
        rhi.destroy_shader(outline_fs);

        Ok(Self {
            rhi,
            lit,
            editor_pipelines,
            meshes,
            mask_pipeline,
            outline_pipeline,
            gizmo_pipeline,
            translate_axis_vb,
            translate_axis_ib,
            translate_axis_index_count,
            rotate_ring_vb,
            rotate_ring_ib,
            rotate_ring_index_count,
            outline_bgl,
            mask_sampler,
            outline_bg: None,
            target_format,
            depth: None,
            depth_size: (0, 0),
            mask_texture: None,
        })
    }

    /// Set the project root used by the mesh registry to resolve
    /// user-mesh asset paths. Call this when a project is loaded /
    /// switched. Engine meshes (`engine/cube` etc.) work regardless.
    pub fn set_project_root(&mut self, root: std::path::PathBuf) {
        self.meshes.set_project_root(root);
    }

    /// Render `entities` into `ctx.view` at `ctx.rect` (in framebuffer
    /// pixels). Uses `LoadOp::Load` for color so a UI compositor's
    /// underlying paint is preserved; `LoadOp::Clear(1.0)` for depth.
    ///
    /// `gizmo` is the editor's translate / rotate overlay drawn on
    /// top of the scene when something is selected; pass `None` to
    /// skip (e.g. play mode, no selection).
    pub fn render(
        &mut self,
        entities: &[EntityInstance],
        ctx: &WgpuPassCtx<'_>,
        camera: ViewportCamera,
        mode: RenderMode,
        gizmo: Option<GizmoOverlay>,
        config: RenderConfig,
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
            if let Some(m) = self.mask_texture.take() {
                self.rhi.destroy_texture(m);
            }
            if let Some(bg) = self.outline_bg.take() {
                self.rhi.destroy_bind_group(bg);
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
            // Mask must match the depth/color extent — same wgpu rule
            // about all attachments in a pass sharing extent. Plus
            // post-process samples it with kernel offsets in screen
            // texels, so the screen size has to be the texture size.
            match self.rhi.create_texture(&TextureDesc {
                label: Some("viewport.mask"),
                dimension: TextureDimension::D2 {
                    width: depth_w,
                    height: depth_h,
                    array_layers: 1,
                },
                format: MASK_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                mip_levels: 1,
                sample_count: 1,
            }) {
                Ok(m) => {
                    self.mask_texture = Some(m);
                    // Build the outline bind group against the new
                    // mask texture + (persistent) sampler.
                    match self.rhi.create_bind_group(&BindGroupDesc {
                        label: Some("outline.bg"),
                        layout: self.outline_bgl,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Texture(m),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Sampler(self.mask_sampler),
                            },
                        ],
                    }) {
                        Ok(bg) => self.outline_bg = Some(bg),
                        Err(e) => {
                            log::warn!("viewport: outline bind-group create failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    log::warn!("viewport: mask texture create failed: {e}");
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
            // Panel rect in physical pixels — what NDC maps to after
            // `set_viewport(...)`. Using `depth_w/h` (swapchain size)
            // here gave under-sized lines whenever the panel was
            // smaller than the swapchain, since the shader's
            // `pixels * 2 / size` conversion was using the wrong
            // denominator.
            viewport_size: Vec4::new(
                ctx.rect.width() as f32,
                ctx.rect.height() as f32,
                0.0,
                0.0,
            ),
        };
        self.rhi
            .write_buffer(self.lit.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        let selected = match mode {
            RenderMode::Edit { selected_entity } => selected_entity,
            RenderMode::Play => None,
        };

        // Resolve each entity's mesh handle and build the per-frame
        // draw list + instance buffer. Slot layout:
        //   [0..entity_cap)      = entity slots (one per input entity).
        //   [entity_cap..+6)     = gizmo slots (6 axes / rings).
        // An entity slot is occupied if (a) the entity has a Mesh
        // we successfully resolved, or (b) the entity is selected
        // and meshless -- we draw a small "selection marker" sphere
        // at its world position so the dot pickup is visible.
        const GIZMO_SLOTS: usize = 6;
        let entity_cap = (MAX_INSTANCES as usize).saturating_sub(GIZMO_SLOTS);
        let count = entities.len().min(entity_cap);

        struct EntityDraw {
            slot: u32,
            mesh: MeshHandle,
            entity: Entity,
        }
        let mut entity_draws: Vec<EntityDraw> = Vec::with_capacity(count);

        let total_slots = count + if gizmo.is_some() { GIZMO_SLOTS } else { 0 };
        if total_slots > 0 {
            let mut instance_bytes: Vec<u8> =
                vec![0; total_slots * INSTANCE_STRIDE as usize];
            for (i, inst) in entities.iter().take(count).enumerate() {
                let highlight = selected == Some(inst.entity);
                let final_tint = if highlight {
                    inst.tint * 1.4 + Vec4::splat(0.15)
                } else {
                    inst.tint
                };

                // Resolve the entity's mesh (if any); for selected
                // meshless entities, fall back to a small sphere
                // marker so they're still pickable / visible.
                let (model, mesh_handle) = match inst
                    .mesh
                    .as_ref()
                    .and_then(|p| self.meshes.ensure(&mut self.rhi, p))
                {
                    Some(handle) => (inst.world_xform, Some(handle)),
                    None if highlight => {
                        // Selection marker: small sphere at world pos.
                        let pos = inst.world_xform.col(3).truncate();
                        const MARKER_SIZE: f32 = 0.25;
                        let marker = Mat4::from_scale_rotation_translation(
                            Vec3::splat(MARKER_SIZE),
                            glam::Quat::IDENTITY,
                            pos,
                        );
                        let h = self
                            .meshes
                            .ensure(&mut self.rhi, &AssetPath::new(ENGINE_SPHERE));
                        (marker, h)
                    }
                    None => continue,
                };

                let Some(handle) = mesh_handle else {
                    continue;
                };

                let offset = i * INSTANCE_STRIDE as usize;
                let uniform = InstanceUniform {
                    model,
                    tint: final_tint,
                };
                instance_bytes[offset..offset + INSTANCE_UNIFORM_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&uniform));
                entity_draws.push(EntityDraw {
                    slot: i as u32,
                    mesh: handle,
                    entity: inst.entity,
                });
            }
            if let Some(g) = gizmo {
                use crate::gizmo::Axis;
                const HIGHLIGHT_TINT: Vec4 = Vec4::new(1.0, 0.85, 0.20, 1.0);

                let mut write = |slot: usize, model: Mat4, axis: Axis| {
                    let base = Vec4::from_array(axis.tint());
                    let tint = if g.active_axis == Some(axis) {
                        HIGHLIGHT_TINT
                    } else {
                        base
                    };
                    let uniform = InstanceUniform { model, tint };
                    let offset = (count + slot) * INSTANCE_STRIDE as usize;
                    instance_bytes[offset..offset + INSTANCE_UNIFORM_SIZE as usize]
                        .copy_from_slice(bytemuck::bytes_of(&uniform));
                };
                for (i, axis) in Axis::ALL.iter().enumerate() {
                    write(i, translate_axis_model(g.origin, *axis, g.scale_world), *axis);
                }
                for (i, axis) in Axis::ALL.iter().enumerate() {
                    write(3 + i, rotate_ring_model(g.origin, *axis, g.scale_world), *axis);
                }
            }
            self.rhi.write_buffer(self.lit.instance_buffer, 0, &instance_bytes);
        }

        // ─── Now resolve all the wgpu objects we need for the
        // render pass. All of these are immutable borrows from the
        // RHI's pools — fine to hold simultaneously. ──────────────

        let Some(depth_view) = self.rhi.texture_view(depth) else {
            return;
        };
        // RenderMode::Play forces lit (this is the actual game look).
        // Edit mode honours the editor toggles.
        let main_pipeline_handle = match mode {
            RenderMode::Play => self.lit.pipeline,
            RenderMode::Edit { .. } => {
                config.select_pipeline(&self.lit, &self.editor_pipelines)
            }
        };
        let Some(pipeline) = self.rhi.graphics_pipeline(main_pipeline_handle) else {
            return;
        };
        let Some(mask_pipeline) = self.rhi.graphics_pipeline(self.mask_pipeline) else {
            return;
        };
        let Some(outline_pipeline) =
            self.rhi.graphics_pipeline(self.outline_pipeline)
        else {
            return;
        };
        let Some(camera_bg) = self.rhi.bind_group(self.lit.camera_bg) else {
            return;
        };
        let Some(instance_bg) = self.rhi.bind_group(self.lit.instance_bg) else {
            return;
        };
        // Resolve every entity-draw's wgpu vb/ib upfront so the
        // render-pass loop can borrow them all simultaneously
        // (RHI returns immutable borrows out of the same pool).
        let mut resolved_draws: Vec<(u32, &wgpu::Buffer, &wgpu::Buffer, u32)> =
            Vec::with_capacity(entity_draws.len());
        for d in &entity_draws {
            let Some(vb) = self.rhi.buffer(d.mesh.vertex_buffer) else {
                continue;
            };
            let Some(ib) = self.rhi.buffer(d.mesh.index_buffer) else {
                continue;
            };
            resolved_draws.push((d.slot, vb, ib, d.mesh.index_count));
        }
        let Some(translate_axis_vb) = self.rhi.buffer(self.translate_axis_vb) else {
            return;
        };
        let Some(translate_axis_ib) = self.rhi.buffer(self.translate_axis_ib) else {
            return;
        };
        let Some(rotate_ring_vb) = self.rhi.buffer(self.rotate_ring_vb) else {
            return;
        };
        let Some(rotate_ring_ib) = self.rhi.buffer(self.rotate_ring_ib) else {
            return;
        };
        let Some(gizmo_pipeline) = self.rhi.graphics_pipeline(self.gizmo_pipeline) else {
            return;
        };
        let mask_view = self
            .mask_texture
            .and_then(|m| self.rhi.texture_view(m));
        let outline_bg = self.outline_bg.and_then(|bg| self.rhi.bind_group(bg));

        // Selected entity's draw record (slot + mesh handle's
        // wgpu buffers + index count). The mask pass uses it to
        // re-render the silhouette into the R8 mask. `None` if the
        // selection has no resolvable mesh AND no marker (which
        // shouldn't happen because meshless-selected gets a marker).
        let outline_draw: Option<(u32, &wgpu::Buffer, &wgpu::Buffer, u32)> = match mode {
            RenderMode::Edit {
                selected_entity: Some(sel),
            } => entity_draws
                .iter()
                .find(|d| d.entity == sel)
                .and_then(|d| {
                    let vb = self.rhi.buffer(d.mesh.vertex_buffer)?;
                    let ib = self.rhi.buffer(d.mesh.index_buffer)?;
                    Some((d.slot, vb, ib, d.mesh.index_count))
                }),
            _ => None,
        };
        let do_outline =
            outline_draw.is_some() && mask_view.is_some() && outline_bg.is_some();

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
        let viewport = (
            ctx.rect.x0 as f32,
            ctx.rect.y0 as f32,
            ctx.rect.width() as f32,
            ctx.rect.height() as f32,
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ocarina viewport encoder"),
            });

        // ─── Lit pass: scene → swapchain color (LoadOp::Load
        // preserves Floem's UI underneath) + our depth target. ────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina viewport lit pass"),
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
            pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
            pass.set_scissor_rect(sx, sy, sw, sh);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera_bg, &[]);
            // One draw per resolved entity. Bind the entity's mesh
            // before each draw -- entities can have different meshes
            // (built-in cube, built-in sphere, user OBJ, ...) so we
            // can't share buffers across the loop.
            for (slot, vb, ib, index_count) in &resolved_draws {
                let dyn_offset = (*slot as u64 * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
        }

        // ─── Outline = mask pass + post-process pass. Skipped when
        // there's no selection or the resize fall-back hasn't yet
        // produced a fresh mask texture / bind group. ────────────
        if do_outline {
            let (outline_slot, outline_vb, outline_ib, outline_index_count) =
                outline_draw.unwrap();
            let mask_view = mask_view.unwrap();
            let outline_bg = outline_bg.unwrap();

            // Mask pass: render the selected mesh as solid 1.0 into
            // the R8 mask texture. Cleared to 0.0 so unselected
            // pixels read as "outside silhouette" in the post.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ocarina viewport mask pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: mask_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
                pass.set_scissor_rect(sx, sy, sw, sh);
                pass.set_pipeline(mask_pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                let dyn_offset = (u64::from(outline_slot) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.set_vertex_buffer(0, outline_vb.slice(..));
                pass.set_index_buffer(outline_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..outline_index_count, 0, 0..1);
            }

            // Outline post-process: fullscreen quad over the
            // swapchain (LoadOp::Load), reads the mask + 4 neighbors,
            // alpha-blends the outline color where the mask boundary
            // falls inside the kernel.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ocarina viewport outline post"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ctx.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
                pass.set_scissor_rect(sx, sy, sw, sh);
                pass.set_pipeline(outline_pipeline);
                pass.set_bind_group(0, outline_bg, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // ─── Gizmo pass: render translate-axis line+arrows AND
        // rotate-axis rings on top of the scene. Single render pass
        // because both share the same shader / pipeline state — only
        // the vertex buffer + index buffer change between the two
        // sets of three draws.
        //
        // Instance buffer layout written above:
        //   slots [count..count+3) — translate X / Y / Z
        //   slots [count+3..count+6) — rotate    X / Y / Z
        if gizmo.is_some() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina viewport gizmo pass"),
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
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
            pass.set_scissor_rect(sx, sy, sw, sh);
            pass.set_pipeline(gizmo_pipeline);
            pass.set_bind_group(0, camera_bg, &[]);

            // Translate axes: 3 instances of the line+arrow mesh.
            pass.set_vertex_buffer(0, translate_axis_vb.slice(..));
            pass.set_index_buffer(translate_axis_ib.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..3u32 {
                let dyn_offset = ((count as u64 + u64::from(i)) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.draw_indexed(0..self.translate_axis_index_count, 0, 0..1);
            }

            // Rotate rings: 3 instances of the ring mesh.
            pass.set_vertex_buffer(0, rotate_ring_vb.slice(..));
            pass.set_index_buffer(rotate_ring_ib.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..3u32 {
                let dyn_offset =
                    ((count as u64 + 3 + u64::from(i)) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, instance_bg, &[dyn_offset]);
                pass.draw_indexed(0..self.rotate_ring_index_count, 0, 0..1);
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

fn upload_gizmo_mesh(
    rhi: &mut WgpuRhi,
    label: &str,
    mesh: &mesh::GizmoMesh,
) -> RhiResult<(BufferHandle, BufferHandle, u32)> {
    let vb_size = (mesh.vertices.len() * mesh::GizmoVertex::SIZE) as u64;
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

/// Build the model matrix for a rotate-axis ring at `origin`. The
/// base ring (`mesh::build_rotate_ring`) lies in the XY plane; this
/// rotates it into the plane perpendicular to `axis`:
///
/// - Z axis ring → keep XY plane (no rotation).
/// - X axis ring → rotate 90° around Y so it sits in the YZ plane.
/// - Y axis ring → rotate 90° around X so it sits in the XZ plane.
fn rotate_ring_model(origin: Vec3, axis: crate::gizmo::Axis, scale_world: f32) -> Mat4 {
    use crate::gizmo::Axis;
    let rotation = match axis {
        Axis::X => Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Axis::Y => Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Axis::Z => Mat4::IDENTITY,
    };
    Mat4::from_translation(origin) * rotation * Mat4::from_scale(Vec3::splat(scale_world))
}

/// Build the model matrix for a translate-axis line+arrow at
/// `origin`. The base mesh (`mesh::build_translate_axis`) points
/// along `+X`; this rotates it into the +Y / +Z direction for the
/// other two axes.
fn translate_axis_model(origin: Vec3, axis: crate::gizmo::Axis, scale_world: f32) -> Mat4 {
    use crate::gizmo::Axis;
    let rotation = match axis {
        Axis::X => Mat4::IDENTITY,
        // Rotate +X to +Y around Z.
        Axis::Y => Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
        // Rotate +X to +Z around Y.
        Axis::Z => Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2),
    };
    Mat4::from_translation(origin) * rotation * Mat4::from_scale(Vec3::splat(scale_world))
}

/// Gizmo overlay pipeline using `gizmo_2d.wgsl`. Vertex layout is
/// `mesh::GizmoVertex` (anchor position, neighbor position, extras),
/// not the lit cube's interleaved format. Depth is `Always` with
/// no write so handles render on top of the scene.
fn build_gizmo_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
    color_format: TextureFormat,
) -> RhiResult<PipelineHandle> {
    let attributes = [
        // position
        VertexAttribute { format: VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
        // neighbor
        VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
        // extras (xy)
        VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
    ];
    let vertex_buffers = [VertexBufferLayout {
        array_stride: mesh::GizmoVertex::SIZE as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &attributes,
    }];
    let color_targets = [Some(ColorTargetState {
        format: color_format,
        blend: Some(BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("gizmo_2d"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage { shader: vs, buffers: &vertex_buffers },
            fragment: Some(FragmentStage { shader: fs, targets: &color_targets }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
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
        bind_group_layouts: &[camera_bgl, instance_bgl],
        push_constant_ranges: &[],
    })
}

/// Mask pipeline. Same vertex layout + camera/instance bind groups
/// as the lit pipeline, but the fragment writes a single channel into
/// an R8Unorm target. No depth, no blend — every pixel the selected
/// mesh covers is just stamped with 1.0.
fn build_mask_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
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
        array_stride: ocarina_render::Vertex::SIZE as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &attributes,
    }];
    let color_targets = [Some(ColorTargetState {
        format: MASK_FORMAT,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("mask"),
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
            depth_stencil: None,
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[camera_bgl, instance_bgl],
        push_constant_ranges: &[],
    })
}

/// Outline post-process pipeline. Fullscreen quad generated from
/// `@builtin(vertex_index)` so no vertex buffer is bound. Bind group
/// 0 holds the mask texture + sampler; the fragment samples those and
/// alpha-blends the outline color into the swapchain.
fn build_outline_post_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    outline_bgl: BindGroupLayoutHandle,
    color_format: TextureFormat,
) -> RhiResult<PipelineHandle> {
    let color_targets = [Some(ColorTargetState {
        format: color_format,
        blend: Some(BlendState::ALPHA_BLENDING),
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("outline_post"),
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
            depth_stencil: None,
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[outline_bgl],
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
