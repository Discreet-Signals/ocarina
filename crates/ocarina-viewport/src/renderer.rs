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

use std::collections::HashMap;

use bevy_ecs::entity::Entity;
use floem_renderer::WgpuPassCtx;
use glam::{Mat4, Vec3, Vec4};

use ocarina_core::components::{Material, Sky};
use ocarina_core::AssetPath;
use ocarina_render::editor::{EditorPipelines, PipelineLayout, RenderConfig};
use ocarina_render::shader::ShaderRegistry;
use ocarina_render::{
    build_lights_uniform, build_material_textures_bg, build_sky_pipeline, CameraUniform,
    ColorSpace, DefaultPipeline, InstanceUniform, LightInstance, MaterialTextureKey,
    MaterialUniform, MeshHandle, MeshRegistry, SkyPipeline, SkyUniform, TextureRegistry,
    ENGINE_SPHERE, INSTANCE_STRIDE, MATERIAL_STRIDE, MATERIAL_UNIFORM_SIZE, MAX_INSTANCES,
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
// pixel offsets to NDC; the default and unlit shaders declare it for
// layout compatibility and ignore it.
const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;

/// Camera state ready to upload to the GPU. Computed by the caller --
/// the renderer doesn't know about orbit cameras (editor) vs gameplay
/// cameras (runtime).
#[derive(Clone, Copy)]
pub struct ViewportCamera {
    pub view_proj: Mat4,
    /// Camera world-space position. Plumbed into the camera UBO's
    /// `eye_pos` slot for PBR view-direction reconstruction.
    pub eye_pos: Vec3,
}

impl ViewportCamera {
    /// Camera with `view_proj` set, `eye_pos` zeroed. Hosts that care
    /// about specular shading should fill `eye_pos` from the inverse
    /// view matrix translation column before passing to `render`.
    #[must_use]
    pub fn standard(view_proj: Mat4) -> Self {
        Self {
            view_proj,
            eye_pos: Vec3::ZERO,
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
    /// Per-entity tint read by the simple-pipeline shaders (mask /
    /// gizmo / unlit). The default scene shader reads
    /// `material.albedo` instead -- the renderer mirrors albedo into
    /// `tint` before upload, plus an optional editor-selection
    /// brightness boost on top.
    pub tint: Vec4,
    /// Asset path of the entity's mesh. `None` means the entity has
    /// no `Mesh` component (e.g. a camera) and isn't drawn in the
    /// lit pass; the editor may still surface a small selection
    /// marker for it.
    pub mesh: Option<ocarina_core::AssetPath>,
    /// PBR material parameters + texture asset paths. Defaults to
    /// `Material::default()` when the entity has no `Material`
    /// component.
    pub material: Material,
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

    /// Default scene pipeline + camera/instance UBOs + bind groups,
    /// owned by the shared rendering core. Used for normal lit draws.
    default_pipeline: DefaultPipeline,
    /// 3-stop gradient sky pipeline. Runs BEFORE the lit pass when
    /// the world has a `Sky` component, replacing the lit pass's
    /// solid clear with a per-pixel world-space gradient.
    sky_pipeline: SkyPipeline,
    /// Editor-only alt pipelines (wireframe + unlit). Selected via
    /// [`RenderConfig`] passed to `render()`. Compiled in here
    /// because viewport always builds with ocarina-render's
    /// `editor` feature on.
    editor_pipelines: EditorPipelines,
    /// Shared mesh registry. Built-in `engine/cube`, `engine/sphere`
    /// pre-loaded; user OBJs cached on first reference. Same handle
    /// type the runtime uses, same loading path.
    meshes: MeshRegistry,
    /// Texture registry for material albedo / normal / metallic-
    /// roughness / emission slots. Pre-populated with 1×1 white /
    /// black / flat-normal defaults so missing textures still bind
    /// cleanly.
    textures: TextureRegistry,
    /// Shader registry: drives the per-shader pipeline cache the
    /// `default_pipeline` looks up via [`DefaultPipeline::pipeline_for`].
    shader_registry: ShaderRegistry,
    /// Per-material texture bind groups, keyed by the texture handle
    /// quadruple. Two entities with the same material textures share
    /// one bind group; lookup is the hot path inside the entity loop.
    material_bg_cache: HashMap<MaterialTextureKey, BindGroupHandle>,

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

        // Default scene pipeline + camera/instance UBOs (shared with
        // the runtime). After this call we never construct
        // scene-side resources locally again.
        let default_pipeline = ocarina_render::default::build_default_pipeline(
            &mut rhi,
            target_format,
            DEPTH_FORMAT,
        )?;

        // Sky pipeline: a fullscreen quad that runs first whenever
        // the world has a `Sky` component. Owns its own UBO + bind
        // group; doesn't share the scene pipeline's frame_bgl.
        let sky_pipeline = build_sky_pipeline(&mut rhi, target_format, DEPTH_FORMAT)?;

        // Editor-only alt pipelines (wireframe, unlit). Always built
        // because the viewport always enables ocarina-render's
        // `editor` feature.
        let editor_pipelines = EditorPipelines::build(
            &mut rhi,
            &default_pipeline,
            target_format,
            DEPTH_FORMAT,
        )?;

        // Mesh registry. Built-ins (engine/cube, engine/sphere)
        // pre-loaded; user-asset paths cached on first reference.
        let mut meshes = MeshRegistry::new();
        meshes.register_builtins(&mut rhi)?;

        // Texture registry: 1×1 white / black / flat-normal defaults
        // pre-uploaded so a missing material texture binds cleanly.
        let textures = TextureRegistry::new(&mut rhi)?;

        // Shader registry: pre-populates `engine/shaders/default`,
        // resolves project-relative `.wgsl` files on demand.
        let shader_registry = ShaderRegistry::new();

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
            default_pipeline.frame_bgl,
            default_pipeline.simple_instance_bgl,
            target_format,
        )?;
        rhi.destroy_shader(gizmo_vs);
        rhi.destroy_shader(gizmo_fs);

        // Mask pipeline. Same camera/instance bind groups as the
        // default scene pipeline; only the fragment output differs
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
            default_pipeline.frame_bgl,
            default_pipeline.simple_instance_bgl,
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
            default_pipeline,
            sky_pipeline,
            editor_pipelines,
            meshes,
            textures,
            shader_registry,
            material_bg_cache: HashMap::new(),
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

    /// Set the project root used by the mesh / texture / shader
    /// registries to resolve project-relative asset paths. Call this
    /// when a project is loaded / switched. Engine assets
    /// (`engine/cube`, `engine/textures/white`, etc.) work regardless.
    pub fn set_project_root(&mut self, root: std::path::PathBuf) {
        self.meshes.set_project_root(root.clone());
        self.textures.set_project_root(root.clone());
        self.shader_registry.set_project_root(root);
    }

    /// Resolve `material`'s four texture slots, look up (or build)
    /// the cached group-2 bind group, and return its handle. The
    /// renderer calls this once per entity per frame; the cache hit
    /// rate is high because most scenes have a handful of materials.
    fn ensure_material_bg(&mut self, material: &Material) -> Option<BindGroupHandle> {
        let albedo = self.textures.ensure_or_white(
            &mut self.rhi,
            &material.albedo_texture,
            ColorSpace::Srgb,
        );
        let normal = self
            .textures
            .ensure_or_flat_normal(&mut self.rhi, &material.normal_texture);
        let metallic_roughness = self.textures.ensure_or_white(
            &mut self.rhi,
            &material.metallic_roughness_texture,
            ColorSpace::Linear,
        );
        let emission = self.textures.ensure_or_white(
            &mut self.rhi,
            &material.emission_texture,
            ColorSpace::Srgb,
        );
        let key = MaterialTextureKey {
            albedo,
            normal,
            metallic_roughness,
            emission,
        };
        if let Some(h) = self.material_bg_cache.get(&key) {
            return Some(*h);
        }
        match build_material_textures_bg(
            &mut self.rhi,
            self.default_pipeline.material_textures_bgl,
            key,
        ) {
            Ok(handle) => {
                self.material_bg_cache.insert(key, handle);
                Some(handle)
            }
            Err(e) => {
                log::warn!("viewport: material bind-group create failed: {e}");
                None
            }
        }
    }

    /// Bind group for the "no material" fallback (selection marker,
    /// future placeholder geometry). Resolves to the cached default
    /// white-texture set.
    fn ensure_default_material_bg(&mut self) -> Option<BindGroupHandle> {
        self.ensure_material_bg(&Material::default())
    }

    /// Render `entities` into `ctx.view` at `ctx.rect` (in framebuffer
    /// pixels). Uses `LoadOp::Clear` for color (the viewport runs as
    /// a pre-paint pass under Floem UI) and `LoadOp::Clear(1.0)` for
    /// depth.
    ///
    /// `scene_shader` selects which scene shader the lit pass binds.
    /// Pass `engine/shaders/default` for the built-in PBR shader, or
    /// any project-relative `.wgsl` path (resolved through the
    /// internal [`ShaderRegistry`]) to swap in a user shader. Misses
    /// fall back to the default with a warning.
    ///
    /// `gizmo` is the editor's translate / rotate overlay drawn on
    /// top of the scene when something is selected; pass `None` to
    /// skip (e.g. play mode, no selection).
    pub fn render(
        &mut self,
        entities: &[EntityInstance],
        lights: &[LightInstance],
        sky: Option<Sky>,
        scene_shader: &AssetPath,
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
            eye_pos: Vec4::new(camera.eye_pos.x, camera.eye_pos.y, camera.eye_pos.z, 0.0),
            // Panel rect in physical pixels -- what NDC maps to after
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
        self.rhi.write_buffer(
            self.default_pipeline.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_uniform),
        );

        // Pack scene lights into the lights UBO. Empty list → a
        // single fully-zeroed buffer; the shader's `count = 0` loop
        // contributes only the ambient floor.
        let lights_uniform = build_lights_uniform(lights);
        self.rhi.write_buffer(
            self.default_pipeline.lights_buffer,
            0,
            bytemuck::bytes_of(&lights_uniform),
        );

        // Upload the sky uniform if a Sky component is present in
        // the world. The sky pass runs only when this happens; the
        // lit pass below switches its LoadOp accordingly.
        if let Some(sky) = sky {
            let sky_uniform = SkyUniform::build(&sky, camera.view_proj, camera.eye_pos);
            self.rhi.write_buffer(
                self.sky_pipeline.buffer,
                0,
                bytemuck::bytes_of(&sky_uniform),
            );
        }

        let selected = match mode {
            RenderMode::Edit { selected_entity } => selected_entity,
            RenderMode::Play => None,
        };

        // Resolve each entity's mesh handle and build the per-frame
        // draw list + instance buffer + material buffer. Slot
        // layout:
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
            /// Per-material bind group for the scene pipeline's
            /// group 2. `None` means "use default white textures"
            /// (selection marker + gizmo geometry).
            material_bg: Option<BindGroupHandle>,
        }
        let mut entity_draws: Vec<EntityDraw> = Vec::with_capacity(count);

        // Default white material for the selection marker + gizmo
        // entities (resolved once, cached in `material_bg_cache`).
        let default_material_bg = self.ensure_default_material_bg();

        let total_slots = count + if gizmo.is_some() { GIZMO_SLOTS } else { 0 };
        if total_slots > 0 {
            let mut instance_bytes: Vec<u8> =
                vec![0; total_slots * INSTANCE_STRIDE as usize];
            let mut material_bytes: Vec<u8> =
                vec![0; total_slots * MATERIAL_STRIDE as usize];
            for (i, inst) in entities.iter().take(count).enumerate() {
                let highlight = selected == Some(inst.entity);

                // Resolve the entity's mesh (if any); for selected
                // meshless entities, fall back to a small sphere
                // marker so they're still pickable / visible.
                let (model, mesh_handle, is_marker) = match inst
                    .mesh
                    .as_ref()
                    .and_then(|p| self.meshes.ensure(&mut self.rhi, p))
                {
                    Some(handle) => (inst.world_xform, Some(handle), false),
                    None if highlight => {
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
                        (marker, h, true)
                    }
                    None => continue,
                };

                let Some(handle) = mesh_handle else {
                    continue;
                };

                // Mirror albedo into the instance UBO's `tint` so the
                // simple-pipeline shaders (mask / gizmo / unlit) and
                // the scene shader agree on the entity's display
                // colour. Editor selection bumps both with a
                // brightness boost.
                let base_albedo = inst.material.albedo;
                let display_albedo = if highlight {
                    base_albedo * 1.4 + Vec4::splat(0.15)
                } else {
                    base_albedo
                };

                let instance_uniform = InstanceUniform {
                    model,
                    tint: display_albedo,
                };
                let material_uniform = if is_marker {
                    // Selection marker is a small "you are here"
                    // sphere; doesn't deserve the entity's PBR
                    // material. Render flat-coloured.
                    let mut m = MaterialUniform::neutral();
                    m.albedo = display_albedo;
                    m
                } else {
                    let mut m = MaterialUniform::from_component(&inst.material);
                    if highlight {
                        m.albedo = display_albedo;
                    }
                    m
                };

                let inst_off = i * INSTANCE_STRIDE as usize;
                instance_bytes[inst_off..inst_off + INSTANCE_UNIFORM_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&instance_uniform));
                let mat_off = i * MATERIAL_STRIDE as usize;
                material_bytes[mat_off..mat_off + MATERIAL_UNIFORM_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&material_uniform));

                // Per-material textures bind group. Marker uses
                // defaults; real entities resolve their material
                // texture set through the cache.
                let material_bg = if is_marker {
                    default_material_bg
                } else {
                    self.ensure_material_bg(&inst.material)
                };

                entity_draws.push(EntityDraw {
                    slot: i as u32,
                    mesh: handle,
                    entity: inst.entity,
                    material_bg,
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
                    let off = (count + slot) * INSTANCE_STRIDE as usize;
                    instance_bytes[off..off + INSTANCE_UNIFORM_SIZE as usize]
                        .copy_from_slice(bytemuck::bytes_of(&uniform));
                    // Gizmo draws go through the simple pipeline so
                    // material UBO is never read; leaving the slot
                    // zeroed is fine.
                };
                for (i, axis) in Axis::ALL.iter().enumerate() {
                    write(i, translate_axis_model(g.origin, *axis, g.scale_world), *axis);
                }
                for (i, axis) in Axis::ALL.iter().enumerate() {
                    write(3 + i, rotate_ring_model(g.origin, *axis, g.scale_world), *axis);
                }
            }
            self.rhi
                .write_buffer(self.default_pipeline.instance_buffer, 0, &instance_bytes);
            self.rhi
                .write_buffer(self.default_pipeline.material_buffer, 0, &material_bytes);
        }

        // ─── Now resolve all the wgpu objects we need for the
        // render pass. All of these are immutable borrows from the
        // RHI's pools — fine to hold simultaneously. ──────────────

        // Pipeline selection -- done up front so any pipeline-cache
        // build (Stage 9: alternative scene shaders selected via
        // `RenderSettings`) lands BEFORE we acquire any immutable
        // RHI borrows for the render pass below. RenderMode::Play
        // uses the [`RenderSettings`]-driven scene shader (full PBR
        // layout); edit mode honours the editor toggles -- wireframe
        // still takes the scene path, "lighting off" drops to the
        // simple unlit pipeline.
        let (main_pipeline_handle, main_layout) = match mode {
            RenderMode::Play => {
                let h = self.default_pipeline.pipeline_for(
                    &mut self.rhi,
                    &mut self.shader_registry,
                    scene_shader,
                );
                (h, PipelineLayout::Scene)
            }
            RenderMode::Edit { .. } => {
                let _ = scene_shader; // editor mode ignores it for now
                config.select_pipeline(&self.default_pipeline, &self.editor_pipelines)
            }
        };

        let Some(depth_view) = self.rhi.texture_view(depth) else {
            return;
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
        let Some(frame_bg) = self.rhi.bind_group(self.default_pipeline.frame_bg) else {
            return;
        };
        let Some(scene_instance_bg) =
            self.rhi.bind_group(self.default_pipeline.instance_bg)
        else {
            return;
        };
        let Some(simple_instance_bg) =
            self.rhi.bind_group(self.default_pipeline.simple_instance_bg)
        else {
            return;
        };
        // Sky pipeline + bind group. Always resolved (we created
        // them in `from_existing`); the actual sky pass below only
        // runs when `sky` is `Some`.
        let sky_runs = sky.is_some();
        let sky_pipeline = if sky_runs {
            self.rhi.graphics_pipeline(self.sky_pipeline.pipeline)
        } else {
            None
        };
        let sky_bg = if sky_runs {
            self.rhi.bind_group(self.sky_pipeline.bg)
        } else {
            None
        };
        let do_sky = sky_runs && sky_pipeline.is_some() && sky_bg.is_some();
        // Resolve every entity-draw's wgpu vb/ib + per-material
        // bind group upfront so the render-pass loop can borrow them
        // all simultaneously (RHI returns immutable borrows out of
        // the same pool).
        struct Resolved<'a> {
            slot: u32,
            vb: &'a wgpu::Buffer,
            ib: &'a wgpu::Buffer,
            index_count: u32,
            material_bg: Option<&'a wgpu::BindGroup>,
        }
        let mut resolved_draws: Vec<Resolved<'_>> = Vec::with_capacity(entity_draws.len());
        for d in &entity_draws {
            let Some(vb) = self.rhi.buffer(d.mesh.vertex_buffer) else {
                continue;
            };
            let Some(ib) = self.rhi.buffer(d.mesh.index_buffer) else {
                continue;
            };
            let material_bg = d.material_bg.and_then(|bg| self.rhi.bind_group(bg));
            resolved_draws.push(Resolved {
                slot: d.slot,
                vb,
                ib,
                index_count: d.mesh.index_count,
                material_bg,
            });
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
            } => resolved_draws
                .iter()
                .find(|r| {
                    entity_draws
                        .iter()
                        .find(|d| d.slot == r.slot)
                        .map_or(false, |d| d.entity == sel)
                })
                .map(|r| (r.slot, r.vb, r.ib, r.index_count)),
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

        // ─── Sky pass: fullscreen gradient quad. Runs FIRST when
        // the world has a `Sky` component; replaces the lit pass's
        // solid clear with a per-pixel world-space gradient. The
        // lit pass below detects this via `do_sky` and switches its
        // LoadOp from Clear to Load so it paints on top. ─────────
        if do_sky {
            let sky_pipeline_obj = sky_pipeline.unwrap();
            let sky_bg_obj = sky_bg.unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina viewport sky pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
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
            pass.set_pipeline(sky_pipeline_obj);
            pass.set_bind_group(0, sky_bg_obj, &[]);
            pass.draw(0..6, 0..1);
        }

        // ─── Lit pass: scene → swapchain color. When the sky pass
        // ran first, this pass loads the existing colour (so geometry
        // overdraws the gradient where it covers); otherwise it
        // clears to a fixed editor-grey background. ─────────────
        {
            let lit_color_load = if do_sky {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.07,
                    g: 0.09,
                    b: 0.12,
                    a: 1.0,
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina viewport lit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: lit_color_load,
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
            pass.set_bind_group(0, frame_bg, &[]);
            // One draw per resolved entity. Branch by pipeline
            // layout: scene (default + wireframe) binds the
            // instance + material UBOs at group 1 and the per-
            // material textures at group 2. Simple (unlit) binds
            // only the instance UBO.
            match main_layout {
                PipelineLayout::Scene => {
                    for r in &resolved_draws {
                        let inst_off = (u64::from(r.slot) * INSTANCE_STRIDE) as u32;
                        let mat_off = (u64::from(r.slot) * MATERIAL_STRIDE) as u32;
                        pass.set_bind_group(
                            1,
                            scene_instance_bg,
                            &[inst_off, mat_off],
                        );
                        if let Some(bg) = r.material_bg {
                            pass.set_bind_group(2, bg, &[]);
                        }
                        pass.set_vertex_buffer(0, r.vb.slice(..));
                        pass.set_index_buffer(r.ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..r.index_count, 0, 0..1);
                    }
                }
                PipelineLayout::Simple => {
                    for r in &resolved_draws {
                        let inst_off = (u64::from(r.slot) * INSTANCE_STRIDE) as u32;
                        pass.set_bind_group(1, simple_instance_bg, &[inst_off]);
                        pass.set_vertex_buffer(0, r.vb.slice(..));
                        pass.set_index_buffer(r.ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..r.index_count, 0, 0..1);
                    }
                }
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
                pass.set_bind_group(0, frame_bg, &[]);
                let dyn_offset = (u64::from(outline_slot) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, simple_instance_bg, &[dyn_offset]);
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
            pass.set_bind_group(0, frame_bg, &[]);

            // Translate axes: 3 instances of the line+arrow mesh.
            pass.set_vertex_buffer(0, translate_axis_vb.slice(..));
            pass.set_index_buffer(translate_axis_ib.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..3u32 {
                let dyn_offset = ((count as u64 + u64::from(i)) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, simple_instance_bg, &[dyn_offset]);
                pass.draw_indexed(0..self.translate_axis_index_count, 0, 0..1);
            }

            // Rotate rings: 3 instances of the ring mesh.
            pass.set_vertex_buffer(0, rotate_ring_vb.slice(..));
            pass.set_index_buffer(rotate_ring_ib.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..3u32 {
                let dyn_offset =
                    ((count as u64 + 3 + u64::from(i)) * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, simple_instance_bg, &[dyn_offset]);
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
