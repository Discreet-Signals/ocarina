//! Unified scene renderer.
//!
//! Both hosts (the editor's viewport panel and the runtime's
//! windowed mode) feed the same `SceneRenderer` per-frame and let
//! it record a [`CommandList`]. The host only owns the RHI handle
//! to the swapchain target and submits the resulting command list;
//! everything from sky → lit → optional editor overlays → post-
//! process chain → final blit is centralised here so the runtime
//! and editor stay in lockstep without duplicating draw code.
//!
//! The renderer is RHI-only -- no `wgpu` types appear in the public
//! API. Editor-only extras (mask + outline + gizmo + wireframe /
//! unlit alt pipelines) are split into a separate [`record_editor_overlays`]
//! step the viewport calls between scene and compositor; the runtime
//! skips it entirely.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use glam::{Mat4, Vec3, Vec4};

use ocarina_core::components::{Material, Sky};
use ocarina_core::AssetPath;

use ocarina_rhi::{
    AsyncPipelineId, BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, BufferDesc, BufferHandle,
    BufferUsage, ColorAttachment, ColorTargetState, ColorWrites, CommandList, FragmentStage,
    GraphicsPipelineDesc, IndexFormat, LoadOp, MemoryLocation, MultisampleState, PipelineDesc,
    PipelineHandle, PipelineKind, PrimitiveState, Rect, Rhi, RhiError, RhiResult,
    SamplerBindingType, SamplerDesc, SamplerHandle, ShaderBytecode, ShaderDesc, ShaderStage,
    ShaderStageFlags, StoreOp, TextureDesc, TextureDimension, TextureFormat, TextureHandle,
    TextureSampleType, TextureUsage, TextureViewDimension, VertexStage, Viewport,
};

use crate::default::{
    build_lights_uniform, build_material_textures_bg, CameraUniform, DefaultPipeline,
    InstanceUniform, LightInstance, MaterialTextureKey, MaterialUniform, INSTANCE_STRIDE,
    MATERIAL_STRIDE, MATERIAL_UNIFORM_SIZE, MAX_INSTANCES,
};
use crate::mesh::{MeshHandle, MeshRegistry};
use crate::reflect::{
    self, BindingKind as ReflectBindingKind, DefaultValue, ScalarKind, UniformField,
};
use crate::shader::ShaderRegistry;
use crate::sky::{build_sky_pipeline, SkyPipeline, SkyUniform};
use crate::texture::{ColorSpace, TextureRegistry};

#[cfg(feature = "editor")]
use crate::editor::EditorPipelines;
#[cfg(feature = "editor")]
use crate::editor_overlays::{EditorOverlays, GizmoOverlay, SelectedDraw};

/// Depth attachment format for all hosts.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

const COMPOSITOR_WGSL: &str = include_str!("../shaders/compositor.wgsl");

#[repr(C)]
#[derive(Copy, Clone)]
struct PostUniforms {
    resolution: [f32; 2],
    time: f32,
    delta: f32,
}
unsafe impl bytemuck::Zeroable for PostUniforms {}
unsafe impl bytemuck::Pod for PostUniforms {}
const POST_UNIFORM_SIZE: u64 = std::mem::size_of::<PostUniforms>() as u64;

// ─── Public types ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct ViewportCamera {
    pub view_proj: Mat4,
    pub eye_pos: Vec3,
}

impl ViewportCamera {
    /// Pack into the shared [`CameraUniform`] including viewport size.
    pub fn to_uniform(self, viewport_size: (f32, f32)) -> CameraUniform {
        CameraUniform {
            view_proj: self.view_proj,
            eye_pos: Vec4::new(self.eye_pos.x, self.eye_pos.y, self.eye_pos.z, 0.0),
            viewport_size: Vec4::new(viewport_size.0, viewport_size.1, 0.0, 0.0),
        }
    }
}

#[derive(Clone)]
pub struct EntityInstance {
    pub entity: Entity,
    pub world_xform: Mat4,
    pub mesh: Option<AssetPath>,
    pub material: Material,
}

/// Per-frame inputs for [`SceneRenderer::record_scene`] and
/// [`SceneRenderer::record_compositor`].
pub struct SceneArgs<'a> {
    /// The host's swapchain target. Provided by:
    /// - runtime: `rhi.acquire_next_image(swapchain)`.
    /// - editor: `WgpuRhi::register_external_view(floem_view, ..)`.
    pub target: TextureHandle,
    /// Full extent of `target` in pixels.
    pub target_size: (u32, u32),
    /// Panel rectangle in framebuffer pixels (x, y, w, h). Drives
    /// `set_viewport` + `set_scissor` so drawing is clipped to the
    /// host's panel area within the target. For the runtime this is
    /// always (0, 0, target_w, target_h).
    pub rect: (f32, f32, f32, f32),
    pub camera: ViewportCamera,
    pub entities: &'a [EntityInstance],
    pub lights: &'a [LightInstance],
    pub sky: Option<Sky>,
    pub scene_shader: &'a AssetPath,
    pub post_process: &'a [AssetPath],
}

// ─── SceneRenderer ──────────────────────────────────────────────

pub struct SceneRenderer {
    target_format: TextureFormat,

    default_pipeline: DefaultPipeline,
    sky_pipeline: SkyPipeline,
    meshes: MeshRegistry,
    textures: TextureRegistry,
    shader_registry: ShaderRegistry,
    material_bg_cache: HashMap<MaterialTextureKey, BindGroupHandle>,

    /// Lazy depth attachment, sized to the target.
    depth: Option<TextureHandle>,
    depth_size: (u32, u32),

    /// Lazy offscreen RT used when the camera has any post-process
    /// chain bound. When the chain is empty we render directly to
    /// the swapchain target and skip allocation.
    scene_color: Option<TextureHandle>,
    scene_color_size: (u32, u32),

    // Compositor (post-process) state.
    compositor_bgl: ocarina_rhi::BindGroupLayoutHandle,
    compositor_uniform_bgl: ocarina_rhi::BindGroupLayoutHandle,
    compositor_sampler: SamplerHandle,
    compositor_uniform_buffer: BufferHandle,
    compositor_uniform_bg: BindGroupHandle,
    compositor_default_pipeline: PipelineHandle,
    compositor_user_pipelines: HashMap<AssetPath, UserCompositor>,
    compositor_build_state: HashMap<AssetPath, CompositorBuildState>,
    chain_textures: HashMap<AssetPath, [TextureHandle; 2]>,
    chain_textures_size: (u32, u32),
    /// Per-frame transient bind groups built for the chain pass.
    /// Held here for one frame so the borrowed views referenced by
    /// the bind group survive the pass; drained at the start of
    /// each `record_compositor` call.
    compositor_deferred_bgs: Vec<BindGroupHandle>,
    compositor_frame_index: u64,
    compositor_start: std::time::Instant,
    compositor_last_frame: std::time::Instant,

    /// Editor-only alt pipelines (wireframe, unlit). Always built
    /// when the `editor` feature is on.
    #[cfg(feature = "editor")]
    pub editor_pipelines: EditorPipelines,
    /// Editor-only overlay state (mask + outline + gizmo).
    #[cfg(feature = "editor")]
    pub editor_overlays: EditorOverlays,
}

impl SceneRenderer {
    /// Construct a renderer against an existing RHI. The runtime
    /// passes its `WgpuRhi`; the editor passes its `WgpuRhi`
    /// (built via `WgpuRhi::from_existing` on Floem's device).
    pub fn new(rhi: &mut dyn Rhi, target_format: TextureFormat) -> RhiResult<Self> {
        let default_pipeline = crate::default::build_default_pipeline(
            rhi,
            target_format,
            DEPTH_FORMAT,
        )?;
        let sky_pipeline = build_sky_pipeline(rhi, target_format, DEPTH_FORMAT)?;

        #[cfg(feature = "editor")]
        let editor_pipelines = EditorPipelines::build(
            rhi,
            &default_pipeline,
            target_format,
            DEPTH_FORMAT,
        )?;
        #[cfg(feature = "editor")]
        let editor_overlays = EditorOverlays::build(
            rhi,
            &default_pipeline,
            target_format,
            DEPTH_FORMAT,
        )?;

        let mut meshes = MeshRegistry::new();
        meshes.register_builtins(rhi)?;
        let textures = TextureRegistry::new(rhi)?;
        let shader_registry = ShaderRegistry::new();

        // Compositor BGL: input_color (texture), prev_color (texture),
        // sampler. Bindings 3+ free for user textures.
        let compositor_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("compositor.scene_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })?;
        let compositor_uniform_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("compositor.uniform_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(POST_UNIFORM_SIZE),
                },
                count: None,
            }],
        })?;
        let compositor_sampler = rhi.create_sampler(&SamplerDesc {
            label: Some("compositor.sampler"),
            mag_filter: ocarina_rhi::FilterMode::Linear,
            min_filter: ocarina_rhi::FilterMode::Linear,
            mipmap_filter: ocarina_rhi::FilterMode::Nearest,
            ..SamplerDesc::default()
        })?;
        let compositor_uniform_buffer = rhi.create_buffer(&BufferDesc {
            label: Some("compositor.uniforms"),
            size: POST_UNIFORM_SIZE,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let compositor_uniform_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("compositor.uniform_bg"),
            layout: compositor_uniform_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: compositor_uniform_buffer,
                    offset: 0,
                    size: Some(POST_UNIFORM_SIZE),
                },
            }],
        })?;
        let compositor_vs = rhi.create_shader(&ShaderDesc {
            label: Some("compositor.wgsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(COMPOSITOR_WGSL),
        })?;
        let compositor_fs = rhi.create_shader(&ShaderDesc {
            label: Some("compositor.wgsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(COMPOSITOR_WGSL),
        })?;
        let compositor_default_pipeline = build_compositor_pipeline(
            rhi,
            compositor_vs,
            compositor_fs,
            compositor_bgl,
            compositor_uniform_bgl,
            None,
            target_format,
            "compositor.default",
        )?;
        rhi.destroy_shader(compositor_vs);
        rhi.destroy_shader(compositor_fs);

        let now = std::time::Instant::now();
        Ok(Self {
            target_format,
            default_pipeline,
            sky_pipeline,
            meshes,
            textures,
            shader_registry,
            material_bg_cache: HashMap::new(),
            depth: None,
            depth_size: (0, 0),
            scene_color: None,
            scene_color_size: (0, 0),
            compositor_bgl,
            compositor_uniform_bgl,
            compositor_sampler,
            compositor_uniform_buffer,
            compositor_uniform_bg,
            compositor_default_pipeline,
            compositor_user_pipelines: HashMap::new(),
            compositor_build_state: HashMap::new(),
            chain_textures: HashMap::new(),
            chain_textures_size: (0, 0),
            compositor_deferred_bgs: Vec::new(),
            compositor_frame_index: 0,
            compositor_start: now,
            compositor_last_frame: now,
            #[cfg(feature = "editor")]
            editor_pipelines,
            #[cfg(feature = "editor")]
            editor_overlays,
        })
    }

    /// Record the editor's mask + outline + gizmo passes into `cl`.
    /// Must be called between [`Self::record_scene`] and
    /// [`Self::record_compositor`]. `paint_target` is the texture
    /// the outline + gizmo passes write into -- get it from
    /// [`Self::paint_target`] so it matches whatever `record_scene`
    /// chose (scene_color when post-process is bound, args.target
    /// otherwise).
    ///
    /// Cfg-gated to the `editor` feature: the runtime never sees
    /// this method.
    #[cfg(feature = "editor")]
    pub fn record_editor_overlays(
        &mut self,
        rhi: &mut dyn Rhi,
        paint_target: TextureHandle,
        target_size: (u32, u32),
        rect: (f32, f32, f32, f32),
        selected_draws: &[SelectedDraw],
        gizmo: Option<&GizmoOverlay>,
        cl: &mut CommandList,
    ) {
        self.editor_overlays.ensure_mask(rhi, target_size);
        let Some(depth) = self.depth else {
            return;
        };
        let viewport = Viewport {
            x: rect.0,
            y: rect.1,
            width: rect.2,
            height: rect.3,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = Rect {
            x: rect.0.max(0.0) as u32,
            y: rect.1.max(0.0) as u32,
            width: rect.2.max(0.0) as u32,
            height: rect.3.max(0.0) as u32,
        };
        self.editor_overlays.record(
            rhi,
            &self.default_pipeline,
            paint_target,
            depth,
            viewport,
            scissor,
            selected_draws,
            gizmo,
            cl,
        );
    }

    /// Set the project root used by the asset registries to resolve
    /// relative paths.
    pub fn set_project_root(&mut self, root: Arc<std::path::PathBuf>) {
        self.meshes.set_project_root(root.clone());
        self.textures.set_project_root(root.clone());
        self.shader_registry.set_project_root(root);
    }

    /// Default pipeline accessor (used by editor pipelines that need
    /// shared BGLs for their own pipeline construction).
    pub fn default_pipeline(&self) -> &DefaultPipeline {
        &self.default_pipeline
    }

    /// Target color format the renderer was constructed for. Hosts
    /// use this to decide whether to rebuild on swapchain-format
    /// changes.
    pub fn target_format(&self) -> TextureFormat {
        self.target_format
    }

    /// Return the texture `record_scene` last painted into for the
    /// given args. When the post-process chain is non-empty that's
    /// `scene_color`; otherwise it's `args.target`. Editor hosts
    /// pass this as `paint_target` to
    /// [`Self::record_editor_overlays`] so the outline + gizmo
    /// passes land on the same surface.
    pub fn paint_target(&self, args: &SceneArgs<'_>) -> TextureHandle {
        let needs_post = args.post_process.iter().any(|p| !p.is_empty());
        if needs_post {
            self.scene_color.unwrap_or(args.target)
        } else {
            args.target
        }
    }

    /// Mesh registry accessor.
    pub fn meshes_mut(&mut self) -> &mut MeshRegistry {
        &mut self.meshes
    }

    /// Texture registry accessor.
    pub fn textures_mut(&mut self) -> &mut TextureRegistry {
        &mut self.textures
    }

    /// Shader registry accessor.
    pub fn shader_registry_mut(&mut self) -> &mut ShaderRegistry {
        &mut self.shader_registry
    }

    /// Resolve `material`'s four texture slots, look up (or build)
    /// the cached group-2 bind group, and return its handle.
    fn ensure_material_bg(
        &mut self,
        rhi: &mut dyn Rhi,
        material: &Material,
    ) -> Option<BindGroupHandle> {
        let albedo =
            self.textures
                .ensure_or_white(rhi, &material.albedo_texture, ColorSpace::Srgb);
        let normal = self.textures.ensure_or_flat_normal(rhi, &material.normal_texture);
        let metallic_roughness = self.textures.ensure_or_white(
            rhi,
            &material.metallic_roughness_texture,
            ColorSpace::Linear,
        );
        let emission =
            self.textures
                .ensure_or_white(rhi, &material.emission_texture, ColorSpace::Srgb);
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
            rhi,
            self.default_pipeline.material_textures_bgl,
            key,
        ) {
            Ok(handle) => {
                self.material_bg_cache.insert(key, handle);
                Some(handle)
            }
            Err(e) => {
                log::warn!("scene: material bind-group create failed: {e}");
                None
            }
        }
    }

    /// Allocate / resize per-frame attachments based on whether a
    /// post-process chain is bound. Called at the top of every
    /// frame's recording sequence.
    fn ensure_targets(
        &mut self,
        rhi: &mut dyn Rhi,
        target_size: (u32, u32),
        needs_post: bool,
        chain: &[&AssetPath],
    ) {
        let (w, h) = target_size;
        if w == 0 || h == 0 {
            return;
        }

        // Depth always exists.
        if self.depth_size != target_size {
            if let Some(t) = self.depth.take() {
                rhi.destroy_texture(t);
            }
            match rhi.create_texture(&TextureDesc {
                label: Some("scene.depth"),
                dimension: TextureDimension::D2 {
                    width: w,
                    height: h,
                    array_layers: 1,
                },
                format: DEPTH_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT,
                mip_levels: 1,
                sample_count: 1,
            }) {
                Ok(d) => {
                    self.depth = Some(d);
                    self.depth_size = target_size;
                }
                Err(e) => {
                    log::warn!("scene: depth create failed: {e}");
                    return;
                }
            }
        }

        // scene_color + chain textures only when a chain is in play.
        if needs_post {
            let make_attachment = |label: &'static str, target: &mut dyn Rhi| {
                target.create_texture(&TextureDesc {
                    label: Some(label),
                    dimension: TextureDimension::D2 {
                        width: w,
                        height: h,
                        array_layers: 1,
                    },
                    format: self.target_format,
                    usage: TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::TEXTURE_BINDING,
                    mip_levels: 1,
                    sample_count: 1,
                })
            };

            if self.scene_color_size != target_size
                || self.chain_textures_size != target_size
            {
                if let Some(t) = self.scene_color.take() {
                    rhi.destroy_texture(t);
                }
                for (_, [a, b]) in self.chain_textures.drain() {
                    rhi.destroy_texture(a);
                    rhi.destroy_texture(b);
                }
                self.scene_color_size = (0, 0);
                self.chain_textures_size = (0, 0);
            }

            if self.scene_color.is_none() {
                match make_attachment("scene.scene_color", rhi) {
                    Ok(t) => {
                        self.scene_color = Some(t);
                        self.scene_color_size = target_size;
                    }
                    Err(e) => {
                        log::warn!("scene: scene_color create failed: {e}");
                        return;
                    }
                }
            }

            // Drop chain entries for shaders no longer in the chain.
            let chain_paths: Vec<AssetPath> =
                chain.iter().map(|p| (*p).clone()).collect();
            self.chain_textures.retain(|p, [a, b]| {
                if chain_paths.iter().any(|c| c == p) {
                    true
                } else {
                    rhi.destroy_texture(*a);
                    rhi.destroy_texture(*b);
                    false
                }
            });

            for path in chain {
                if !self.chain_textures.contains_key(*path) {
                    let label_a: &'static str = Box::leak(
                        format!("scene.chain[{}].ping_a", path.as_str()).into_boxed_str(),
                    );
                    let label_b: &'static str = Box::leak(
                        format!("scene.chain[{}].ping_b", path.as_str()).into_boxed_str(),
                    );
                    let a = match make_attachment(label_a, rhi) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!(
                                "scene: chain ping create failed `{}`: {e}",
                                path.as_str()
                            );
                            return;
                        }
                    };
                    let b = match make_attachment(label_b, rhi) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!(
                                "scene: chain pong create failed `{}`: {e}",
                                path.as_str()
                            );
                            rhi.destroy_texture(a);
                            return;
                        }
                    };
                    self.chain_textures.insert((*path).clone(), [a, b]);
                }
            }
            self.chain_textures_size = target_size;
        } else {
            // No chain. Free if previously allocated.
            if let Some(t) = self.scene_color.take() {
                rhi.destroy_texture(t);
            }
            for (_, [a, b]) in self.chain_textures.drain() {
                rhi.destroy_texture(a);
                rhi.destroy_texture(b);
            }
            self.scene_color_size = (0, 0);
            self.chain_textures_size = (0, 0);
        }
    }

    /// Record sky + lit passes into the supplied [`CommandList`].
    /// Output goes to the offscreen `scene_color` if a post chain
    /// is bound, otherwise straight to `args.target`.
    pub fn record_scene(
        &mut self,
        rhi: &mut dyn Rhi,
        args: &SceneArgs<'_>,
        cl: &mut CommandList,
    ) {
        let chain: Vec<&AssetPath> =
            args.post_process.iter().filter(|p| !p.is_empty()).collect();
        let needs_post = !chain.is_empty();

        // Drain last frame's transient compositor bind groups.
        for bg in self.compositor_deferred_bgs.drain(..) {
            rhi.destroy_bind_group(bg);
        }

        self.ensure_targets(rhi, args.target_size, needs_post, &chain);

        let Some(depth) = self.depth else {
            return;
        };

        // Upload camera + lights UBOs.
        let cam_uniform = args
            .camera
            .to_uniform((args.rect.2, args.rect.3));
        rhi.write_buffer(
            self.default_pipeline.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_uniform),
        );
        let lights_uniform = build_lights_uniform(args.lights);
        rhi.write_buffer(
            self.default_pipeline.lights_buffer,
            0,
            bytemuck::bytes_of(&lights_uniform),
        );

        // Sky uniform (only when a Sky component is present).
        if let Some(sky) = args.sky {
            let sky_uniform = SkyUniform::build(&sky, args.camera.view_proj, args.camera.eye_pos);
            rhi.write_buffer(
                self.sky_pipeline.buffer,
                0,
                bytemuck::bytes_of(&sky_uniform),
            );
        }

        // Resolve scene pipeline (hot-reload aware).
        let scene_pipeline = self.default_pipeline.pipeline_for(
            rhi,
            &mut self.shader_registry,
            args.scene_shader,
        );

        // Pack instances + materials.
        let count = args.entities.len().min(MAX_INSTANCES as usize);
        struct DrawCall {
            mesh: MeshHandle,
            slot: u32,
            material_bg: Option<BindGroupHandle>,
        }
        let mut draws: Vec<DrawCall> = Vec::with_capacity(count);
        let mut instance_bytes = vec![0u8; count * INSTANCE_STRIDE as usize];
        let mut material_bytes = vec![0u8; count * MATERIAL_STRIDE as usize];
        for (slot, inst) in args.entities.iter().take(count).enumerate() {
            let Some(mesh_path) = inst.mesh.as_ref() else {
                continue;
            };
            let Some(mesh_handle) = self.meshes.ensure(rhi, mesh_path) else {
                continue;
            };
            let inst_uniform = InstanceUniform {
                model: inst.world_xform,
                tint: inst.material.albedo,
            };
            let mat_uniform = MaterialUniform::from_component(&inst.material);
            let off_i = slot * INSTANCE_STRIDE as usize;
            instance_bytes[off_i..off_i + std::mem::size_of::<InstanceUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&inst_uniform));
            let off_m = slot * MATERIAL_STRIDE as usize;
            material_bytes[off_m..off_m + MATERIAL_UNIFORM_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&mat_uniform));

            let material_bg = self.ensure_material_bg(rhi, &inst.material);
            draws.push(DrawCall {
                mesh: mesh_handle,
                slot: slot as u32,
                material_bg,
            });
        }
        if !instance_bytes.is_empty() {
            rhi.write_buffer(self.default_pipeline.instance_buffer, 0, &instance_bytes);
            rhi.write_buffer(self.default_pipeline.material_buffer, 0, &material_bytes);
        }

        // Pick the color attachment: offscreen if chain in play,
        // else direct-to-target.
        let color_target = self
            .scene_color
            .filter(|_| needs_post)
            .unwrap_or(args.target);

        let viewport = Viewport {
            x: args.rect.0,
            y: args.rect.1,
            width: args.rect.2,
            height: args.rect.3,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = Rect {
            x: args.rect.0.max(0.0) as u32,
            y: args.rect.1.max(0.0) as u32,
            width: args.rect.2.max(0.0) as u32,
            height: args.rect.3.max(0.0) as u32,
        };

        // Sky pass (when present) writes the cleared backdrop; lit
        // pass loads it. Otherwise lit pass clears the color.
        if args.sky.is_some() {
            let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
                label: Some("scene.sky"),
                color_attachments: &[Some(ColorAttachment {
                    view: color_target,
                    resolve_target: None,
                    load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                    store: StoreOp::Store,
                })],
                depth_stencil_attachment: None,
            });
            p.set_viewport(viewport);
            p.set_scissor(scissor);
            p.set_pipeline(self.sky_pipeline.pipeline);
            p.set_bind_group(0, self.sky_pipeline.bg, &[]);
            // Sky is a single oversized triangle (sky.wgsl). The
            // rasteriser clips overhang to the viewport; no shared
            // diagonal means fewer helper-quad fragment invocations.
            p.draw(0..3, 0..1);
        }

        // Lit pass.
        let lit_load = if args.sky.is_some() {
            LoadOp::Load
        } else {
            LoadOp::Clear([0.05, 0.06, 0.07, 1.0])
        };
        let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
            label: Some("scene.lit"),
            color_attachments: &[Some(ColorAttachment {
                view: color_target,
                resolve_target: None,
                load: lit_load,
                store: StoreOp::Store,
            })],
            depth_stencil_attachment: Some(ocarina_rhi::DepthStencilAttachment {
                view: depth,
                depth_load: LoadOp::Clear(1.0),
                depth_store: StoreOp::Store,
                stencil_load: LoadOp::Load,
                stencil_store: StoreOp::Discard,
                read_only_depth: false,
                read_only_stencil: true,
            }),
        });
        p.set_viewport(viewport);
        p.set_scissor(scissor);
        p.set_pipeline(scene_pipeline);
        p.set_bind_group(0, self.default_pipeline.frame_bg, &[]);

        for d in &draws {
            let inst_off = d.slot as u64 * INSTANCE_STRIDE;
            let mat_off = d.slot as u64 * MATERIAL_STRIDE;
            p.set_bind_group(
                1,
                self.default_pipeline.instance_bg,
                &[inst_off as u32, mat_off as u32],
            );
            if let Some(bg) = d.material_bg {
                p.set_bind_group(2, bg, &[]);
            }
            p.set_vertex_buffer(0, d.mesh.vertex_buffer, 0, None);
            p.set_index_buffer(d.mesh.index_buffer, 0, None, IndexFormat::Uint32);
            p.draw_indexed(0..d.mesh.index_count, 0, 0..1);
        }
    }

    /// Record the post-process chain (if any) followed by the final
    /// blit to `args.target`. Skipped when `args.post_process` is
    /// empty / contains only blank paths -- in that case
    /// [`record_scene`] already wrote straight to the target and
    /// the caller is done.
    pub fn record_compositor(
        &mut self,
        rhi: &mut dyn Rhi,
        args: &SceneArgs<'_>,
        cl: &mut CommandList,
    ) {
        let chain: Vec<&AssetPath> =
            args.post_process.iter().filter(|p| !p.is_empty()).collect();
        if chain.is_empty() {
            return;
        }
        let Some(scene_color) = self.scene_color else {
            return;
        };

        // Update PostUniforms.
        let now = std::time::Instant::now();
        let delta = now
            .duration_since(self.compositor_last_frame)
            .as_secs_f32();
        self.compositor_last_frame = now;
        let post = PostUniforms {
            resolution: [args.rect.2, args.rect.3],
            time: now.duration_since(self.compositor_start).as_secs_f32(),
            delta,
        };
        rhi.write_buffer(
            self.compositor_uniform_buffer,
            0,
            bytemuck::bytes_of(&post),
        );

        // Resolve user pipelines for each chain entry.
        let chain_pipelines: Vec<PipelineHandle> = chain
            .iter()
            .map(|p| self.ensure_compositor_pipeline(rhi, p, args.target_size))
            .collect();

        let parity = (self.compositor_frame_index & 1) as usize;
        let prev_parity = parity ^ 1;
        self.compositor_frame_index = self.compositor_frame_index.wrapping_add(1);

        struct ChainBuilt {
            pipeline: PipelineHandle,
            write_target: TextureHandle,
            user_bg: BindGroupHandle,
            user_group_bg: Option<BindGroupHandle>,
        }
        let mut built: Vec<ChainBuilt> = Vec::with_capacity(chain.len());
        let mut prev_output = scene_color;
        for (i, path) in chain.iter().enumerate() {
            let Some([a, b]) = self.chain_textures.get(*path).copied() else {
                continue;
            };
            let pingpong = [a, b];
            let write_target = pingpong[parity];
            let prev_color = pingpong[prev_parity];
            let input_color = prev_output;
            let bg_res = rhi.create_bind_group(&BindGroupDesc {
                label: Some("scene.chain.user_bg"),
                layout: self.compositor_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Texture(input_color),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Texture(prev_color),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(self.compositor_sampler),
                    },
                ],
            });
            let bg = match bg_res {
                Ok(b) => b,
                Err(e) => {
                    log::warn!(
                        "scene: chain bg create failed for `{}`: {e}",
                        path.as_str()
                    );
                    break;
                }
            };
            self.compositor_deferred_bgs.push(bg);
            let user_group_bg = self
                .compositor_user_pipelines
                .get(*path)
                .and_then(|c| c.user_group.as_ref())
                .map(|g| g.bg);
            built.push(ChainBuilt {
                pipeline: chain_pipelines[i],
                write_target,
                user_bg: bg,
                user_group_bg,
            });
            prev_output = write_target;
        }

        let viewport = Viewport {
            x: args.rect.0,
            y: args.rect.1,
            width: args.rect.2,
            height: args.rect.3,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = Rect {
            x: args.rect.0.max(0.0) as u32,
            y: args.rect.1.max(0.0) as u32,
            width: args.rect.2.max(0.0) as u32,
            height: args.rect.3.max(0.0) as u32,
        };

        // Chain passes.
        for b in &built {
            let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
                label: Some("scene.compositor.chain"),
                color_attachments: &[Some(ColorAttachment {
                    view: b.write_target,
                    resolve_target: None,
                    load: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                    store: StoreOp::Store,
                })],
                depth_stencil_attachment: None,
            });
            p.set_viewport(viewport);
            p.set_scissor(scissor);
            p.set_pipeline(b.pipeline);
            p.set_bind_group(0, b.user_bg, &[]);
            p.set_bind_group(1, self.compositor_uniform_bg, &[]);
            if let Some(g2) = b.user_group_bg {
                p.set_bind_group(2, g2, &[]);
            }
            p.draw(0..3, 0..1);
        }

        // Final identity blit reads last shader's output, writes to target.
        if let Some(last) = built.last() {
            let blit_bg = match rhi.create_bind_group(&BindGroupDesc {
                label: Some("scene.compositor.blit_bg"),
                layout: self.compositor_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Texture(last.write_target),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Texture(scene_color),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(self.compositor_sampler),
                    },
                ],
            }) {
                Ok(bg) => {
                    self.compositor_deferred_bgs.push(bg);
                    Some(bg)
                }
                Err(e) => {
                    log::warn!("scene: compositor blit bg create failed: {e}");
                    None
                }
            };
            if let Some(bg) = blit_bg {
                let mut p = cl.begin_render_pass(&ocarina_rhi::RenderPassDesc {
                    label: Some("scene.compositor.blit"),
                    color_attachments: &[Some(ColorAttachment {
                        view: args.target,
                        resolve_target: None,
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    })],
                    depth_stencil_attachment: None,
                });
                p.set_viewport(viewport);
                p.set_scissor(scissor);
                p.set_pipeline(self.compositor_default_pipeline);
                p.set_bind_group(0, bg, &[]);
                p.set_bind_group(1, self.compositor_uniform_bg, &[]);
                p.draw(0..3, 0..1);
            }
        }
    }

    /// Resolve / hot-reload-update / async-build the user shader
    /// pipeline for one chain entry. Returns the best handle to
    /// render with right now (cached pipeline if a build is in
    /// flight; default identity if nothing has compiled yet).
    fn ensure_compositor_pipeline(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        extent: (u32, u32),
    ) -> PipelineHandle {
        if path.is_empty() {
            return self.compositor_default_pipeline;
        }
        // Drain async completions first so this frame sees them.
        self.drain_compositor_completions(rhi, extent);

        self.shader_registry.invalidate(path);
        let Some(source) = self.shader_registry.ensure(path) else {
            return self
                .compositor_user_pipelines
                .get(path)
                .map(|c| c.pipeline)
                .unwrap_or(self.compositor_default_pipeline);
        };
        let current_hash = hash_str(&source.wgsl);

        if let Some(cached) = self.compositor_user_pipelines.get(path) {
            let extent_ok = match &cached.user_group {
                Some(g) => g.extent == extent,
                None => true,
            };
            if cached.source_hash == current_hash && extent_ok {
                return cached.pipeline;
            }
        }

        let in_flight_hash = self
            .compositor_build_state
            .get(path)
            .and_then(|s| s.in_flight.as_ref())
            .map(|f| f.hash);
        if in_flight_hash == Some(current_hash) {
            return self
                .compositor_user_pipelines
                .get(path)
                .map(|c| c.pipeline)
                .unwrap_or(self.compositor_default_pipeline);
        }
        if in_flight_hash.is_some() {
            let state = self
                .compositor_build_state
                .entry(path.clone())
                .or_default();
            state.pending = Some(PendingBuild {
                hash: current_hash,
                source: source.wgsl.to_string(),
            });
        } else {
            let source_text = source.wgsl.to_string();
            match self.submit_compositor_build(rhi, path, current_hash, &source_text, extent) {
                Ok(in_flight) => {
                    let state = self
                        .compositor_build_state
                        .entry(path.clone())
                        .or_default();
                    state.in_flight = Some(in_flight);
                }
                Err(e) => {
                    log::warn!(
                        "scene: failed to submit async pipeline build for `{}`: {e}",
                        path.as_str()
                    );
                }
            }
        }
        self.compositor_user_pipelines
            .get(path)
            .map(|c| c.pipeline)
            .unwrap_or(self.compositor_default_pipeline)
    }

    fn submit_compositor_build(
        &mut self,
        rhi: &mut dyn Rhi,
        path: &AssetPath,
        hash: u64,
        wgsl: &str,
        extent: (u32, u32),
    ) -> Result<InFlightBuild, RhiError> {
        let reflection = match reflect::reflect(wgsl) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("scene: reflect failed `{}`: {e}", path.as_str());
                Default::default()
            }
        };
        let user_group = build_user_group(rhi, &reflection, extent, self.target_format);

        let vs = rhi.create_shader(&ShaderDesc {
            label: Some("compositor.user:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(wgsl),
        })?;
        let fs = match rhi.create_shader(&ShaderDesc {
            label: Some("compositor.user:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(wgsl),
        }) {
            Ok(h) => h,
            Err(e) => {
                rhi.destroy_shader(vs);
                destroy_user_group(rhi, user_group);
                return Err(e);
            }
        };

        let mut bgls: Vec<ocarina_rhi::BindGroupLayoutHandle> =
            vec![self.compositor_bgl, self.compositor_uniform_bgl];
        if let Some(ug) = user_group.as_ref() {
            bgls.push(ug.bgl);
        }
        let color_targets = [Some(ColorTargetState {
            format: self.target_format,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        })];
        let desc = PipelineDesc {
            label: Some("compositor.user.async"),
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
            bind_group_layouts: &bgls,
            push_constant_ranges: &[],
        };
        let id = rhi.create_pipeline_async(&desc);
        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);
        Ok(InFlightBuild {
            id,
            hash,
            user_group,
        })
    }

    fn drain_compositor_completions(&mut self, rhi: &mut dyn Rhi, extent: (u32, u32)) {
        let completions = rhi.poll_async_pipelines();
        if completions.is_empty() {
            return;
        }
        for completion in completions {
            let mut owner_path: Option<AssetPath> = None;
            for (p, state) in self.compositor_build_state.iter() {
                if state
                    .in_flight
                    .as_ref()
                    .map(|f| f.id == completion.id)
                    .unwrap_or(false)
                {
                    owner_path = Some(p.clone());
                    break;
                }
            }
            let Some(path) = owner_path else {
                if let Ok(handle) = completion.result {
                    rhi.destroy_pipeline(handle);
                }
                continue;
            };
            let (in_flight, pending) = {
                let state = self.compositor_build_state.get_mut(&path).unwrap();
                (
                    state.in_flight.take().expect("matched above"),
                    state.pending.take(),
                )
            };
            match completion.result {
                Ok(handle) => {
                    if let Some(stale) = self.compositor_user_pipelines.remove(&path) {
                        rhi.destroy_pipeline(stale.pipeline);
                        destroy_user_group(rhi, stale.user_group);
                    }
                    self.compositor_user_pipelines.insert(
                        path.clone(),
                        UserCompositor {
                            pipeline: handle,
                            source_hash: in_flight.hash,
                            user_group: in_flight.user_group,
                        },
                    );
                }
                Err(e) => {
                    log::warn!(
                        "scene: async compositor build failed for `{}`: {e}",
                        path.as_str()
                    );
                    destroy_user_group(rhi, in_flight.user_group);
                }
            }
            if let Some(pending) = pending {
                let _ = self.submit_compositor_build(
                    rhi,
                    &path,
                    pending.hash,
                    &pending.source,
                    extent,
                );
            }
        }
    }
}

// ─── Compositor cache types ─────────────────────────────────────

#[derive(Default)]
struct CompositorBuildState {
    in_flight: Option<InFlightBuild>,
    pending: Option<PendingBuild>,
}

struct InFlightBuild {
    id: AsyncPipelineId,
    hash: u64,
    user_group: Option<UserGroup>,
}

struct PendingBuild {
    hash: u64,
    source: String,
}

struct UserCompositor {
    pipeline: PipelineHandle,
    source_hash: u64,
    user_group: Option<UserGroup>,
}

struct UserGroup {
    bgl: ocarina_rhi::BindGroupLayoutHandle,
    bg: BindGroupHandle,
    uniform: Option<UserUniform>,
    storage_textures: Vec<TextureHandle>,
    extent: (u32, u32),
}

struct UserUniform {
    buffer: BufferHandle,
    #[allow(dead_code)]
    size: u32,
}

fn hash_str(s: &str) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(s.as_bytes());
    h.finish()
}

fn build_compositor_pipeline(
    rhi: &mut dyn Rhi,
    vs: ocarina_rhi::ShaderHandle,
    fs: ocarina_rhi::ShaderHandle,
    scene_bgl: ocarina_rhi::BindGroupLayoutHandle,
    uniform_bgl: ocarina_rhi::BindGroupLayoutHandle,
    user_bgl: Option<ocarina_rhi::BindGroupLayoutHandle>,
    color_format: TextureFormat,
    label: &'static str,
) -> RhiResult<PipelineHandle> {
    let color_targets = [Some(ColorTargetState {
        format: color_format,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];
    let layouts_with: [ocarina_rhi::BindGroupLayoutHandle; 3];
    let layouts_without: [ocarina_rhi::BindGroupLayoutHandle; 2];
    let layouts: &[ocarina_rhi::BindGroupLayoutHandle] = match user_bgl {
        Some(b) => {
            layouts_with = [scene_bgl, uniform_bgl, b];
            &layouts_with
        }
        None => {
            layouts_without = [scene_bgl, uniform_bgl];
            &layouts_without
        }
    };
    rhi.create_pipeline(&PipelineDesc {
        label: Some(label),
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
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    })
}

fn build_user_group(
    rhi: &mut dyn Rhi,
    reflection: &reflect::ShaderReflection,
    extent: (u32, u32),
    target_format: TextureFormat,
) -> Option<UserGroup> {
    let mut user_bindings: Vec<&reflect::BindingInfo> = reflection
        .bindings
        .iter()
        .filter(|b| b.group == 2 && !b.is_engine_managed())
        .collect();
    if user_bindings.is_empty() {
        return None;
    }
    user_bindings.sort_by_key(|b| b.binding);

    let mut bgl_entries: Vec<BindGroupLayoutEntry> = Vec::new();
    let mut uniform_plan: Option<(u32, u32, &Vec<UniformField>)> = None;
    struct StoragePlan {
        binding: u32,
        format: TextureFormat,
        access: ocarina_rhi::StorageTextureAccess,
        view_dimension: TextureViewDimension,
    }
    let mut storage_plans: Vec<StoragePlan> = Vec::new();
    for b in &user_bindings {
        match &b.kind {
            ReflectBindingKind::Uniform {
                size_bytes, fields, ..
            } => {
                let aligned = ((*size_bytes + 15) / 16) * 16;
                bgl_entries.push(BindGroupLayoutEntry {
                    binding: b.binding,
                    visibility: ShaderStageFlags::FRAGMENT | ShaderStageFlags::VERTEX,
                    binding_type: BindingType::UniformBuffer {
                        has_dynamic_offset: false,
                        min_binding_size: Some(u64::from(aligned.max(16))),
                    },
                    count: None,
                });
                uniform_plan = Some((b.binding, aligned, fields));
            }
            ReflectBindingKind::StorageTexture {
                format,
                dimension,
                access,
            } => {
                let Some(rhi_format) = storage_format_to_rhi(*format) else {
                    log::warn!(
                        "scene: storage texture binding {} unsupported format {:?}",
                        b.binding,
                        format
                    );
                    continue;
                };
                let rhi_access = storage_access_to_rhi(*access);
                let rhi_dim = reflect_dim_to_rhi(*dimension);
                bgl_entries.push(BindGroupLayoutEntry {
                    binding: b.binding,
                    visibility: ShaderStageFlags::FRAGMENT,
                    binding_type: BindingType::StorageTexture {
                        access: rhi_access,
                        format: rhi_format,
                        view_dimension: rhi_dim,
                    },
                    count: None,
                });
                storage_plans.push(StoragePlan {
                    binding: b.binding,
                    format: rhi_format,
                    access: rhi_access,
                    view_dimension: rhi_dim,
                });
            }
            _ => {
                log::warn!(
                    "scene: user binding kind not wired (group {} binding {})",
                    b.group,
                    b.binding
                );
            }
        }
    }
    if bgl_entries.is_empty() {
        return None;
    }
    let _ = target_format;
    let bgl = match rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("compositor.user.group2.bgl"),
        entries: &bgl_entries,
    }) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("scene: user group2 bgl create failed: {e}");
            return None;
        }
    };

    let mut uniform: Option<UserUniform> = None;
    let mut storage_textures: Vec<TextureHandle> = Vec::new();
    let mut bg_entries: Vec<BindGroupEntry> = Vec::new();

    if let Some((binding, size, fields)) = uniform_plan {
        let buffer = match rhi.create_buffer(&BufferDesc {
            label: Some("compositor.user.uniform"),
            size: u64::from(size),
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        }) {
            Ok(h) => h,
            Err(e) => {
                log::warn!("scene: user uniform buffer create failed: {e}");
                rhi.destroy_bind_group_layout(bgl);
                return None;
            }
        };
        let mut bytes = vec![0u8; size as usize];
        for f in fields {
            pack_default(&mut bytes, f);
        }
        rhi.write_buffer(buffer, 0, &bytes);
        bg_entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::Buffer {
                buffer,
                offset: 0,
                size: Some(u64::from(size)),
            },
        });
        uniform = Some(UserUniform { buffer, size });
    }

    for plan in &storage_plans {
        let usage = TextureUsage::STORAGE_BINDING | TextureUsage::TEXTURE_BINDING;
        let tex = match rhi.create_texture(&TextureDesc {
            label: Some("compositor.user.storage_texture"),
            dimension: TextureDimension::D2 {
                width: extent.0,
                height: extent.1,
                array_layers: 1,
            },
            format: plan.format,
            usage,
            mip_levels: 1,
            sample_count: 1,
        }) {
            Ok(h) => h,
            Err(e) => {
                log::warn!("scene: user storage texture create failed: {e}");
                if let Some(u) = uniform {
                    rhi.destroy_buffer(u.buffer);
                }
                for t in storage_textures {
                    rhi.destroy_texture(t);
                }
                rhi.destroy_bind_group_layout(bgl);
                return None;
            }
        };
        bg_entries.push(BindGroupEntry {
            binding: plan.binding,
            resource: BindingResource::Texture(tex),
        });
        storage_textures.push(tex);
        let _ = plan.access;
        let _ = plan.view_dimension;
    }

    let bg = match rhi.create_bind_group(&BindGroupDesc {
        label: Some("compositor.user.group2.bg"),
        layout: bgl,
        entries: &bg_entries,
    }) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("scene: user group2 bg create failed: {e}");
            if let Some(u) = uniform {
                rhi.destroy_buffer(u.buffer);
            }
            for t in storage_textures {
                rhi.destroy_texture(t);
            }
            rhi.destroy_bind_group_layout(bgl);
            return None;
        }
    };

    Some(UserGroup {
        bgl,
        bg,
        uniform,
        storage_textures,
        extent,
    })
}

fn destroy_user_group(rhi: &mut dyn Rhi, group: Option<UserGroup>) {
    let Some(g) = group else {
        return;
    };
    rhi.destroy_bind_group(g.bg);
    for t in g.storage_textures {
        rhi.destroy_texture(t);
    }
    if let Some(u) = g.uniform {
        rhi.destroy_buffer(u.buffer);
    }
    rhi.destroy_bind_group_layout(g.bgl);
}

fn storage_format_to_rhi(f: reflect::StorageFormat) -> Option<TextureFormat> {
    match f {
        reflect::StorageFormat::Rgba8Unorm => Some(TextureFormat::Rgba8Unorm),
        reflect::StorageFormat::Rgba16Float => Some(TextureFormat::Rgba16Float),
        reflect::StorageFormat::Rgba32Float => Some(TextureFormat::Rgba32Float),
        reflect::StorageFormat::R32Float => Some(TextureFormat::R32Float),
        reflect::StorageFormat::Rgba8Snorm | reflect::StorageFormat::Other => None,
    }
}

fn storage_access_to_rhi(a: reflect::StorageAccess) -> ocarina_rhi::StorageTextureAccess {
    match a {
        reflect::StorageAccess::Load => ocarina_rhi::StorageTextureAccess::ReadOnly,
        reflect::StorageAccess::Store => ocarina_rhi::StorageTextureAccess::WriteOnly,
        reflect::StorageAccess::LoadStore => ocarina_rhi::StorageTextureAccess::ReadWrite,
    }
}

fn reflect_dim_to_rhi(d: reflect::TextureDimension) -> TextureViewDimension {
    match d {
        reflect::TextureDimension::D1 => TextureViewDimension::D1,
        reflect::TextureDimension::D2 => TextureViewDimension::D2,
        reflect::TextureDimension::D3 => TextureViewDimension::D3,
        reflect::TextureDimension::D2Array => TextureViewDimension::D2Array,
        reflect::TextureDimension::Cube => TextureViewDimension::Cube,
        reflect::TextureDimension::CubeArray => TextureViewDimension::CubeArray,
    }
}

fn pack_default(bytes: &mut [u8], field: &UniformField) {
    let off = field.offset as usize;
    let Some(default) = &field.metadata.default else {
        return;
    };
    let write_f32 = |out: &mut [u8], at: usize, v: f32| {
        let end = at + 4;
        if end <= out.len() {
            out[at..end].copy_from_slice(&v.to_le_bytes());
        }
    };
    match (default, field.kind) {
        (DefaultValue::F32(v), ScalarKind::F32) => write_f32(bytes, off, *v),
        (DefaultValue::Vec2(v), ScalarKind::Vec2) => {
            write_f32(bytes, off, v[0]);
            write_f32(bytes, off + 4, v[1]);
        }
        (DefaultValue::Vec3(v), ScalarKind::Vec3) => {
            write_f32(bytes, off, v[0]);
            write_f32(bytes, off + 4, v[1]);
            write_f32(bytes, off + 8, v[2]);
        }
        (DefaultValue::Vec4(v), ScalarKind::Vec4) => {
            write_f32(bytes, off, v[0]);
            write_f32(bytes, off + 4, v[1]);
            write_f32(bytes, off + 8, v[2]);
            write_f32(bytes, off + 12, v[3]);
        }
        _ => {}
    }
}

