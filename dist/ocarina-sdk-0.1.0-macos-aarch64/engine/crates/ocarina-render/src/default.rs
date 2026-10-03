//! Default scene pipeline + UBO formats.
//!
//! The "draw a mesh through PBR with proper lights and material
//! params + textures" path used by every host. The shader bound here
//! is the engine's built-in `engine/shaders/default`; alternative
//! shaders selected via [`RenderSettings`](crate::shader::RenderSettings)
//! reuse the same bind group layouts and resources, so swapping a
//! shader at runtime is just `set_pipeline(other_handle)` -- the
//! camera / lights / instance / material bindings carry over.
//!
//! ## Bind group layout (final, post-stage-7)
//!
//! Group 0 (frame, set once per frame):
//!   - binding 0: camera UBO
//!   - binding 1: lights UBO
//!   - binding 2: linear sampler
//!
//! Group 1 (per-draw, dynamic offsets):
//!   - binding 0: instance UBO (dynamic offset)
//!   - binding 1: material UBO (dynamic offset)
//!
//! Group 2 (per-material, built per-entity, cached by texture set):
//!   - binding 0: albedo texture (sRGB)
//!   - binding 1: normal texture (linear)
//!   - binding 2: metallic-roughness texture (linear, glTF-packed:
//!                green = roughness, blue = metallic)
//!   - binding 3: emission texture (sRGB)
//!
//! Mask / gizmo / unlit shaders use a `simple_instance_bgl` (just
//! the instance UBO at group 1, no material, no textures) and skip
//! group 2 entirely.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use ocarina_rhi::{
    AddressMode, BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BufferDesc,
    BufferHandle, BufferUsage, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState,
    FilterMode, FragmentStage, GraphicsPipelineDesc, MemoryLocation, MultisampleState,
    PipelineDesc, PipelineHandle, PipelineKind, PrimitiveState, Rhi, RhiResult, SamplerBindingType,
    SamplerDesc, SamplerHandle, ShaderBytecode, ShaderDesc, ShaderStage, ShaderStageFlags,
    TextureFormat, TextureHandle, TextureSampleType, TextureViewDimension, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexStage, VertexStepMode,
};

use ocarina_core::AssetPath;

use crate::mesh::Vertex;

// ────────────────────────── Camera UBO ──────────────────────────

/// Camera UBO at (0, 0). 96 bytes; 16-byte aligned by std140 rules.
///
/// `eye_pos` is the camera's world-space position, needed by PBR
/// shading for the view direction term. `viewport_size` is the panel
/// rect in physical pixels (xy); the editor's 2D gizmo shader reads
/// this for pixel-to-NDC conversion. Default and unlit shaders
/// declare these fields for layout compatibility and ignore
/// `viewport_size`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CameraUniform {
    pub view_proj: Mat4,
    pub eye_pos: Vec4,
    pub viewport_size: Vec4,
}

impl CameraUniform {
    /// Camera UBO with `view_proj` set, `eye_pos` and
    /// `viewport_size` zeroed. Hosts fill the remaining fields
    /// before uploading.
    #[must_use]
    pub fn standard(view_proj: Mat4) -> Self {
        Self {
            view_proj,
            eye_pos: Vec4::ZERO,
            viewport_size: Vec4::ZERO,
        }
    }
}

pub const CAMERA_UNIFORM_SIZE: u64 = std::mem::size_of::<CameraUniform>() as u64;

// ────────────────────────── Light UBO ──────────────────────────

/// Encoding of [`LightKind`] used in shader-side `Light.kind_pad.x`.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum LightKind {
    Directional = 0,
    Point = 1,
    Spot = 2,
}

/// One slot in the lights array. 80 bytes; 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct LightUniform {
    pub kind_pad: [u32; 4],
    pub color: Vec4,
    pub position: Vec4,
    pub direction: Vec4,
    pub params: Vec4,
}

pub const LIGHT_UNIFORM_SIZE: u64 = std::mem::size_of::<LightUniform>() as u64;

/// Maximum lights uploaded per frame. The shader iterates this
/// fixed-size array up to `count`; raising it is a one-line change
/// here plus a matching constant in `default.wgsl`.
pub const MAX_LIGHTS: usize = 16;

/// All scene lights packed into one UBO.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct LightsUniform {
    pub count_pad: [u32; 4],
    pub lights: [LightUniform; MAX_LIGHTS],
}

impl LightsUniform {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            count_pad: [0; 4],
            lights: [LightUniform::zeroed(); MAX_LIGHTS],
        }
    }
}

pub const LIGHTS_UNIFORM_SIZE: u64 = std::mem::size_of::<LightsUniform>() as u64;

// ────────────────────────── Light instance ──────────────────────────

/// CPU-side description of one scene light. The renderer collects
/// these from World queries (`DirectionalLight + Transform`,
/// `PointLight + Transform`, `SpotLight + Transform`) once per
/// frame, packs into a [`LightsUniform`], and uploads.
#[derive(Clone, Copy, Debug)]
pub enum LightInstance {
    Directional {
        color: Vec3,
        intensity: f32,
        direction: Vec3,
    },
    Point {
        color: Vec3,
        intensity: f32,
        position: Vec3,
        range: f32,
    },
    Spot {
        color: Vec3,
        intensity: f32,
        position: Vec3,
        direction: Vec3,
        range: f32,
        inner_cos: f32,
        outer_cos: f32,
    },
}

impl LightInstance {
    /// Pack into the shader-facing [`LightUniform`] format.
    #[must_use]
    pub fn to_uniform(self) -> LightUniform {
        match self {
            Self::Directional {
                color,
                intensity,
                direction,
            } => LightUniform {
                kind_pad: [LightKind::Directional as u32, 0, 0, 0],
                color: color.extend(intensity),
                position: Vec4::ZERO,
                direction: direction.extend(0.0),
                params: Vec4::ZERO,
            },
            Self::Point {
                color,
                intensity,
                position,
                range,
            } => LightUniform {
                kind_pad: [LightKind::Point as u32, 0, 0, 0],
                color: color.extend(intensity),
                position: position.extend(0.0),
                direction: Vec4::ZERO,
                params: Vec4::new(range, 0.0, 0.0, 0.0),
            },
            Self::Spot {
                color,
                intensity,
                position,
                direction,
                range,
                inner_cos,
                outer_cos,
            } => LightUniform {
                kind_pad: [LightKind::Spot as u32, 0, 0, 0],
                color: color.extend(intensity),
                position: position.extend(0.0),
                direction: direction.extend(0.0),
                params: Vec4::new(range, inner_cos, outer_cos, 0.0),
            },
        }
    }
}

#[must_use]
pub fn build_lights_uniform(lights: &[LightInstance]) -> LightsUniform {
    let count = lights.len().min(MAX_LIGHTS);
    let mut packed = LightsUniform::empty();
    packed.count_pad[0] = count as u32;
    for (i, light) in lights.iter().take(count).enumerate() {
        packed.lights[i] = light.to_uniform();
    }
    packed
}

pub fn collect_lights_from_world(
    world: &mut bevy_ecs::world::World,
    world_xforms: &std::collections::HashMap<bevy_ecs::entity::Entity, Mat4>,
) -> Vec<LightInstance> {
    use bevy_ecs::entity::Entity;
    use ocarina_core::components::{DirectionalLight, PointLight, SpotLight};

    let mut out = Vec::new();
    {
        let mut q = world.query::<(Entity, &DirectionalLight)>();
        for (entity, light) in q.iter(world) {
            let xform = world_xforms
                .get(&entity)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let direction = xform
                .transform_vector3(Vec3::NEG_Z)
                .normalize_or_zero();
            out.push(LightInstance::Directional {
                color: light.color,
                intensity: light.intensity,
                direction,
            });
        }
    }
    {
        let mut q = world.query::<(Entity, &PointLight)>();
        for (entity, light) in q.iter(world) {
            let xform = world_xforms
                .get(&entity)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let position = xform.col(3).truncate();
            out.push(LightInstance::Point {
                color: light.color,
                intensity: light.intensity,
                position,
                range: light.range,
            });
        }
    }
    {
        let mut q = world.query::<(Entity, &SpotLight)>();
        for (entity, light) in q.iter(world) {
            let xform = world_xforms
                .get(&entity)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let position = xform.col(3).truncate();
            let direction = xform
                .transform_vector3(Vec3::NEG_Z)
                .normalize_or_zero();
            let inner_cos = light.inner_angle.to_radians().cos();
            let outer_cos = light.outer_angle.to_radians().cos();
            out.push(LightInstance::Spot {
                color: light.color,
                intensity: light.intensity,
                position,
                direction,
                range: light.range,
                inner_cos,
                outer_cos,
            });
        }
    }
    out
}

// ────────────────────────── Instance UBO ──────────────────────────

/// Per-instance UBO at (1, 0). 80 bytes of payload, padded by the
/// stride to satisfy wgpu's `min_uniform_buffer_offset_alignment`.
///
/// `tint` is the per-entity color the editor's mask/gizmo/unlit
/// shaders read directly (those shaders don't go through the PBR
/// material UBO). The default scene shader reads
/// `material.albedo * instance.tint` so the two stay consistent --
/// the renderer mirrors `Material.albedo` into both slots.
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

/// Cap on per-frame instances. Way more than current scenes need.
pub const MAX_INSTANCES: u32 = 1024;

// ────────────────────────── Material UBO ──────────────────────────

/// Per-instance material parameters. 64 bytes payload, 16-byte
/// aligned. Multiplied with the 4 sampled textures by the default
/// scene shader to produce the final PBR inputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MaterialUniform {
    /// `rgb` = base color tint; `a` = alpha.
    pub albedo: Vec4,
    /// `rgb` = emission color; `a` = emission strength multiplier.
    pub emission: Vec4,
    /// `x` = metallic, `y` = roughness, `z` = reserved, `w` = reserved.
    pub metal_rough: Vec4,
}

pub const MATERIAL_UNIFORM_SIZE: u64 = std::mem::size_of::<MaterialUniform>() as u64;

/// Stride between consecutive entries in the material buffer.
/// Same 256-byte alignment rule as the instance buffer.
pub const MATERIAL_STRIDE: u64 = 256;

impl MaterialUniform {
    /// Default material: opaque white, roughness 0.5, metallic 0,
    /// no emission. Identity for the PBR math.
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            albedo: Vec4::ONE,
            emission: Vec4::new(0.0, 0.0, 0.0, 0.0),
            metal_rough: Vec4::new(0.0, 0.5, 0.0, 0.0),
        }
    }

    /// Build from an [`ocarina_core::components::Material`] component.
    #[must_use]
    pub fn from_component(material: &ocarina_core::components::Material) -> Self {
        Self {
            albedo: material.albedo,
            emission: material
                .emission_color
                .extend(material.emission_strength),
            metal_rough: Vec4::new(material.metallic, material.roughness, 0.0, 0.0),
        }
    }
}

// ────────────────────────── Pipeline ──────────────────────────

/// Source for the engine's built-in default scene shader.
pub const DEFAULT_SCENE_WGSL: &str = include_str!("../shaders/default.wgsl");

/// All RHI handles owned by the default pipeline factory. Hosts hold
/// one of these for the lifetime of the renderer.
pub struct DefaultPipeline {
    /// Pipeline using `engine/shaders/default`. Alternative scene
    /// shaders live in [`Self::pipeline_cache`].
    pub pipeline: PipelineHandle,
    target_format: TextureFormat,
    depth_format: TextureFormat,

    // ─── Group 0: frame ───
    pub frame_bgl: BindGroupLayoutHandle,
    pub camera_buffer: BufferHandle,
    pub lights_buffer: BufferHandle,
    pub sampler: SamplerHandle,
    pub frame_bg: BindGroupHandle,

    // ─── Group 1 (scene): instance + material UBOs ───
    pub instance_bgl: BindGroupLayoutHandle,
    pub instance_buffer: BufferHandle,
    pub material_buffer: BufferHandle,
    pub instance_bg: BindGroupHandle,

    // ─── Group 2 (scene): material textures ───
    pub material_textures_bgl: BindGroupLayoutHandle,

    // ─── Group 1 (mask/gizmo/unlit): instance only ───
    pub simple_instance_bgl: BindGroupLayoutHandle,
    pub simple_instance_bg: BindGroupHandle,

    /// Backwards-compat aliases for the bgl/bg names callers used in
    /// stages 1-6. New code should prefer `frame_*`.
    pub camera_bgl: BindGroupLayoutHandle,
    pub camera_bg: BindGroupHandle,

    /// Lazily-built pipelines keyed by scene-shader path. Each entry
    /// stores the pipeline handle plus a 64-bit hash of the WGSL
    /// source it was built from -- on every `pipeline_for` call we
    /// re-read the file and compare hashes so save-then-render
    /// triggers a hot-recompile without requiring a TOML change.
    pub pipeline_cache: HashMap<AssetPath, ScenePipelineCacheEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScenePipelineCacheEntry {
    pub pipeline: PipelineHandle,
    pub source_hash: u64,
}

fn hash_wgsl(s: &str) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(s.as_bytes());
    h.finish()
}

impl DefaultPipeline {
    /// Return the pipeline handle for `shader_path`, building it on
    /// first request and rebuilding if the on-disk source changed
    /// since the last call. Returns the built-in default if the
    /// build fails (engine/shader registry handles the warning).
    pub fn pipeline_for(
        &mut self,
        rhi: &mut dyn Rhi,
        registry: &mut crate::shader::ShaderRegistry,
        shader_path: &AssetPath,
    ) -> PipelineHandle {
        // Re-read the source so a save in the user's editor shows
        // up immediately. The registry's source cache is for cross-
        // consumer sharing, not a hot-reload barrier.
        registry.invalidate(shader_path);
        let source = registry.ensure_or_default(shader_path);
        let current_hash = hash_wgsl(&source.wgsl);
        if let Some(entry) = self.pipeline_cache.get(shader_path) {
            if entry.source_hash == current_hash {
                return entry.pipeline;
            }
            // Source changed on disk. Drop the stale pipeline so its
            // wgpu module / pipeline can be released before the
            // rebuild below allocates a fresh one.
            let stale = self.pipeline_cache.remove(shader_path).unwrap();
            rhi.destroy_pipeline(stale.pipeline);
        }
        match build_scene_pipeline_against(
            rhi,
            self.frame_bgl,
            self.instance_bgl,
            self.material_textures_bgl,
            &source.label,
            &source.wgsl,
            self.target_format,
            self.depth_format,
            PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
        ) {
            Ok(handle) => {
                self.pipeline_cache.insert(
                    shader_path.clone(),
                    ScenePipelineCacheEntry {
                        pipeline: handle,
                        source_hash: current_hash,
                    },
                );
                handle
            }
            Err(e) => {
                log::warn!(
                    "ocarina-render: failed to build pipeline for `{}`: {e} -- falling back to default",
                    shader_path.as_str()
                );
                self.pipeline
            }
        }
    }
}

/// Build the default pipeline + UBOs against `rhi`. `target_format`
/// is the swapchain / panel-target color format the pipeline will be
/// drawn into. `depth_format` is the renderer's depth attachment.
pub fn build_default_pipeline(
    rhi: &mut dyn Rhi,
    target_format: TextureFormat,
    depth_format: TextureFormat,
) -> RhiResult<DefaultPipeline> {
    // ─── Group 0: camera + lights + sampler ───
    let frame_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.frame.bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(CAMERA_UNIFORM_SIZE),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(LIGHTS_UNIFORM_SIZE),
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
    let camera_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.camera.ubo"),
        size: CAMERA_UNIFORM_SIZE,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let lights_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.lights.ubo"),
        size: LIGHTS_UNIFORM_SIZE,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let zero_lights = LightsUniform::empty();
    rhi.write_buffer(lights_buffer, 0, bytemuck::bytes_of(&zero_lights));

    let sampler = rhi.create_sampler(&SamplerDesc {
        label: Some("render.sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..SamplerDesc::default()
    })?;

    let frame_bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.frame.bg"),
        layout: frame_bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: camera_buffer,
                    offset: 0,
                    size: Some(CAMERA_UNIFORM_SIZE),
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: lights_buffer,
                    offset: 0,
                    size: Some(LIGHTS_UNIFORM_SIZE),
                },
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(sampler),
            },
        ],
    })?;

    // ─── Group 1 (scene): instance + material both dyn-offset ───
    let instance_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.instance.bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: true,
                    min_binding_size: Some(INSTANCE_UNIFORM_SIZE),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: true,
                    min_binding_size: Some(MATERIAL_UNIFORM_SIZE),
                },
                count: None,
            },
        ],
    })?;
    let instance_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.instance.ubo"),
        size: INSTANCE_STRIDE * u64::from(MAX_INSTANCES),
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let material_buffer = rhi.create_buffer(&BufferDesc {
        label: Some("render.material.ubo"),
        size: MATERIAL_STRIDE * u64::from(MAX_INSTANCES),
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    let instance_bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.instance.bg"),
        layout: instance_bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: instance_buffer,
                    offset: 0,
                    size: Some(INSTANCE_UNIFORM_SIZE),
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: material_buffer,
                    offset: 0,
                    size: Some(MATERIAL_UNIFORM_SIZE),
                },
            },
        ],
    })?;

    // ─── Group 2 (scene): 4 material textures ───
    let texture_entry = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStageFlags::FRAGMENT,
        binding_type: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let material_textures_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.material_textures.bgl"),
        entries: &[
            texture_entry(0),
            texture_entry(1),
            texture_entry(2),
            texture_entry(3),
        ],
    })?;

    // ─── Group 1 (simple, for mask/gizmo/unlit): instance only ───
    let simple_instance_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("render.simple_instance.bgl"),
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
    let simple_instance_bg = rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.simple_instance.bg"),
        layout: simple_instance_bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: instance_buffer,
                offset: 0,
                size: Some(INSTANCE_UNIFORM_SIZE),
            },
        }],
    })?;

    // ─── Default scene pipeline ───
    let pipeline = build_scene_pipeline_against(
        rhi,
        frame_bgl,
        instance_bgl,
        material_textures_bgl,
        "render.default",
        DEFAULT_SCENE_WGSL,
        target_format,
        depth_format,
        PrimitiveState {
            cull_mode: None,
            ..PrimitiveState::default()
        },
    )?;

    let mut pipeline_cache = HashMap::new();
    // The engine default shader is built-in; its source is the
    // bundled `default.wgsl` constant, never re-read from disk.
    // Hash it once so the hot-reload path's hash compare is happy
    // (it'll always match itself, so the cache always hits).
    let default_source_hash = hash_wgsl(crate::shader::ENGINE_DEFAULT_WGSL);
    pipeline_cache.insert(
        AssetPath::new(crate::shader::ENGINE_SHADER_DEFAULT),
        ScenePipelineCacheEntry {
            pipeline,
            source_hash: default_source_hash,
        },
    );

    Ok(DefaultPipeline {
        pipeline,
        target_format,
        depth_format,
        frame_bgl,
        camera_buffer,
        lights_buffer,
        sampler,
        frame_bg,
        instance_bgl,
        instance_buffer,
        material_buffer,
        instance_bg,
        material_textures_bgl,
        simple_instance_bgl,
        simple_instance_bg,
        camera_bgl: frame_bgl,
        camera_bg: frame_bg,
        pipeline_cache,
    })
}

/// Build a scene pipeline (group 0/1/2 layout) against a specific
/// WGSL source. Used both internally for the built-in default and by
/// the cache when a project swaps in its own scene shader through
/// [`RenderSettings`](crate::shader::RenderSettings).
pub fn build_scene_pipeline_against(
    rhi: &mut dyn Rhi,
    frame_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
    material_textures_bgl: BindGroupLayoutHandle,
    label: &str,
    wgsl: &str,
    target_format: TextureFormat,
    depth_format: TextureFormat,
    primitive: PrimitiveState,
) -> RhiResult<PipelineHandle> {
    let vs_label = format!("{label}:vs_main");
    let fs_label = format!("{label}:fs_main");
    let vs = rhi.create_shader(&ShaderDesc {
        label: Some(&vs_label),
        stage: ShaderStage::Vertex,
        entry_point: "vs_main",
        bytecode: ShaderBytecode::Wgsl(wgsl),
    })?;
    let fs = rhi.create_shader(&ShaderDesc {
        label: Some(&fs_label),
        stage: ShaderStage::Fragment,
        entry_point: "fs_main",
        bytecode: ShaderBytecode::Wgsl(wgsl),
    })?;

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

    let result = rhi.create_pipeline(&PipelineDesc {
        label: Some(label),
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
        bind_group_layouts: &[frame_bgl, instance_bgl, material_textures_bgl],
        push_constant_ranges: &[],
    });

    rhi.destroy_shader(vs);
    rhi.destroy_shader(fs);
    result
}

/// Build a "simple" pipeline (group 0 + simple instance group only,
/// no material UBO, no textures). Used by the editor's unlit
/// pipeline and the viewport's mask / gizmo passes.
pub fn build_simple_pipeline_against(
    rhi: &mut dyn Rhi,
    frame_bgl: BindGroupLayoutHandle,
    simple_instance_bgl: BindGroupLayoutHandle,
    label: &str,
    wgsl: &str,
    target_format: TextureFormat,
    depth_format: TextureFormat,
    primitive: PrimitiveState,
    color_blend: Option<ocarina_rhi::BlendState>,
    depth_compare: CompareFunction,
    depth_write_enabled: bool,
    color_targets_override: Option<&[Option<ColorTargetState>]>,
) -> RhiResult<PipelineHandle> {
    let vs_label = format!("{label}:vs_main");
    let fs_label = format!("{label}:fs_main");
    let vs = rhi.create_shader(&ShaderDesc {
        label: Some(&vs_label),
        stage: ShaderStage::Vertex,
        entry_point: "vs_main",
        bytecode: ShaderBytecode::Wgsl(wgsl),
    })?;
    let fs = rhi.create_shader(&ShaderDesc {
        label: Some(&fs_label),
        stage: ShaderStage::Fragment,
        entry_point: "fs_main",
        bytecode: ShaderBytecode::Wgsl(wgsl),
    })?;

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
    let default_targets = [Some(ColorTargetState {
        format: target_format,
        blend: color_blend,
        write_mask: ColorWrites::ALL,
    })];
    let targets = color_targets_override.unwrap_or(&default_targets);

    let result = rhi.create_pipeline(&PipelineDesc {
        label: Some(label),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets,
            }),
            primitive,
            depth_stencil: Some(DepthStencilState {
                format: depth_format,
                depth_write_enabled,
                depth_compare,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                bias: Default::default(),
            }),
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[frame_bgl, simple_instance_bgl],
        push_constant_ranges: &[],
    });

    rhi.destroy_shader(vs);
    rhi.destroy_shader(fs);
    result
}

// ────────────────────────── Material bind-group helper ──────────────────────────

/// Cache key for a material texture set: the four texture handles
/// the renderer will bind to group 2. Comparing handles instead of
/// asset paths means re-sampled textures (same path, different
/// handle after a reload) get a new bind group automatically.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialTextureKey {
    pub albedo: TextureHandle,
    pub normal: TextureHandle,
    pub metallic_roughness: TextureHandle,
    pub emission: TextureHandle,
}

/// Build a per-material bind group for group 2 with the supplied
/// texture views. The renderer caches the result keyed by
/// [`MaterialTextureKey`] so two entities with the same texture set
/// share one bind group.
pub fn build_material_textures_bg(
    rhi: &mut dyn Rhi,
    layout: BindGroupLayoutHandle,
    key: MaterialTextureKey,
) -> RhiResult<BindGroupHandle> {
    rhi.create_bind_group(&BindGroupDesc {
        label: Some("render.material_textures.bg"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Texture(key.albedo),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Texture(key.normal),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Texture(key.metallic_roughness),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Texture(key.emission),
            },
        ],
    })
}
