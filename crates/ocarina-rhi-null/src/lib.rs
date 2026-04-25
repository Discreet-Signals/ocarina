//! Null RHI backend.
//!
//! A no-op implementation of [`Rhi`] used for:
//!
//! 1. **Unit tests** of the trait surface (create/destroy/record/submit
//!    round-trips work).
//! 2. **The anti-leak guardrail** — if this backend can't implement the trait
//!    without backend-specific hacks, the trait has a wgpu-ism that needs
//!    pulling out.
//!
//! Resources are tracked in a generational pool (`PoolOfAlive`) so handles
//! are validated on destroy. Submitted command lists are stashed in
//! `submitted_lists` for inspection by tests.

use ocarina_rhi::{
    AdapterKind, BindGroupDesc, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutHandle,
    BufferDesc, BufferHandle, Capabilities, CommandList, Features, Limits, PipelineDesc,
    PipelineHandle, Rhi, RhiError, RhiResult, SamplerDesc, SamplerHandle, ShaderDesc, ShaderHandle,
    SwapchainDesc, SwapchainHandle, TextureDesc, TextureHandle, TextureRegion,
};

const BACKEND_NAME: &str = "null";

/// Generational pool of "alive" slots. Each slot stores its current generation.
/// `alloc` finds a free slot (preferring reuse to keep slot numbers low);
/// `free` bumps the generation so any outstanding handle to this slot is
/// invalidated.
#[derive(Debug, Default)]
struct Pool {
    /// `generations[i]` is the current generation of slot `i`. Even = free,
    /// odd = alive.
    generations: Vec<u32>,
    /// Indices of free slots for O(1) allocation.
    free: Vec<u32>,
}

impl Pool {
    fn alloc(&mut self) -> (u32, u32) {
        if let Some(slot) = self.free.pop() {
            let generation = self.generations[slot as usize] + 1;
            self.generations[slot as usize] = generation;
            (slot, generation)
        } else {
            let slot = self.generations.len() as u32;
            self.generations.push(1);
            (slot, 1)
        }
    }

    fn free(&mut self, slot: u32, generation: u32) -> Result<(), &'static str> {
        let idx = slot as usize;
        let current = self.generations.get(idx).copied().ok_or("slot out of range")?;
        if current != generation {
            return Err("stale generation");
        }
        self.generations[idx] = current.wrapping_add(1);
        self.free.push(slot);
        Ok(())
    }

    fn is_alive(&self, slot: u32, generation: u32) -> bool {
        self.generations
            .get(slot as usize)
            .copied()
            .is_some_and(|g| g == generation)
    }

    fn alive_count(&self) -> usize {
        self.generations.iter().filter(|g| **g % 2 == 1).count()
    }
}

#[derive(Default)]
pub struct NullRhi {
    caps: Option<Capabilities>,

    buffers: Pool,
    textures: Pool,
    samplers: Pool,
    shaders: Pool,
    bgls: Pool,
    bgs: Pool,
    pipelines: Pool,
    swapchains: Pool,

    /// Swapchain-owned textures acquired this frame; freed at `present`.
    swapchain_images: Pool,

    /// Last submitted command lists, preserved for test inspection.
    pub submitted_lists: Vec<CommandList>,

    /// Monotonic counters for cheap assertions.
    pub stats: NullStats,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NullStats {
    pub creates: u64,
    pub destroys: u64,
    pub writes: u64,
    pub submits: u64,
    pub presents: u64,
    pub acquires: u64,
    pub waits: u64,
}

impl NullRhi {
    #[must_use]
    pub fn new() -> Self {
        Self {
            caps: Some(Capabilities {
                backend_name: BACKEND_NAME,
                device_name: "NullDevice".to_owned(),
                adapter_kind: AdapterKind::Cpu,
                features: Features::PUSH_CONSTANTS,
                limits: Limits::conservative(),
            }),
            ..Default::default()
        }
    }

    /// Number of live resources across all pools. Useful for "no leaks" asserts.
    #[must_use]
    pub fn alive_resources(&self) -> usize {
        self.buffers.alive_count()
            + self.textures.alive_count()
            + self.samplers.alive_count()
            + self.shaders.alive_count()
            + self.bgls.alive_count()
            + self.bgs.alive_count()
            + self.pipelines.alive_count()
            + self.swapchains.alive_count()
    }
}

// ────────────────────────── Rhi impl ──────────────────────────

impl Rhi for NullRhi {
    fn capabilities(&self) -> &Capabilities {
        self.caps
            .as_ref()
            .expect("NullRhi::new must have initialized capabilities")
    }

    fn backend_name(&self) -> &'static str {
        BACKEND_NAME
    }

    // ── create ──

    fn create_buffer(&mut self, _desc: &BufferDesc<'_>) -> RhiResult<BufferHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.buffers.alloc();
        Ok(BufferHandle::new(slot, generation))
    }

    fn create_texture(&mut self, _desc: &TextureDesc<'_>) -> RhiResult<TextureHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.textures.alloc();
        Ok(TextureHandle::new(slot, generation))
    }

    fn create_sampler(&mut self, _desc: &SamplerDesc<'_>) -> RhiResult<SamplerHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.samplers.alloc();
        Ok(SamplerHandle::new(slot, generation))
    }

    fn create_shader(&mut self, _desc: &ShaderDesc<'_>) -> RhiResult<ShaderHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.shaders.alloc();
        Ok(ShaderHandle::new(slot, generation))
    }

    fn create_bind_group_layout(
        &mut self,
        _desc: &BindGroupLayoutDesc<'_>,
    ) -> RhiResult<BindGroupLayoutHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.bgls.alloc();
        Ok(BindGroupLayoutHandle::new(slot, generation))
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> RhiResult<BindGroupHandle> {
        if !self.bgls.is_alive(desc.layout.raw().slot, desc.layout.raw().generation) {
            return Err(RhiError::InvalidHandle("BindGroupLayoutHandle"));
        }
        self.stats.creates += 1;
        let (slot, generation) = self.bgs.alloc();
        Ok(BindGroupHandle::new(slot, generation))
    }

    fn create_pipeline(&mut self, _desc: &PipelineDesc<'_>) -> RhiResult<PipelineHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.pipelines.alloc();
        Ok(PipelineHandle::new(slot, generation))
    }

    fn create_swapchain(&mut self, _desc: &SwapchainDesc<'_>) -> RhiResult<SwapchainHandle> {
        self.stats.creates += 1;
        let (slot, generation) = self.swapchains.alloc();
        Ok(SwapchainHandle::new(slot, generation))
    }

    // ── destroy (immediate in the null backend — no in-flight frames to guard) ──

    fn destroy_buffer(&mut self, h: BufferHandle) {
        if self.buffers.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_texture(&mut self, h: TextureHandle) {
        if self.textures.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_sampler(&mut self, h: SamplerHandle) {
        if self.samplers.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_shader(&mut self, h: ShaderHandle) {
        if self.shaders.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_bind_group_layout(&mut self, h: BindGroupLayoutHandle) {
        if self.bgls.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_bind_group(&mut self, h: BindGroupHandle) {
        if self.bgs.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_pipeline(&mut self, h: PipelineHandle) {
        if self.pipelines.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }
    fn destroy_swapchain(&mut self, h: SwapchainHandle) {
        if self.swapchains.free(h.raw().slot, h.raw().generation).is_ok() {
            self.stats.destroys += 1;
        }
    }

    // ── upload ──

    fn write_buffer(&mut self, _handle: BufferHandle, _offset: u64, _data: &[u8]) {
        self.stats.writes += 1;
    }

    fn write_texture(
        &mut self,
        _handle: TextureHandle,
        _region: &TextureRegion,
        _data: &[u8],
        _bytes_per_row: u32,
        _rows_per_image: u32,
    ) {
        self.stats.writes += 1;
    }

    // ── frame lifecycle ──

    fn acquire_next_image(&mut self, swapchain: SwapchainHandle) -> RhiResult<TextureHandle> {
        if !self
            .swapchains
            .is_alive(swapchain.raw().slot, swapchain.raw().generation)
        {
            return Err(RhiError::InvalidHandle("SwapchainHandle"));
        }
        self.stats.acquires += 1;
        let (slot, generation) = self.swapchain_images.alloc();
        Ok(TextureHandle::new(slot, generation))
    }

    fn resize_swapchain(
        &mut self,
        swapchain: SwapchainHandle,
        _width: u32,
        _height: u32,
    ) -> RhiResult<()> {
        if !self
            .swapchains
            .is_alive(swapchain.raw().slot, swapchain.raw().generation)
        {
            return Err(RhiError::InvalidHandle("SwapchainHandle"));
        }
        Ok(())
    }

    fn submit(&mut self, list: CommandList) -> RhiResult<()> {
        self.stats.submits += 1;
        self.submitted_lists.push(list);
        Ok(())
    }

    fn present(&mut self, swapchain: SwapchainHandle) -> RhiResult<()> {
        if !self
            .swapchains
            .is_alive(swapchain.raw().slot, swapchain.raw().generation)
        {
            return Err(RhiError::InvalidHandle("SwapchainHandle"));
        }
        self.stats.presents += 1;
        // Release frame-scope swapchain images.
        self.swapchain_images = Pool::default();
        Ok(())
    }

    fn wait_idle(&mut self) {
        self.stats.waits += 1;
    }
}

// ────────────────────────── Tests ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ocarina_rhi::{
        BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
        BufferUsage, ColorAttachment, ColorTargetState, ColorWrites, FilterMode, FragmentStage,
        GraphicsPipelineDesc, IndexFormat, LoadOp, MemoryLocation, MultisampleState, PipelineKind,
        PresentMode, PrimitiveState, RenderPassDesc, SamplerBindingType, ShaderBytecode,
        ShaderStage, ShaderStageFlags, StoreOp, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsage, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat,
        VertexStage, VertexStepMode, Viewport,
    };

    fn rhi() -> NullRhi {
        NullRhi::new()
    }

    #[test]
    fn backend_identity() {
        let r = rhi();
        assert_eq!(r.backend_name(), "null");
        assert_eq!(r.capabilities().backend_name, "null");
        assert_eq!(r.capabilities().adapter_kind, AdapterKind::Cpu);
        assert!(r.capabilities().features.contains(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn create_and_destroy_buffer() {
        let mut r = rhi();
        let h = r
            .create_buffer(&BufferDesc {
                label: Some("test-vb"),
                size: 1024,
                usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        assert!(h.is_valid());
        assert_eq!(r.alive_resources(), 1);
        r.destroy_buffer(h);
        assert_eq!(r.alive_resources(), 0);
        assert_eq!(r.stats.creates, 1);
        assert_eq!(r.stats.destroys, 1);
    }

    #[test]
    fn generations_advance_on_slot_reuse() {
        let mut r = rhi();
        let a = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::VERTEX,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        r.destroy_buffer(a);
        let b = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::VERTEX,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        // Slot should be reused, but generation must differ.
        assert_eq!(a.raw().slot, b.raw().slot);
        assert_ne!(a.raw().generation, b.raw().generation);
    }

    #[test]
    fn stale_handle_is_detectable() {
        let mut r = rhi();
        let h = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::UNIFORM,
                memory: MemoryLocation::Upload,
            })
            .unwrap();
        r.destroy_buffer(h);
        // Destroying twice should be a no-op (not increment destroys).
        let destroys_before = r.stats.destroys;
        r.destroy_buffer(h);
        assert_eq!(r.stats.destroys, destroys_before);
    }

    #[test]
    fn create_full_pipeline_chain() {
        let mut r = rhi();

        let vs = r
            .create_shader(&ShaderDesc {
                label: Some("vs"),
                stage: ShaderStage::Vertex,
                entry_point: "vs_main",
                bytecode: ShaderBytecode::Wgsl("// vs"),
            })
            .unwrap();
        let fs = r
            .create_shader(&ShaderDesc {
                label: Some("fs"),
                stage: ShaderStage::Fragment,
                entry_point: "fs_main",
                bytecode: ShaderBytecode::Wgsl("// fs"),
            })
            .unwrap();

        let bgl = r
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("globals"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStageFlags::ALL_GRAPHICS,
                        binding_type: BindingType::UniformBuffer {
                            has_dynamic_offset: false,
                            min_binding_size: Some(64),
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
            })
            .unwrap();

        let ubo = r
            .create_buffer(&BufferDesc {
                label: Some("globals"),
                size: 64,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        let tex = r
            .create_texture(&TextureDesc {
                label: Some("albedo"),
                dimension: TextureDimension::D2 {
                    width: 256,
                    height: 256,
                    array_layers: 1,
                },
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                mip_levels: 1,
                sample_count: 1,
            })
            .unwrap();
        let sampler = r
            .create_sampler(&SamplerDesc {
                label: Some("linear"),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                ..SamplerDesc::default()
            })
            .unwrap();

        let bg = r
            .create_bind_group(&BindGroupDesc {
                label: Some("globals"),
                layout: bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: ubo,
                            offset: 0,
                            size: None,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Texture(tex),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(sampler),
                    },
                ],
            })
            .unwrap();

        let attrs = [VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }];
        let buffers = [VertexBufferLayout {
            array_stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: &attrs,
        }];
        let targets = [Some(ColorTargetState {
            format: TextureFormat::Bgra8UnormSrgb,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        })];

        let pipeline = r
            .create_pipeline(&PipelineDesc {
                label: Some("tri"),
                kind: PipelineKind::Graphics(GraphicsPipelineDesc {
                    vertex: VertexStage {
                        shader: vs,
                        buffers: &buffers,
                    },
                    fragment: Some(FragmentStage {
                        shader: fs,
                        targets: &targets,
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                }),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
            .unwrap();

        assert!(pipeline.is_valid());
        assert!(bg.is_valid());

        // Tear everything down.
        r.destroy_pipeline(pipeline);
        r.destroy_bind_group(bg);
        r.destroy_bind_group_layout(bgl);
        r.destroy_sampler(sampler);
        r.destroy_texture(tex);
        r.destroy_buffer(ubo);
        r.destroy_shader(fs);
        r.destroy_shader(vs);
        assert_eq!(r.alive_resources(), 0);
    }

    #[test]
    fn bind_group_rejects_stale_layout() {
        let mut r = rhi();
        let bgl = r
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: &[],
            })
            .unwrap();
        r.destroy_bind_group_layout(bgl);
        let result = r.create_bind_group(&BindGroupDesc {
            label: None,
            layout: bgl,
            entries: &[],
        });
        assert!(matches!(result, Err(RhiError::InvalidHandle(_))));
    }

    #[test]
    fn record_and_submit_render_pass() {
        let mut r = rhi();

        let sc = r
            .create_swapchain(&SwapchainDesc {
                label: Some("main"),
                width: 1280,
                height: 800,
                format: TextureFormat::Bgra8UnormSrgb,
                present_mode: PresentMode::Fifo,
                image_count: 2,
            })
            .unwrap();
        let image = r.acquire_next_image(sc).unwrap();

        let mut list = CommandList::new();
        {
            let attachments = [Some(ColorAttachment {
                view: image,
                resolve_target: None,
                load: LoadOp::Clear([0.1, 0.2, 0.3, 1.0]),
                store: StoreOp::Store,
            })];
            let mut pass = list.begin_render_pass(&RenderPassDesc {
                label: Some("frame"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
            });
            pass.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 1280.0,
                height: 800.0,
                min_depth: 0.0,
                max_depth: 1.0,
            });
            pass.draw(0..3, 0..1);
        }
        assert!(!list.is_empty());

        r.submit(list).unwrap();
        r.present(sc).unwrap();

        assert_eq!(r.stats.submits, 1);
        assert_eq!(r.stats.presents, 1);
        assert_eq!(r.stats.acquires, 1);
        assert_eq!(r.submitted_lists.len(), 1);
    }

    #[test]
    fn record_compute_pass() {
        let mut r = rhi();
        let mut list = CommandList::new();
        {
            let mut pass = list.begin_compute_pass(Some("blur"));
            pass.set_push_constants(0, &[0u8; 16]);
            pass.dispatch(64, 64, 1);
        }
        r.submit(list).unwrap();
        assert_eq!(r.stats.submits, 1);
    }

    #[test]
    fn index_buffer_and_draw_indexed() {
        let mut r = rhi();
        let vb = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::VERTEX,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        let ib = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 128,
                usage: BufferUsage::INDEX,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        let sc = r
            .create_swapchain(&SwapchainDesc {
                label: None,
                width: 100,
                height: 100,
                format: TextureFormat::Bgra8UnormSrgb,
                present_mode: PresentMode::Fifo,
                image_count: 2,
            })
            .unwrap();
        let image = r.acquire_next_image(sc).unwrap();
        let mut list = CommandList::new();
        {
            let attachments = [Some(ColorAttachment {
                view: image,
                resolve_target: None,
                load: LoadOp::Load,
                store: StoreOp::Store,
            })];
            let mut pass = list.begin_render_pass(&RenderPassDesc {
                label: None,
                color_attachments: &attachments,
                depth_stencil_attachment: None,
            });
            pass.set_vertex_buffer(0, vb, 0, None);
            pass.set_index_buffer(ib, 0, None, IndexFormat::Uint32);
            pass.draw_indexed(0..36, 0, 0..1);
        }
        r.submit(list).unwrap();
        r.present(sc).unwrap();
    }

    #[test]
    fn copy_buffer_to_buffer() {
        let mut r = rhi();
        let a = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryLocation::Upload,
            })
            .unwrap();
        let b = r
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_DST | BufferUsage::STORAGE,
                memory: MemoryLocation::Device,
            })
            .unwrap();
        let mut list = CommandList::new();
        list.copy_buffer_to_buffer(a, 0, b, 0, 256);
        r.submit(list).unwrap();
    }

    #[test]
    fn debug_markers_are_recorded() {
        let mut list = CommandList::new();
        list.push_debug_group("frame");
        list.insert_debug_marker("draw-world");
        list.pop_debug_group();
        assert_eq!(list.commands().len(), 3);
    }

    #[test]
    fn limits_are_nonzero() {
        let r = rhi();
        let l = &r.capabilities().limits;
        assert!(l.max_texture_dimension_2d >= 2048);
        assert!(l.max_bind_groups >= 4);
        assert!(l.max_color_attachments >= 4);
    }

    /// The guardrail test: we exercise the full trait surface against the
    /// null backend. If a future change to the `Rhi` trait adds a method
    /// with a backend-specific concept that can't be no-op'd, this test
    /// stops compiling — which is the signal that the trait has leaked a
    /// wgpu-ism.
    #[test]
    fn trait_is_fully_backend_agnostic() {
        let mut r: Box<dyn Rhi> = Box::new(NullRhi::new());
        let _ = r.capabilities();
        let _ = r.backend_name();
        r.wait_idle();
        let sc = r
            .create_swapchain(&SwapchainDesc {
                label: None,
                width: 1,
                height: 1,
                format: TextureFormat::Bgra8UnormSrgb,
                present_mode: PresentMode::Fifo,
                image_count: 2,
            })
            .unwrap();
        let _ = r.acquire_next_image(sc).unwrap();
        r.resize_swapchain(sc, 2, 2).unwrap();
        r.submit(CommandList::new()).unwrap();
        r.present(sc).unwrap();
        r.destroy_swapchain(sc);
    }
}
