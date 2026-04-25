//! wgpu-backed implementation of [`ocarina_rhi::Rhi`].

use std::num::NonZeroU64;

use ocarina_rhi::{
    BindGroupDesc, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutHandle, BindingResource,
    BufferDesc, BufferHandle, Capabilities, CommandList, PipelineDesc, PipelineHandle, PipelineKind,
    PresentMode, RenderPassInfo, Rhi, RhiError, RhiResult, SamplerDesc, SamplerHandle,
    ShaderBytecode, ShaderDesc, ShaderHandle, SwapchainDesc, SwapchainHandle, TextureDesc,
    TextureHandle, TextureRegion,
};

use crate::capability::capabilities_from_adapter;
use crate::command;
use crate::convert as cv;
use crate::destroy::{DestroyQueue, Pending};
use crate::pool::Pool;

pub const BACKEND_NAME: &str = "wgpu";

// ────────── Pool entries ──────────

pub(crate) struct ShaderEntry {
    pub module: wgpu::ShaderModule,
    pub entry_point: String,
}

pub(crate) struct TextureEntry {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[allow(dead_code)] // retained for mip/readback paths in later phases
    pub format: wgpu::TextureFormat,
    #[allow(dead_code)]
    pub size: wgpu::Extent3d,
}

pub(crate) enum PipelineEntry {
    Graphics(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// Swapchain state for standalone rendering to a platform window.
///
/// `current` holds the acquired [`wgpu::SurfaceTexture`] for the active
/// frame. It's consumed in `present` — draining this between `acquire` and
/// `present` would lose the frame. `view_handle` is the borrowed texture
/// handle returned to user code from `acquire_next_image`; it's invalidated
/// (generation-bumped) on `present` so stale handles are detectable.
pub(crate) struct SwapchainEntry {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub current: Option<wgpu::SurfaceTexture>,
    pub view_slot: Option<u32>,
}

pub(crate) struct Pools {
    pub buffers: Pool<wgpu::Buffer>,
    pub textures: Pool<TextureEntry>,
    pub samplers: Pool<wgpu::Sampler>,
    pub shaders: Pool<ShaderEntry>,
    pub bgls: Pool<wgpu::BindGroupLayout>,
    pub bgs: Pool<wgpu::BindGroup>,
    pub pipelines: Pool<PipelineEntry>,
    pub swapchains: Pool<SwapchainEntry>,
}

impl Default for Pools {
    fn default() -> Self {
        Self {
            buffers: Pool::new(),
            textures: Pool::new(),
            samplers: Pool::new(),
            shaders: Pool::new(),
            bgls: Pool::new(),
            bgs: Pool::new(),
            pipelines: Pool::new(),
            swapchains: Pool::new(),
        }
    }
}

// ────────── WgpuRhi ──────────

pub struct WgpuRhi {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// The wgpu `Instance` is only retained when `WgpuRhi` owns the full
    /// context (standalone mode). In editor mode we piggyback on eframe's
    /// context, which doesn't hand us its instance. Kept alive here so the
    /// surface stored in the swapchain pool has a valid backing.
    #[allow(dead_code)]
    pub(crate) instance: Option<wgpu::Instance>,
    pub(crate) pools: Pools,
    destroy: DestroyQueue,
    caps: Capabilities,
}

impl WgpuRhi {
    /// Build a `WgpuRhi` from a pre-existing wgpu context (e.g., the one
    /// eframe creates for egui rendering). This is the editor-integration
    /// entry point.
    ///
    /// Swapchain creation is not supported from `from_existing` because we
    /// don't own a `wgpu::Instance`. Use [`WgpuRhi::new_for_window`] for
    /// standalone mode.
    #[must_use]
    pub fn from_existing(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::Adapter) -> Self {
        let caps = capabilities_from_adapter(adapter, &device);
        log::info!(
            "WgpuRhi initialized: {} ({:?}) [editor / shared context]",
            caps.device_name,
            caps.adapter_kind
        );
        Self {
            device,
            queue,
            instance: None,
            pools: Pools::default(),
            destroy: DestroyQueue::new(),
            caps,
        }
    }

    /// Build a `WgpuRhi` that owns its wgpu context and presents to a
    /// platform window. Blocks on adapter/device request via `pollster`.
    ///
    /// Returns the `WgpuRhi` plus the initial [`SwapchainHandle`]. The
    /// caller keeps ownership of the window (passed by `Arc`) and is
    /// responsible for calling [`Rhi::resize_swapchain`] when the window
    /// resizes.
    pub fn new_for_window<W>(
        window: std::sync::Arc<W>,
        width: u32,
        height: u32,
        present_mode: PresentMode,
    ) -> RhiResult<(Self, SwapchainHandle)>
    where
        W: raw_window_handle::HasWindowHandle
            + raw_window_handle::HasDisplayHandle
            + Send
            + Sync
            + 'static,
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface: wgpu::Surface<'static> = instance
            .create_surface(std::sync::Arc::clone(&window))
            .map_err(|e| RhiError::InitFailed(format!("create_surface: {e}")))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|e| RhiError::AdapterUnavailable(e.to_string()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ocarina.device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| RhiError::DeviceUnavailable(e.to_string()))?;

        // Pick a surface format. Prefer sRGB for "correct" rendering; fall
        // back to whatever the surface reports first.
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: cv::present_mode(present_mode),
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let caps = capabilities_from_adapter(&adapter, &device);
        log::info!(
            "WgpuRhi initialized: {} ({:?}) [standalone / {}x{} / format {:?}]",
            caps.device_name,
            caps.adapter_kind,
            width,
            height,
            surface_format,
        );

        let mut rhi = Self {
            device,
            queue,
            instance: Some(instance),
            pools: Pools::default(),
            destroy: DestroyQueue::new(),
            caps,
        };

        let entry = SwapchainEntry {
            surface,
            config,
            current: None,
            view_slot: None,
        };
        let (slot, generation) = rhi.pools.swapchains.insert(entry);
        Ok((rhi, SwapchainHandle::new(slot, generation)))
    }

    /// Underlying wgpu device (for interop — e.g. registering textures with
    /// egui-wgpu). Using this explicitly ties calling code to the wgpu
    /// backend, which is a deliberate opt-in.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Underlying wgpu queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Look up the underlying `wgpu::TextureView` for a handle. Needed by
    /// egui-wgpu to display our render targets as images.
    #[must_use]
    pub fn texture_view(&self, handle: TextureHandle) -> Option<&wgpu::TextureView> {
        self.pools
            .textures
            .get(handle.raw().slot, handle.raw().generation)
            .map(|t| &t.view)
    }

    /// Look up the underlying `wgpu::Texture` for a handle.
    #[must_use]
    pub fn texture(&self, handle: TextureHandle) -> Option<&wgpu::Texture> {
        self.pools
            .textures
            .get(handle.raw().slot, handle.raw().generation)
            .map(|t| &t.texture)
    }

    // Internal: release resources from a destroy bucket.
    fn release_bucket(&mut self, bucket: &crate::destroy::FrameBucket) {
        for p in &bucket.buffers {
            self.pools.buffers.remove(p.slot, p.generation);
        }
        for p in &bucket.textures {
            self.pools.textures.remove(p.slot, p.generation);
        }
        for p in &bucket.samplers {
            self.pools.samplers.remove(p.slot, p.generation);
        }
        for p in &bucket.shaders {
            self.pools.shaders.remove(p.slot, p.generation);
        }
        for p in &bucket.bgls {
            self.pools.bgls.remove(p.slot, p.generation);
        }
        for p in &bucket.bgs {
            self.pools.bgs.remove(p.slot, p.generation);
        }
        for p in &bucket.pipelines {
            self.pools.pipelines.remove(p.slot, p.generation);
        }
        for p in &bucket.swapchains {
            self.pools.swapchains.remove(p.slot, p.generation);
        }
    }
}

// ────────── Rhi impl ──────────

impl Rhi for WgpuRhi {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn backend_name(&self) -> &'static str {
        BACKEND_NAME
    }

    // ── create ──

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> RhiResult<BufferHandle> {
        let usage = cv::buffer_usages(desc.usage, desc.memory);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label,
            size: desc.size,
            usage,
            mapped_at_creation: false,
        });
        let (slot, generation) = self.pools.buffers.insert(buffer);
        Ok(BufferHandle::new(slot, generation))
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> RhiResult<TextureHandle> {
        let (dim, size) = cv::texture_dimension(desc.dimension);
        let format = cv::texture_format(desc.format);
        let usage = cv::texture_usages(desc.usage);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label,
            size,
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: dim,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: desc.label,
            ..Default::default()
        });
        let entry = TextureEntry {
            texture,
            view,
            format,
            size,
        };
        let (slot, generation) = self.pools.textures.insert(entry);
        Ok(TextureHandle::new(slot, generation))
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> RhiResult<SamplerHandle> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label,
            address_mode_u: cv::address_mode(desc.address_mode_u),
            address_mode_v: cv::address_mode(desc.address_mode_v),
            address_mode_w: cv::address_mode(desc.address_mode_w),
            mag_filter: cv::filter_mode(desc.mag_filter),
            min_filter: cv::filter_mode(desc.min_filter),
            mipmap_filter: cv::mipmap_filter(desc.mipmap_filter),
            lod_min_clamp: desc.lod_min,
            lod_max_clamp: desc.lod_max,
            compare: desc.compare.map(cv::compare_function),
            anisotropy_clamp: desc.anisotropy_clamp.max(1),
            border_color: desc.border_color.map(cv::border_color),
        });
        let (slot, generation) = self.pools.samplers.insert(sampler);
        Ok(SamplerHandle::new(slot, generation))
    }

    fn create_shader(&mut self, desc: &ShaderDesc<'_>) -> RhiResult<ShaderHandle> {
        let source = match desc.bytecode {
            ShaderBytecode::Wgsl(src) => wgpu::ShaderSource::Wgsl(src.into()),
            // SPIR-V consumed directly (HLSL is compiled to SPIR-V upstream
            // via DXC). Requires the `spirv` feature on wgpu.
            ShaderBytecode::Spirv(words) => wgpu::ShaderSource::SpirV(words.into()),
            ShaderBytecode::Msl(_) | ShaderBytecode::Dxil(_) => {
                return Err(RhiError::FeatureUnsupported(
                    "wgpu backend consumes WGSL or SPIR-V; MSL/DXIL require a different backend",
                ));
            }
        };
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label,
                source,
            });
        let entry = ShaderEntry {
            module,
            entry_point: desc.entry_point.to_owned(),
        };
        let (slot, generation) = self.pools.shaders.insert(entry);
        Ok(ShaderHandle::new(slot, generation))
    }

    fn create_bind_group_layout(
        &mut self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> RhiResult<BindGroupLayoutHandle> {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = desc
            .entries
            .iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: cv::shader_stages(e.visibility),
                ty: cv::binding_type(e.binding_type),
                count: e.count.and_then(std::num::NonZeroU32::new),
            })
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: desc.label,
                entries: &entries,
            });
        let (slot, generation) = self.pools.bgls.insert(layout);
        Ok(BindGroupLayoutHandle::new(slot, generation))
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> RhiResult<BindGroupHandle> {
        let layout = self
            .pools
            .bgls
            .get(desc.layout.raw().slot, desc.layout.raw().generation)
            .ok_or(RhiError::InvalidHandle("BindGroupLayoutHandle"))?;

        // Collect entries. BindingResource borrows from pool entries so we
        // need to keep them alive for the `create_bind_group` call.
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(desc.entries.len());
        for e in desc.entries {
            let resource = match e.resource {
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    let buf = self
                        .pools
                        .buffers
                        .get(buffer.raw().slot, buffer.raw().generation)
                        .ok_or(RhiError::InvalidHandle("BufferHandle"))?;
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset,
                        size: size.and_then(NonZeroU64::new),
                    })
                }
                BindingResource::Texture(t) => {
                    let tex = self
                        .pools
                        .textures
                        .get(t.raw().slot, t.raw().generation)
                        .ok_or(RhiError::InvalidHandle("TextureHandle"))?;
                    wgpu::BindingResource::TextureView(&tex.view)
                }
                BindingResource::Sampler(s) => {
                    let sampler = self
                        .pools
                        .samplers
                        .get(s.raw().slot, s.raw().generation)
                        .ok_or(RhiError::InvalidHandle("SamplerHandle"))?;
                    wgpu::BindingResource::Sampler(sampler)
                }
            };
            entries.push(wgpu::BindGroupEntry {
                binding: e.binding,
                resource,
            });
        }

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: desc.label,
            layout,
            entries: &entries,
        });
        let (slot, generation) = self.pools.bgs.insert(bg);
        Ok(BindGroupHandle::new(slot, generation))
    }

    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> RhiResult<PipelineHandle> {
        // Resolve bind group layouts.
        let bg_layouts: Vec<Option<&wgpu::BindGroupLayout>> = desc
            .bind_group_layouts
            .iter()
            .map(|h| {
                self.pools
                    .bgls
                    .get(h.raw().slot, h.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BindGroupLayoutHandle"))
                    .map(Some)
            })
            .collect::<Result<_, _>>()?;

        // wgpu 29 represents push constants as a single byte-sized "immediate"
        // range shared by all stages. Compute the union size from our per-stage
        // ranges; backends with per-stage granularity (VK/D3D12) can honor the
        // original ranges when they land.
        let immediate_size: u32 = desc
            .push_constant_ranges
            .iter()
            .map(|r| r.offset + r.size)
            .max()
            .unwrap_or(0);

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label,
                bind_group_layouts: &bg_layouts,
                immediate_size,
            });

        match desc.kind {
            PipelineKind::Graphics(g) => {
                let vs_entry = self
                    .pools
                    .shaders
                    .get(g.vertex.shader.raw().slot, g.vertex.shader.raw().generation)
                    .ok_or(RhiError::InvalidHandle("ShaderHandle (vertex)"))?;

                // Gather vertex buffer layouts. Attributes borrow the slice,
                // so we stage buffers + attributes in a local owned vec
                // whose lifetime encloses the create call.
                let mut wgpu_attrs: Vec<Vec<wgpu::VertexAttribute>> =
                    Vec::with_capacity(g.vertex.buffers.len());
                for vbl in g.vertex.buffers {
                    let attrs = vbl
                        .attributes
                        .iter()
                        .map(|a| wgpu::VertexAttribute {
                            format: cv::vertex_format(a.format),
                            offset: a.offset,
                            shader_location: a.shader_location,
                        })
                        .collect();
                    wgpu_attrs.push(attrs);
                }
                let buffer_layouts: Vec<wgpu::VertexBufferLayout<'_>> = g
                    .vertex
                    .buffers
                    .iter()
                    .zip(wgpu_attrs.iter())
                    .map(|(vbl, attrs)| wgpu::VertexBufferLayout {
                        array_stride: vbl.array_stride,
                        step_mode: cv::step_mode(vbl.step_mode),
                        attributes: attrs,
                    })
                    .collect();

                let fragment_targets: Vec<Option<wgpu::ColorTargetState>> = g
                    .fragment
                    .as_ref()
                    .map(|f| {
                        f.targets
                            .iter()
                            .map(|target| {
                                target.map(|t| wgpu::ColorTargetState {
                                    format: cv::texture_format(t.format),
                                    blend: t.blend.map(|b| wgpu::BlendState {
                                        color: wgpu::BlendComponent {
                                            src_factor: cv::blend_factor(b.color.src_factor),
                                            dst_factor: cv::blend_factor(b.color.dst_factor),
                                            operation: cv::blend_operation(b.color.operation),
                                        },
                                        alpha: wgpu::BlendComponent {
                                            src_factor: cv::blend_factor(b.alpha.src_factor),
                                            dst_factor: cv::blend_factor(b.alpha.dst_factor),
                                            operation: cv::blend_operation(b.alpha.operation),
                                        },
                                    }),
                                    write_mask: cv::color_writes(t.write_mask),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let fs_entry = if let Some(fs) = g.fragment.as_ref() {
                    Some(
                        self.pools
                            .shaders
                            .get(fs.shader.raw().slot, fs.shader.raw().generation)
                            .ok_or(RhiError::InvalidHandle("ShaderHandle (fragment)"))?,
                    )
                } else {
                    None
                };

                let fragment = if let (Some(fs), Some(fs_entry)) = (g.fragment.as_ref(), fs_entry) {
                    let _ = fs;
                    Some(wgpu::FragmentState {
                        module: &fs_entry.module,
                        entry_point: Some(&fs_entry.entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &fragment_targets,
                    })
                } else {
                    None
                };

                let primitive = wgpu::PrimitiveState {
                    topology: cv::primitive_topology(g.primitive.topology),
                    strip_index_format: g.primitive.strip_index_format.map(cv::index_format),
                    front_face: cv::front_face(g.primitive.front_face),
                    cull_mode: g.primitive.cull_mode.map(cv::cull_face),
                    polygon_mode: cv::polygon_mode(g.primitive.polygon_mode),
                    unclipped_depth: g.primitive.unclipped_depth,
                    conservative: g.primitive.conservative,
                };

                let depth_stencil = g.depth_stencil.map(|d| wgpu::DepthStencilState {
                    format: cv::texture_format(d.format),
                    depth_write_enabled: Some(d.depth_write_enabled),
                    depth_compare: Some(cv::compare_function(d.depth_compare)),
                    stencil: wgpu::StencilState {
                        front: wgpu::StencilFaceState {
                            compare: d
                                .stencil_front
                                .compare
                                .map(cv::compare_function)
                                .unwrap_or(wgpu::CompareFunction::Always),
                            fail_op: cv::stencil_op(d.stencil_front.fail_op),
                            depth_fail_op: cv::stencil_op(d.stencil_front.depth_fail_op),
                            pass_op: cv::stencil_op(d.stencil_front.pass_op),
                        },
                        back: wgpu::StencilFaceState {
                            compare: d
                                .stencil_back
                                .compare
                                .map(cv::compare_function)
                                .unwrap_or(wgpu::CompareFunction::Always),
                            fail_op: cv::stencil_op(d.stencil_back.fail_op),
                            depth_fail_op: cv::stencil_op(d.stencil_back.depth_fail_op),
                            pass_op: cv::stencil_op(d.stencil_back.pass_op),
                        },
                        read_mask: d.stencil_read_mask,
                        write_mask: d.stencil_write_mask,
                    },
                    bias: wgpu::DepthBiasState {
                        constant: d.bias.constant,
                        slope_scale: d.bias.slope_scale,
                        clamp: d.bias.clamp,
                    },
                });

                let multisample = wgpu::MultisampleState {
                    count: g.multisample.count,
                    mask: g.multisample.mask,
                    alpha_to_coverage_enabled: g.multisample.alpha_to_coverage_enabled,
                };

                let pipeline = self.device.create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label: desc.label,
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &vs_entry.module,
                            entry_point: Some(&vs_entry.entry_point),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers: &buffer_layouts,
                        },
                        primitive,
                        depth_stencil,
                        multisample,
                        fragment,
                        multiview_mask: None,
                        cache: None,
                    },
                );
                let (slot, generation) =
                    self.pools.pipelines.insert(PipelineEntry::Graphics(pipeline));
                Ok(PipelineHandle::new(slot, generation))
            }
            PipelineKind::Compute(c) => {
                let cs_entry = self
                    .pools
                    .shaders
                    .get(c.shader.raw().slot, c.shader.raw().generation)
                    .ok_or(RhiError::InvalidHandle("ShaderHandle (compute)"))?;
                let pipeline =
                    self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: desc.label,
                        layout: Some(&layout),
                        module: &cs_entry.module,
                        entry_point: Some(&cs_entry.entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });
                let (slot, generation) =
                    self.pools.pipelines.insert(PipelineEntry::Compute(pipeline));
                Ok(PipelineHandle::new(slot, generation))
            }
        }
    }

    fn create_swapchain(&mut self, _desc: &SwapchainDesc<'_>) -> RhiResult<SwapchainHandle> {
        // The generic `SwapchainDesc` deliberately doesn't carry a window
        // handle — swapchain creation is inherently platform-specific, so
        // it goes through [`WgpuRhi::new_for_window`] instead.
        Err(RhiError::FeatureUnsupported(
            "WgpuRhi::create_swapchain — call WgpuRhi::new_for_window() instead",
        ))
    }

    // ── destroy (deferred) ──

    fn destroy_buffer(&mut self, h: BufferHandle) {
        self.destroy.push_buffer(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_texture(&mut self, h: TextureHandle) {
        self.destroy.push_texture(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_sampler(&mut self, h: SamplerHandle) {
        self.destroy.push_sampler(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_shader(&mut self, h: ShaderHandle) {
        self.destroy.push_shader(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_bind_group_layout(&mut self, h: BindGroupLayoutHandle) {
        self.destroy.push_bgl(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_bind_group(&mut self, h: BindGroupHandle) {
        self.destroy.push_bg(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_pipeline(&mut self, h: PipelineHandle) {
        self.destroy.push_pipeline(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }
    fn destroy_swapchain(&mut self, h: SwapchainHandle) {
        self.destroy.push_swapchain(Pending {
            slot: h.raw().slot,
            generation: h.raw().generation,
        });
    }

    // ── upload ──

    fn write_buffer(&mut self, handle: BufferHandle, offset: u64, data: &[u8]) {
        if let Some(buf) = self
            .pools
            .buffers
            .get(handle.raw().slot, handle.raw().generation)
        {
            self.queue.write_buffer(buf, offset, data);
        } else {
            log::warn!("write_buffer: stale handle {handle:?}");
        }
    }

    fn write_texture(
        &mut self,
        handle: TextureHandle,
        region: &TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
    ) {
        let Some(tex) = self
            .pools
            .textures
            .get(handle.raw().slot, handle.raw().generation)
        else {
            log::warn!("write_texture: stale handle {handle:?}");
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: region.mip_level,
                origin: wgpu::Origin3d {
                    x: region.origin[0],
                    y: region.origin[1],
                    z: region.origin[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width: region.extent[0],
                height: region.extent[1],
                depth_or_array_layers: region.extent[2],
            },
        );
    }

    // ── frame lifecycle ──

    fn acquire_next_image(&mut self, swapchain: SwapchainHandle) -> RhiResult<TextureHandle> {
        let sc_slot = swapchain.raw().slot;
        let sc_gen = swapchain.raw().generation;
        let sc = self
            .pools
            .swapchains
            .get_mut(sc_slot, sc_gen)
            .ok_or(RhiError::InvalidHandle("SwapchainHandle"))?;

        if sc.current.is_some() {
            return Err(RhiError::Backend(
                "acquire_next_image called twice without intervening present".into(),
            ));
        }

        let surface_tex = match sc.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated => {
                // Surface is out of date — caller should resize + retry.
                // Reconfigure eagerly so the next acquire works.
                sc.surface.configure(&self.device, &sc.config);
                return Err(RhiError::SurfaceLost);
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RhiError::SurfaceLost);
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Err(RhiError::SwapchainAcquireFailed(
                    "get_current_texture timed out".into(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                // Window minimized/hidden — treat as a transient skip.
                return Err(RhiError::SwapchainAcquireFailed("surface occluded".into()));
            }
        };

        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let format = surface_tex.texture.format();
        let size = wgpu::Extent3d {
            width: sc.config.width,
            height: sc.config.height,
            depth_or_array_layers: 1,
        };

        // Store the SurfaceTexture for the present step.
        let width = sc.config.width;
        let height = sc.config.height;
        // Drop the borrow so we can insert into textures pool.
        let _ = (width, height);
        let texture_clone = surface_tex.texture.clone();
        sc.current = Some(surface_tex);

        // Expose the frame's backbuffer as a regular TextureHandle so user
        // code uses the same CommandList path as offscreen targets.
        let (tex_slot, tex_gen) = self.pools.textures.insert(TextureEntry {
            texture: texture_clone,
            view,
            format,
            size,
        });
        // Re-borrow swapchain entry to record the view slot for cleanup
        // on present.
        if let Some(sc) = self.pools.swapchains.get_mut(sc_slot, sc_gen) {
            sc.view_slot = Some(tex_slot);
        }

        Ok(TextureHandle::new(tex_slot, tex_gen))
    }

    fn resize_swapchain(
        &mut self,
        swapchain: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> RhiResult<()> {
        let sc = self
            .pools
            .swapchains
            .get_mut(swapchain.raw().slot, swapchain.raw().generation)
            .ok_or(RhiError::InvalidHandle("SwapchainHandle"))?;
        if width == 0 || height == 0 {
            return Ok(()); // minimized window — do nothing
        }
        sc.config.width = width;
        sc.config.height = height;
        sc.surface.configure(&self.device, &sc.config);
        Ok(())
    }

    fn submit(&mut self, list: CommandList) -> RhiResult<()> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ocarina.frame"),
            });
        command::interpret(&mut encoder, &self.pools, list.commands())?;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn present(&mut self, swapchain: SwapchainHandle) -> RhiResult<()> {
        // In editor mode the swapchain is SwapchainHandle::INVALID (no owned
        // surface). The call still drives the per-frame destroy rotation.
        if swapchain.is_valid() {
            let (view_slot, surface_tex) = {
                let sc = self
                    .pools
                    .swapchains
                    .get_mut(swapchain.raw().slot, swapchain.raw().generation)
                    .ok_or(RhiError::InvalidHandle("SwapchainHandle"))?;
                (sc.view_slot.take(), sc.current.take())
            };

            // Retire the TextureHandle handed out from acquire so stale
            // references get the InvalidHandle error next frame. We don't
            // drop the texture until after `present()`; wgpu's
            // `SurfaceTexture::present` does that.
            if let Some(slot) = view_slot {
                self.pools.textures.remove_by_slot(slot);
            }
            if let Some(tex) = surface_tex {
                tex.present();
            }
        }

        if let Some(bucket) = self.destroy.rotate() {
            self.release_bucket(&bucket);
        }
        Ok(())
    }

    fn wait_idle(&mut self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wgpu device poll failed");
        for bucket in self.destroy.drain_all() {
            self.release_bucket(&bucket);
        }
    }
}

// Helper: treat a dummy `RenderPassInfo` construction path as part of this
// module so it compiles without warnings about unused imports.
#[allow(dead_code)]
fn _rp_info_type_anchor(_: &RenderPassInfo) {}
