//! wgpu-backed implementation of [`ocarina_rhi::Rhi`].

use std::num::NonZeroU64;

use ocarina_rhi::{
    AsyncPipelineCompletion, AsyncPipelineId, BindGroupDesc, BindGroupHandle,
    BindGroupLayoutDesc, BindGroupLayoutHandle, BindingResource, BufferDesc, BufferHandle,
    Capabilities, CommandList, PipelineDesc, PipelineHandle, PipelineKind, PresentMode,
    RenderPassInfo, Rhi, RhiError, RhiResult, SamplerDesc, SamplerHandle, ShaderBytecode,
    ShaderDesc, ShaderHandle, SwapchainDesc, SwapchainHandle, TextureDesc, TextureHandle,
    TextureRegion,
};

use crate::capability::capabilities_from_adapter;
use crate::command;
use crate::convert as cv;
use crate::destroy::{DestroyQueue, Pending};
use crate::pipeline_compiler;
use crate::pool::Pool;

pub const BACKEND_NAME: &str = "wgpu";

/// Replace wgpu's default uncaptured-error handler (which panics the
/// process) with a logging one. Without this a malformed user shader
/// or pipeline brings the whole editor down. Errors that escape the
/// targeted scopes in `create_shader` / `create_pipeline` still get
/// logged so they're not silently swallowed.
fn install_non_panicking_error_handler(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|err: wgpu::Error| {
        log::error!("wgpu uncaptured error: {err}");
    }));
}

/// Run `op` inside a wgpu validation error scope. Returns the
/// captured error if validation failed, `None` otherwise. On native
/// the future resolves synchronously after the device has processed
/// the queued work, so the brief block is fine.
fn capture_validation<F, T>(device: &wgpu::Device, op: F) -> (T, Option<wgpu::Error>)
where
    F: FnOnce() -> T,
{
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = op();
    let err = pollster::block_on(device.pop_error_scope());
    (value, err)
}

// ────────── Pool entries ──────────

pub(crate) struct ShaderEntry {
    pub module: wgpu::ShaderModule,
    pub entry_point: String,
}

pub(crate) struct TextureEntry {
    /// Owning texture for entries the RHI allocated (`create_texture`,
    /// swapchain backbuffer). `None` for external entries the editor
    /// registered via [`WgpuRhi::register_external_view`] -- in that
    /// case the host (Floem) owns the parent texture and only hands
    /// us the view. Buffer / texture copy commands need the parent;
    /// they error if invoked against an external entry.
    pub texture: Option<wgpu::Texture>,
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

// Field order matters: Rust drops fields in declaration order. Pool
// resources (Buffer, Texture, etc.) hold internal references back to
// the Device. If `device` were declared above `pools`, the device
// handle would drop first; in editor mode where Floem also holds a
// clone the underlying device survives, but on standalone shutdown
// we have seen wgpu's drop validation panic on stale Buffer handles
// — and any panic from a destructor during process exit upgrades
// to `panic_in_cleanup → abort`. Declaring pools/destroy/queue/device
// in that order, plus the explicit `Drop` below, guarantees pool
// resources tear down first.
pub struct WgpuRhi {
    pub(crate) pools: Pools,
    destroy: DestroyQueue,
    /// Worker thread + channels powering [`Rhi::create_pipeline_async`].
    /// Holds a `wgpu::Device` clone so it can call
    /// `device.create_render_pipeline` off the render thread; results
    /// are drained by `poll_async_pipelines` and adopted into
    /// [`Self::pools.pipelines`] like any synchronous pipeline.
    pipeline_compiler: pipeline_compiler::PipelineCompiler,
    /// Monotonic counter for [`AsyncPipelineId`]s. Bumped per submit;
    /// never reused so callers can compare ids by equality across
    /// frames safely.
    next_async_pipeline_id: u64,
    /// Async-pipeline failures that didn't reach the worker (invalid
    /// handles, missing fragment stage, etc.). Drained alongside
    /// worker-produced results inside `poll_async_pipelines` so the
    /// caller sees a uniform stream.
    async_pipeline_local_failures: Vec<AsyncPipelineCompletion>,
    pub(crate) queue: wgpu::Queue,
    pub(crate) device: wgpu::Device,
    /// The wgpu `Instance` is only retained when `WgpuRhi` owns the full
    /// context (standalone mode). In editor mode we piggyback on the
    /// host's context, which doesn't hand us its instance. Kept alive
    /// here so the surface stored in the swapchain pool has a valid
    /// backing.
    #[allow(dead_code)]
    pub(crate) instance: Option<wgpu::Instance>,
    caps: Capabilities,
}

impl Drop for WgpuRhi {
    fn drop(&mut self) {
        // Belt-and-suspenders against drop ordering: explicitly clear
        // the pools and the destroy queue before the implicit field
        // drops walk over `device`/`queue`/`instance`. wgpu resource
        // handles drop cleanly here because the device clone is still
        // alive in `self.device` at this point.
        self.pools = Pools::default();
        self.destroy = DestroyQueue::new();
    }
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
        install_non_panicking_error_handler(&device);
        log::info!(
            "WgpuRhi initialized: {} ({:?}) [editor / shared context]",
            caps.device_name,
            caps.adapter_kind
        );
        let pipeline_compiler =
            pipeline_compiler::PipelineCompiler::spawn(device.clone());
        Self {
            device,
            queue,
            instance: None,
            pools: Pools::default(),
            destroy: DestroyQueue::new(),
            pipeline_compiler,
            next_async_pipeline_id: 0,
            async_pipeline_local_failures: Vec::new(),
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
        let instance = wgpu::Instance::default();
        let surface: wgpu::Surface<'static> = instance
            .create_surface(std::sync::Arc::clone(&window))
            .map_err(|e| RhiError::InitFailed(format!("create_surface: {e}")))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| RhiError::AdapterUnavailable("no adapter".into()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ocarina.device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| RhiError::DeviceUnavailable(e.to_string()))?;

        // Pick a NON-sRGB surface format. Our scene shader does a
        // manual `linear_to_srgb` encode at the end of the fragment
        // (so the editor's viewport, which uses Floem's vger and
        // pins to non-sRGB targets, displays correctly). Pairing
        // that with an sRGB-encoded swapchain here would
        // double-correct: hardware would re-encode our already-
        // encoded output and the runtime would render too bright.
        // Match the editor's format choice instead so a single
        // shader output path serves both hosts.
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
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
        install_non_panicking_error_handler(&device);
        log::info!(
            "WgpuRhi initialized: {} ({:?}) [standalone / {}x{} / format {:?}]",
            caps.device_name,
            caps.adapter_kind,
            width,
            height,
            surface_format,
        );

        let pipeline_compiler =
            pipeline_compiler::PipelineCompiler::spawn(device.clone());
        let mut rhi = Self {
            device,
            queue,
            instance: Some(instance),
            pools: Pools::default(),
            destroy: DestroyQueue::new(),
            pipeline_compiler,
            next_async_pipeline_id: 0,
            async_pipeline_local_failures: Vec::new(),
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

    /// Look up the underlying `wgpu::Texture` for a handle. Returns
    /// `None` for external entries (Floem-borrowed views) since the
    /// host owns the parent texture in that case.
    #[must_use]
    pub fn texture(&self, handle: TextureHandle) -> Option<&wgpu::Texture> {
        self.pools
            .textures
            .get(handle.raw().slot, handle.raw().generation)
            .and_then(|t| t.texture.as_ref())
    }

    /// Look up the underlying `wgpu::Buffer` for a handle.
    ///
    /// Needed when host code needs to drive a render pass against an
    /// externally-owned target (e.g. the editor's swapchain backbuffer
    /// surfaced through `floem::Renderer::submit_wgpu_pass`) and so
    /// can't go through `submit(CommandList)` — `CommandList` only
    /// references RHI-owned `TextureHandle`s for its color attachments.
    #[must_use]
    pub fn buffer(&self, handle: BufferHandle) -> Option<&wgpu::Buffer> {
        self.pools
            .buffers
            .get(handle.raw().slot, handle.raw().generation)
    }

    /// Look up the underlying `wgpu::BindGroup` for a handle.
    /// See [`buffer`](Self::buffer) for the use-case rationale.
    #[must_use]
    pub fn bind_group(&self, handle: BindGroupHandle) -> Option<&wgpu::BindGroup> {
        self.pools
            .bgs
            .get(handle.raw().slot, handle.raw().generation)
    }

    /// Register a borrowed `wgpu::TextureView` (typically the editor
    /// host's swapchain target handed to us by Floem) as a regular
    /// [`TextureHandle`]. The handle plugs into the same CommandList
    /// path as RHI-allocated textures, letting `ocarina-render`'s
    /// unified renderer record into the host surface without knowing
    /// it's external.
    ///
    /// `view` is cloned cheaply (wgpu's `TextureView` is internally
    /// reference-counted). The host keeps the parent `wgpu::Texture`
    /// alive for the duration the handle is in use; copy ops through
    /// this handle error since we don't own the parent.
    ///
    /// Pair with [`unregister_external_view`] before the parent
    /// texture goes out of scope (typically end-of-frame, before
    /// Floem presents).
    pub fn register_external_view(
        &mut self,
        view: wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> TextureHandle {
        let entry = TextureEntry {
            texture: None,
            view,
            format,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        };
        let (slot, generation) = self.pools.textures.insert(entry);
        TextureHandle::new(slot, generation)
    }

    /// Drop an external view registered via
    /// [`register_external_view`]. Invalidates the handle's
    /// generation so any stale reference fails the next pool lookup.
    /// Must be called before the host releases the parent texture
    /// (typically each frame).
    pub fn unregister_external_view(&mut self, handle: TextureHandle) {
        self.pools
            .textures
            .remove(handle.raw().slot, handle.raw().generation);
    }


    /// The color format of `swapchain`'s backbuffer. Pipelines that
    /// will draw into this swapchain must declare this format as
    /// their color target. Returns `None` for an invalid handle.
    #[must_use]
    pub fn swapchain_format(
        &self,
        handle: SwapchainHandle,
    ) -> Option<ocarina_rhi::TextureFormat> {
        let raw = handle.raw();
        self.pools
            .swapchains
            .get(raw.slot, raw.generation)
            .and_then(|sc| crate::convert::texture_format_from_wgpu(sc.config.format))
    }

    /// Look up the underlying `wgpu::RenderPipeline` for a handle.
    /// Returns `None` if the handle is invalid or refers to a compute
    /// pipeline. See [`buffer`](Self::buffer) for the use-case rationale.
    #[must_use]
    pub fn graphics_pipeline(
        &self,
        handle: PipelineHandle,
    ) -> Option<&wgpu::RenderPipeline> {
        match self
            .pools
            .pipelines
            .get(handle.raw().slot, handle.raw().generation)?
        {
            PipelineEntry::Graphics(p) => Some(p),
            PipelineEntry::Compute(_) => None,
        }
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
            texture: Some(texture),
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
        // Wrap in a validation error scope so a malformed WGSL turns
        // into a returned `Err` instead of an uncaptured device error
        // (which the panic-by-default global handler would crash on).
        let (module, err) = capture_validation(&self.device, || {
            self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label,
                source,
            })
        });
        if let Some(err) = err {
            return Err(RhiError::Backend(format!(
                "shader compile failed for `{}`: {err}",
                desc.label.unwrap_or("<unlabelled>")
            )));
        }
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
        let bg_layouts: Vec<&wgpu::BindGroupLayout> = desc
            .bind_group_layouts
            .iter()
            .map(|h| {
                self.pools
                    .bgls
                    .get(h.raw().slot, h.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BindGroupLayoutHandle"))
            })
            .collect::<Result<_, _>>()?;

        // wgpu 27 takes per-stage push constant ranges directly. Future
        // native backends (VK/D3D12) honour the same per-stage granularity.
        let push_constant_ranges: Vec<wgpu::PushConstantRange> = desc
            .push_constant_ranges
            .iter()
            .map(|r| wgpu::PushConstantRange {
                stages: cv::shader_stages(r.stages),
                range: r.offset..(r.offset + r.size),
            })
            .collect();

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label,
                bind_group_layouts: &bg_layouts,
                push_constant_ranges: &push_constant_ranges,
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
                    depth_write_enabled: d.depth_write_enabled,
                    depth_compare: cv::compare_function(d.depth_compare),
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

                // Capture validation errors so a shader compile that
                // got a placeholder module (or a layout mismatch) turns
                // into Err instead of panicking the device.
                let (pipeline, err) = capture_validation(&self.device, || {
                    self.device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: desc.label,
                            layout: Some(&layout),
                            vertex: wgpu::VertexState {
                                module: &vs_entry.module,
                                entry_point: Some(&vs_entry.entry_point),
                                compilation_options:
                                    wgpu::PipelineCompilationOptions::default(),
                                buffers: &buffer_layouts,
                            },
                            primitive,
                            depth_stencil,
                            multisample,
                            fragment,
                            multiview: None,
                            cache: None,
                        },
                    )
                });
                if let Some(err) = err {
                    return Err(RhiError::Backend(format!(
                        "render pipeline create failed for `{}`: {err}",
                        desc.label.unwrap_or("<unlabelled>")
                    )));
                }
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
                let (pipeline, err) = capture_validation(&self.device, || {
                    self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: desc.label,
                        layout: Some(&layout),
                        module: &cs_entry.module,
                        entry_point: Some(&cs_entry.entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
                });
                if let Some(err) = err {
                    return Err(RhiError::Backend(format!(
                        "compute pipeline create failed for `{}`: {err}",
                        desc.label.unwrap_or("<unlabelled>")
                    )));
                }
                let (slot, generation) =
                    self.pools.pipelines.insert(PipelineEntry::Compute(pipeline));
                Ok(PipelineHandle::new(slot, generation))
            }
        }
    }

    fn create_pipeline_async(
        &mut self,
        desc: &PipelineDesc<'_>,
    ) -> AsyncPipelineId {
        let id = AsyncPipelineId(self.next_async_pipeline_id);
        self.next_async_pipeline_id += 1;

        // Async path is graphics-only for now; compute pipelines are
        // rare in the editor and don't have the recompile-on-save
        // workflow that motivated this code. Surface a local
        // failure via poll_async_pipelines so callers see the same
        // shape as a regular validation failure.
        let g = match desc.kind {
            PipelineKind::Graphics(g) => g,
            PipelineKind::Compute(_) => {
                self.async_pipeline_local_failures.push(
                    AsyncPipelineCompletion {
                        id,
                        result: Err(RhiError::Backend(
                            "create_pipeline_async: compute pipelines are not \
                             supported on the async path"
                                .into(),
                        )),
                    },
                );
                return id;
            }
        };

        // Resolve handles into wgpu types we can move across thread.
        // Failures (invalid handles) post a synthetic Err result via
        // the `pending_async_pipelines_local` queue rather than a
        // worker round-trip.
        let bgls: Vec<wgpu::BindGroupLayout> = match desc
            .bind_group_layouts
            .iter()
            .map(|h| {
                self.pools
                    .bgls
                    .get(h.raw().slot, h.raw().generation)
                    .cloned()
                    .ok_or(RhiError::InvalidHandle("BindGroupLayoutHandle"))
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(e) => {
                self.async_pipeline_local_failures.push(
                    AsyncPipelineCompletion {
                        id,
                        result: Err(e),
                    },
                );
                return id;
            }
        };

        let push_constant_ranges: Vec<wgpu::PushConstantRange> = desc
            .push_constant_ranges
            .iter()
            .map(|r| wgpu::PushConstantRange {
                stages: cv::shader_stages(r.stages),
                range: r.offset..(r.offset + r.size),
            })
            .collect();

        let bgl_refs: Vec<&wgpu::BindGroupLayout> = bgls.iter().collect();
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label,
                bind_group_layouts: &bgl_refs,
                push_constant_ranges: &push_constant_ranges,
            });

        let vs_entry = match self
            .pools
            .shaders
            .get(g.vertex.shader.raw().slot, g.vertex.shader.raw().generation)
        {
            Some(s) => s,
            None => {
                self.async_pipeline_local_failures.push(
                    AsyncPipelineCompletion {
                        id,
                        result: Err(RhiError::InvalidHandle("ShaderHandle (vertex)")),
                    },
                );
                return id;
            }
        };
        let vs_module = vs_entry.module.clone();
        let vs_entry_name = vs_entry.entry_point.clone();

        let (fs_module, fs_entry_name) = match g.fragment.as_ref() {
            Some(fs) => match self
                .pools
                .shaders
                .get(fs.shader.raw().slot, fs.shader.raw().generation)
            {
                Some(e) => (e.module.clone(), e.entry_point.clone()),
                None => {
                    self.async_pipeline_local_failures.push(
                        AsyncPipelineCompletion {
                            id,
                            result: Err(RhiError::InvalidHandle("ShaderHandle (fragment)")),
                        },
                    );
                    return id;
                }
            },
            None => {
                // Async path requires a fragment stage; rasterization-
                // only graphics pipelines don't fit the usual user-
                // shader workflow that drives this API.
                self.async_pipeline_local_failures.push(
                    AsyncPipelineCompletion {
                        id,
                        result: Err(RhiError::Backend(
                            "create_pipeline_async requires a fragment stage".into(),
                        )),
                    },
                );
                return id;
            }
        };

        // Vertex buffer layouts — own the attribute vec so the
        // request can move across threads.
        let vertex_buffers: Vec<pipeline_compiler::OwnedVertexBufferLayout> = g
            .vertex
            .buffers
            .iter()
            .map(|vbl| pipeline_compiler::OwnedVertexBufferLayout {
                array_stride: vbl.array_stride,
                step_mode: cv::step_mode(vbl.step_mode),
                attributes: vbl
                    .attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: cv::vertex_format(a.format),
                        offset: a.offset,
                        shader_location: a.shader_location,
                    })
                    .collect(),
            })
            .collect();

        // Fragment color targets — same shape as the sync path,
        // copied into an owned `Vec<Option<_>>` so slot positions
        // are preserved and the worker can pass `&req.color_targets`
        // straight into wgpu's `&[Option<ColorTargetState>]` field.
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = g
            .fragment
            .as_ref()
            .map(|f| {
                f.targets
                    .iter()
                    .map(|t| {
                        t.as_ref().map(|t| wgpu::ColorTargetState {
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
            depth_write_enabled: d.depth_write_enabled,
            depth_compare: cv::compare_function(d.depth_compare),
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

        self.pipeline_compiler.submit(pipeline_compiler::BuildRequest {
            id,
            label: desc.label.unwrap_or("compositor.async").into(),
            layout,
            vs_module,
            vs_entry: vs_entry_name,
            fs_module,
            fs_entry: fs_entry_name,
            vertex_buffers,
            color_targets,
            primitive,
            depth_stencil,
            multisample,
        });
        id
    }

    fn poll_async_pipelines(&mut self) -> Vec<AsyncPipelineCompletion> {
        // Drain the worker's outbound channel and adopt successful
        // pipelines into the regular pipeline pool. Locally-staged
        // failures (handle resolution failures that never reached
        // the worker) are appended at the end.
        let mut out = Vec::new();
        for r in self.pipeline_compiler.drain() {
            let result = match r.outcome {
                Ok(pipeline) => {
                    let (slot, generation) = self
                        .pools
                        .pipelines
                        .insert(PipelineEntry::Graphics(pipeline));
                    Ok(PipelineHandle::new(slot, generation))
                }
                Err(e) => Err(RhiError::Backend(e)),
            };
            out.push(AsyncPipelineCompletion { id: r.id, result });
        }
        out.append(&mut self.async_pipeline_local_failures);
        out
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
        let Some(parent_tex) = tex.texture.as_ref() else {
            log::warn!(
                "write_texture: handle {handle:?} is an external view; \
                 cannot upload (host owns the parent texture)"
            );
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: parent_tex,
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
            // wgpu 27 returns Ok even when the surface is suboptimal —
            // a `suboptimal: bool` is set on the SurfaceTexture itself.
            // Caller can choose to reconfigure on suboptimal next frame.
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated) => {
                // Surface is out of date — caller should resize + retry.
                // Reconfigure eagerly so the next acquire works.
                sc.surface.configure(&self.device, &sc.config);
                return Err(RhiError::SurfaceLost);
            }
            Err(wgpu::SurfaceError::Lost) => {
                return Err(RhiError::SurfaceLost);
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Err(RhiError::SwapchainAcquireFailed(
                    "get_current_texture timed out".into(),
                ));
            }
            Err(e) => {
                return Err(RhiError::SwapchainAcquireFailed(e.to_string()));
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
            texture: Some(texture_clone),
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
        self.device.poll(wgpu::Maintain::Wait);
        for bucket in self.destroy.drain_all() {
            self.release_bucket(&bucket);
        }
    }
}

// Helper: treat a dummy `RenderPassInfo` construction path as part of this
// module so it compiles without warnings about unused imports.
#[allow(dead_code)]
fn _rp_info_type_anchor(_: &RenderPassInfo) {}
