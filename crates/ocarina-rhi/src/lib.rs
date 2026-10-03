//! Ocarina Render Hardware Interface (RHI).
//!
//! Backend-agnostic GPU abstraction. The [`Rhi`] trait is the full surface —
//! everything above this trait (materials, meshes, scene rendering) is engine
//! code; everything below (device, queue, fences, allocators) lives inside a
//! backend.
//!
//! ## Design invariants
//!
//! - **Opaque handles.** Backends mint [`BufferHandle`], [`TextureHandle`], …
//!   values; user code treats them as tokens. Handles are copy-cheap and
//!   generational, so stale handles are detectable.
//! - **Deferred destruction.** `destroy_*` enqueues release; the backend
//!   retires resources after `MAX_FRAMES_IN_FLIGHT` frames, so user code
//!   never needs to sync.
//! - **Auto state tracking.** Each backend tracks resource states internally
//!   and emits transitions at submit time. User code never writes barriers.
//! - **Modern explicit binding.** Bind group layouts are first-class objects;
//!   pipelines list the layouts they expect.
//! - **HLSL canonical.** Shaders are shipped as pre-compiled bytecode
//!   (SPIR-V / DXIL / MSL / WGSL) — authoring is HLSL via DXC at build time.
//!
//! ## Backends
//!
//! - [`ocarina-rhi-null`](../ocarina_rhi_null/index.html) — no-op backend
//!   used for unit tests and the anti-leak guardrail.
//! - [`ocarina-rhi-wgpu`](../ocarina_rhi_wgpu/index.html) — wgpu backend
//!   (Phase 1). Targets Vulkan / Metal / DX12 native.

pub mod binding;
pub mod capability;
pub mod command;
pub mod error;
pub mod format;
pub mod handle;
pub mod pipeline;
pub mod resource;
pub mod shader;
pub mod swapchain;

// ────────── Re-exports for convenience ──────────

pub use binding::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingResource,
    BindingType, SamplerBindingType, StorageTextureAccess, TextureSampleType,
    TextureViewDimension,
};
pub use capability::{AdapterKind, Capabilities, Features, Limits};
pub use command::{
    ColorAttachment, Command, CommandList, ComputePass, DepthStencilAttachment, LoadOp, Rect,
    RenderPass, RenderPassDesc, RenderPassInfo, StoreOp, TextureCopyRegion, Viewport,
};
pub use error::{RhiError, RhiResult};
pub use format::{IndexFormat, TextureFormat, VertexFormat};
pub use handle::{
    BindGroupHandle, BindGroupLayoutHandle, BufferHandle, PipelineHandle, RawHandle,
    SamplerHandle, ShaderHandle, SwapchainHandle, TextureHandle,
};
pub use pipeline::{
    AsyncPipelineCompletion, AsyncPipelineId, BlendComponent, BlendFactor, BlendOperation,
    BlendState, ColorTargetState, ColorWrites, ComputePipelineDesc, DepthBias,
    DepthStencilState, Face, FragmentStage, FrontFace, GraphicsPipelineDesc,
    MultisampleState, PipelineDesc, PipelineKind, PolygonMode, PrimitiveState,
    PrimitiveTopology, PushConstantRange, StencilFaceState, StencilOp, VertexAttribute,
    VertexBufferLayout, VertexStage, VertexStepMode,
};
pub use resource::{
    AddressMode, BorderColor, BufferDesc, BufferUsage, CompareFunction, FilterMode,
    MemoryLocation, SamplerDesc, TextureDesc, TextureDimension, TextureRegion, TextureUsage,
};
pub use shader::{ShaderBytecode, ShaderDesc, ShaderStage, ShaderStageFlags};
pub use swapchain::{PresentMode, SwapchainDesc};

// ────────── The trait ──────────

/// Backend-agnostic GPU interface. Every backend implements this; every engine
/// layer above the RHI talks to this trait.
///
/// See module-level docs for invariants (deferred destruction, auto state
/// tracking, opaque handles).
pub trait Rhi {
    // ── Capability query ──

    /// Features and limits advertised by this backend on this device.
    fn capabilities(&self) -> &Capabilities;

    /// Short human-readable backend name (`"null"`, `"wgpu"`, `"vulkan"`, ...).
    fn backend_name(&self) -> &'static str;

    // ── Resource creation ──

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> RhiResult<BufferHandle>;
    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> RhiResult<TextureHandle>;
    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> RhiResult<SamplerHandle>;
    fn create_shader(&mut self, desc: &ShaderDesc<'_>) -> RhiResult<ShaderHandle>;
    fn create_bind_group_layout(
        &mut self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> RhiResult<BindGroupLayoutHandle>;
    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> RhiResult<BindGroupHandle>;
    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> RhiResult<PipelineHandle>;

    /// Submit a graphics pipeline build for backend-managed async
    /// compilation. Returns immediately with an opaque
    /// [`AsyncPipelineId`]; the actual [`PipelineHandle`] (or compile
    /// error) becomes available via the next call to
    /// [`Self::poll_async_pipelines`].
    ///
    /// Backends decide where the work runs: the wgpu backend uses an
    /// internal worker thread so `create_render_pipeline`'s
    /// 50-500 ms driver compile doesn't block the render thread on
    /// shader hot-reload; the null backend resolves synchronously
    /// and stages the result for the next poll. Either way, callers
    /// get a stable trait-level shape.
    ///
    /// Compute pipelines are not yet supported on this path; passing
    /// a compute desc is a programming error and the implementation
    /// is allowed to panic or return an error on the next poll.
    fn create_pipeline_async(&mut self, desc: &PipelineDesc<'_>) -> AsyncPipelineId;

    /// Drain async pipeline builds completed since the last call.
    /// Each entry's `id` matches an [`AsyncPipelineId`] previously
    /// returned by [`Self::create_pipeline_async`]; successful builds
    /// yield a [`PipelineHandle`] adopted into this RHI's pool, which
    /// behaves identically to one from [`Self::create_pipeline`]
    /// (resolves through `graphics_pipeline`, must be released with
    /// `destroy_pipeline`).
    ///
    /// Order is implementation-defined. Callers should match by id,
    /// not position.
    fn poll_async_pipelines(&mut self) -> Vec<AsyncPipelineCompletion>;

    fn create_swapchain(&mut self, desc: &SwapchainDesc<'_>) -> RhiResult<SwapchainHandle>;

    // ── Resource destruction (deferred) ──

    fn destroy_buffer(&mut self, handle: BufferHandle);
    fn destroy_texture(&mut self, handle: TextureHandle);
    fn destroy_sampler(&mut self, handle: SamplerHandle);
    fn destroy_shader(&mut self, handle: ShaderHandle);
    fn destroy_bind_group_layout(&mut self, handle: BindGroupLayoutHandle);
    fn destroy_bind_group(&mut self, handle: BindGroupHandle);
    fn destroy_pipeline(&mut self, handle: PipelineHandle);
    fn destroy_swapchain(&mut self, handle: SwapchainHandle);

    // ── Data upload ──

    /// Write data into `handle` at `offset`. For `MemoryLocation::Upload` buffers
    /// this is a CPU memcpy; for `Device`-only buffers the backend stages
    /// through an upload heap.
    fn write_buffer(&mut self, handle: BufferHandle, offset: u64, data: &[u8]);

    /// Write pixels into a texture region. Backend stages as needed.
    fn write_texture(
        &mut self,
        handle: TextureHandle,
        region: &TextureRegion,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
    );

    // ── Swapchain + frame lifecycle ──

    /// Request the next swapchain image. Returns a [`TextureHandle`] that is
    /// valid for the lifetime of the current frame only.
    fn acquire_next_image(&mut self, swapchain: SwapchainHandle) -> RhiResult<TextureHandle>;

    /// Resize a swapchain. Call when the platform window changes size.
    fn resize_swapchain(
        &mut self,
        swapchain: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> RhiResult<()>;

    /// Submit a recorded [`CommandList`] for execution.
    fn submit(&mut self, list: CommandList) -> RhiResult<()>;

    /// Present the most recently acquired image of `swapchain`.
    fn present(&mut self, swapchain: SwapchainHandle) -> RhiResult<()>;

    /// Block until the GPU has retired all in-flight work. Use sparingly —
    /// this defeats pipelining — but necessary before destroying the device
    /// or resizing some resources.
    fn wait_idle(&mut self);
}
