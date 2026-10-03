//! Off-thread graphics pipeline compilation.
//!
//! Internal to the wgpu backend. The trait surface is the
//! `create_pipeline_async` / `poll_async_pipelines` pair on
//! [`Rhi`](ocarina_rhi::Rhi); this module is the wgpu-specific
//! implementation: a worker thread that holds a `wgpu::Device`
//! clone, receives [`BuildRequest`]s with everything needed to
//! create a `wgpu::RenderPipeline` (shader modules, layout, target
//! state, all Clone-cheap wgpu types), and posts back
//! [`BuildResult`]s the RHI drains in `poll_async_pipelines`.
//!
//! Validation errors are captured via a wgpu error scope on the
//! worker side and surfaced as `Err(String)` so the trait-level
//! `Result<PipelineHandle, RhiError>` can carry the failure back to
//! the caller — broken user shaders don't silently produce invalid
//! pipelines.

use std::sync::mpsc;
use std::thread;

use ocarina_rhi::AsyncPipelineId;

/// Owned snapshot of everything `wgpu::Device::create_render_pipeline`
/// needs. Built on the main thread by `WgpuRhi::create_pipeline_async`
/// (which resolves [`ShaderHandle`](ocarina_rhi::ShaderHandle) and
/// [`BindGroupLayoutHandle`](ocarina_rhi::BindGroupLayoutHandle)s into
/// the wgpu types and clones them — `wgpu::ShaderModule` /
/// `BindGroupLayout` / `PipelineLayout` are all internally `Arc`, so
/// this is cheap), then handed off to the worker via channel.
pub(crate) struct BuildRequest {
    pub id: AsyncPipelineId,
    pub label: String,
    pub layout: wgpu::PipelineLayout,
    pub vs_module: wgpu::ShaderModule,
    pub vs_entry: String,
    pub fs_module: wgpu::ShaderModule,
    pub fs_entry: String,
    pub vertex_buffers: Vec<OwnedVertexBufferLayout>,
    pub color_targets: Vec<Option<wgpu::ColorTargetState>>,
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
}

/// Owned form of `wgpu::VertexBufferLayout`. The wgpu type borrows a
/// `&[VertexAttribute]`; we copy that slice so the request can be
/// moved across threads.
pub(crate) struct OwnedVertexBufferLayout {
    pub array_stride: wgpu::BufferAddress,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

/// Outcome of one async build, posted back to the RHI for adoption
/// into the pipeline pool.
pub(crate) struct BuildResult {
    pub id: AsyncPipelineId,
    pub outcome: Result<wgpu::RenderPipeline, String>,
}

/// Owns the worker thread + the channel ends. Drop closes the
/// inbound channel, the worker's `recv` returns `Err`, and the
/// thread exits cleanly.
pub(crate) struct PipelineCompiler {
    tx: mpsc::Sender<BuildRequest>,
    rx: mpsc::Receiver<BuildResult>,
    _thread: thread::JoinHandle<()>,
}

impl PipelineCompiler {
    pub(crate) fn spawn(device: wgpu::Device) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<BuildRequest>();
        let (resp_tx, resp_rx) = mpsc::channel::<BuildResult>();
        let handle = thread::Builder::new()
            .name("ocarina-rhi-wgpu-pipeline-compiler".into())
            .spawn(move || worker(device, req_rx, resp_tx))
            .expect("spawn pipeline compiler thread");
        Self {
            tx: req_tx,
            rx: resp_rx,
            _thread: handle,
        }
    }

    pub(crate) fn submit(&self, req: BuildRequest) {
        let _ = self.tx.send(req);
    }

    pub(crate) fn drain(&self) -> Vec<BuildResult> {
        let mut out = Vec::new();
        while let Ok(r) = self.rx.try_recv() {
            out.push(r);
        }
        out
    }
}

fn worker(
    device: wgpu::Device,
    rx: mpsc::Receiver<BuildRequest>,
    tx: mpsc::Sender<BuildResult>,
) {
    while let Ok(req) = rx.recv() {
        let outcome = compile(&device, &req);
        if tx
            .send(BuildResult {
                id: req.id,
                outcome,
            })
            .is_err()
        {
            break;
        }
    }
}

fn compile(
    device: &wgpu::Device,
    req: &BuildRequest,
) -> Result<wgpu::RenderPipeline, String> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    // Re-borrow the owned vertex layouts as the wgpu API expects.
    let vertex_layouts: Vec<wgpu::VertexBufferLayout<'_>> = req
        .vertex_buffers
        .iter()
        .map(|v| wgpu::VertexBufferLayout {
            array_stride: v.array_stride,
            step_mode: v.step_mode,
            attributes: &v.attributes,
        })
        .collect();

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&req.label),
        layout: Some(&req.layout),
        vertex: wgpu::VertexState {
            module: &req.vs_module,
            entry_point: Some(&req.vs_entry),
            buffers: &vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &req.fs_module,
            entry_point: Some(&req.fs_entry),
            targets: &req.color_targets,
            compilation_options: Default::default(),
        }),
        primitive: req.primitive,
        depth_stencil: req.depth_stencil.clone(),
        multisample: req.multisample,
        multiview: None,
        cache: None,
    });

    let err = block_on(device.pop_error_scope());
    match err {
        Some(e) => Err(format!("{e}")),
        None => Ok(pipeline),
    }
}

/// Single-future blocking driver. Async pipeline futures from wgpu
/// resolve as soon as the device's pending validation queue drains,
/// so we don't need a real runtime — just park the thread until the
/// future wakes us.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct ParkWaker(thread::Thread);
    impl Wake for ParkWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let mut fut = Box::pin(future);
    let waker = Waker::from(Arc::new(ParkWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
