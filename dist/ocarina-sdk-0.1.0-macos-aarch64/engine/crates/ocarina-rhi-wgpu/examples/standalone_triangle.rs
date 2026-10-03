//! Standalone-runtime smoke test for `WgpuRhi::new_for_window` + swapchain.
//!
//! Opens a winit window, creates a `WgpuRhi` that owns its own
//! instance/device/queue, renders a hello-triangle to the swapchain each
//! frame. This is the path a shipped `ocarina-runtime` game will take —
//! the editor uses `WgpuRhi::from_existing` instead.
//!
//! Run with: `cargo run -p ocarina-rhi-wgpu --example standalone_triangle`

use std::sync::Arc;

use ocarina_rhi::{
    ColorAttachment, ColorTargetState, ColorWrites, CommandList, FragmentStage,
    GraphicsPipelineDesc, LoadOp, MultisampleState, PipelineDesc, PipelineHandle, PipelineKind,
    PresentMode, PrimitiveState, RenderPassDesc, Rhi, ShaderBytecode, ShaderDesc, ShaderStage,
    StoreOp, SwapchainHandle, TextureFormat, VertexStage,
};
use ocarina_rhi_wgpu::WgpuRhi;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const TRIANGLE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.60),
        vec2<f32>(-0.60, -0.50),
        vec2<f32>( 0.60, -0.50),
    );
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(0.95, 0.40, 0.25),
        vec3<f32>(0.20, 0.85, 0.35),
        vec3<f32>(0.25, 0.55, 0.95),
    );
    var out: VsOut;
    out.position = vec4<f32>(positions[i], 0.0, 1.0);
    out.color = colors[i];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

struct App {
    // Lazily initialized: winit's `resumed` is where the window (and thus
    // the wgpu surface) can be created.
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    rhi: WgpuRhi,
    swapchain: SwapchainHandle,
    pipeline: PipelineHandle,
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let (mut rhi, swapchain) = WgpuRhi::new_for_window(
            Arc::clone(&window),
            size.width.max(1),
            size.height.max(1),
            PresentMode::Fifo,
        )
        .expect("WgpuRhi::new_for_window failed");

        // Shaders
        let vs = rhi
            .create_shader(&ShaderDesc {
                label: Some("triangle:vs_main"),
                stage: ShaderStage::Vertex,
                entry_point: "vs_main",
                bytecode: ShaderBytecode::Wgsl(TRIANGLE_WGSL),
            })
            .expect("vs");
        let fs = rhi
            .create_shader(&ShaderDesc {
                label: Some("triangle:fs_main"),
                stage: ShaderStage::Fragment,
                entry_point: "fs_main",
                bytecode: ShaderBytecode::Wgsl(TRIANGLE_WGSL),
            })
            .expect("fs");

        // The swapchain's surface format is sRGB; hardcode the matching
        // target format here. Real engines query the swapchain for this.
        let targets = [Some(ColorTargetState {
            format: TextureFormat::Bgra8UnormSrgb,
            blend: None,
            write_mask: ColorWrites::ALL,
        })];

        let pipeline = rhi
            .create_pipeline(&PipelineDesc {
                label: Some("triangle"),
                kind: PipelineKind::Graphics(GraphicsPipelineDesc {
                    vertex: VertexStage {
                        shader: vs,
                        buffers: &[],
                    },
                    fragment: Some(FragmentStage {
                        shader: fs,
                        targets: &targets,
                    }),
                    primitive: PrimitiveState {
                        cull_mode: None,
                        ..PrimitiveState::default()
                    },
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                }),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            })
            .expect("pipeline");

        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);

        Self {
            window,
            rhi,
            swapchain,
            pipeline,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        let _ = self.rhi.resize_swapchain(self.swapchain, w, h);
    }

    fn render(&mut self) {
        let image = match self.rhi.acquire_next_image(self.swapchain) {
            Ok(h) => h,
            Err(e) => {
                log::warn!("acquire failed: {e}");
                return;
            }
        };

        let mut list = CommandList::new();
        {
            let attachments = [Some(ColorAttachment {
                view: image,
                resolve_target: None,
                load: LoadOp::Clear([0.07, 0.09, 0.12, 1.0]),
                store: StoreOp::Store,
            })];
            let mut pass = list.begin_render_pass(&RenderPassDesc {
                label: Some("standalone.triangle"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.draw(0..3, 0..1);
        }
        let _ = self.rhi.submit(list);
        let _ = self.rhi.present(self.swapchain);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("ocarina standalone_triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 600.0));
        let window = event_loop
            .create_window(attrs)
            .expect("create_window failed");
        let window = Arc::new(window);
        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width.max(1), size.height.max(1)),
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run_app");
}
