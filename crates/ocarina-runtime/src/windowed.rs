//! Windowed runtime: opens a winit window, renders the project's
//! entities through wgpu, and ticks scripts every frame. This is the
//! mode shipped binaries run in (and the editor's "play" mode will
//! eventually drive the same `Runtime` through its own surface).
//!
//! Rendering is deliberately small: every entity with a `Transform`
//! is drawn as a tinted unit cube, lit by a single directional light.
//! Real meshes / textures land alongside the asset pipeline.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton as WMouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use ocarina_core::components::BasicMesh;
use ocarina_core::{Key, MouseButton};

use crate::Runtime;

// ─────────────────────────── Public entry ───────────────────────────

/// Open a window, render the project, and tick the runtime each
/// frame until the user closes the window.
///
/// This is the only function shipped binaries need to call. Their
/// `main` is typically:
/// ```ignore
/// fn main() -> anyhow::Result<()> {
///     ocarina_runtime::windowed::run(env!("CARGO_MANIFEST_DIR"))
/// }
/// ```
pub fn run(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = project_root.as_ref().to_path_buf();
    let event_loop = EventLoop::new().context("create event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        project_root,
        state: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("event loop: {e}"))?;
    Ok(())
}

// ─────────────────────────── App state ───────────────────────────

struct App {
    project_root: std::path::PathBuf,
    state: Option<RunState>,
}

struct RunState {
    window: Arc<Window>,
    gpu: Gpu,
    renderer: Renderer,
    runtime: Runtime,
    last_frame: Instant,
    width: u32,
    height: u32,
    /// Cursor position in physical pixels (y measured from top).
    cursor: glam::Vec2,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title("Ocarina")
                .with_inner_size(PhysicalSize::new(1280, 720)),
        ) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("create_window: {e}");
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let gpu = match pollster::block_on(Gpu::new(window.clone(), width, height)) {
            Ok(g) => g,
            Err(e) => {
                log::error!("gpu init: {e}");
                event_loop.exit();
                return;
            }
        };
        let renderer = match Renderer::new(&gpu) {
            Ok(r) => r,
            Err(e) => {
                log::error!("renderer init: {e}");
                event_loop.exit();
                return;
            }
        };
        let (runtime, report) = Runtime::from_project(&self.project_root);
        if !report.errors.is_empty() {
            log::warn!("scene loaded with {} errors", report.errors.len());
            for e in &report.errors {
                log::warn!("  - {e}");
            }
        }
        log::info!(
            "windowed runtime: {} entities, {} script(s) attached",
            runtime.entity_count(),
            runtime.script_count()
        );

        self.state = Some(RunState {
            window,
            gpu,
            renderer,
            runtime,
            last_frame: Instant::now(),
            width,
            height,
            cursor: glam::Vec2::ZERO,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.width = size.width.max(1);
                state.height = size.height.max(1);
                state.gpu.resize(state.width, state.height);
                state.renderer.resize(&state.gpu, state.width, state.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                state.cursor = pos;
                state.runtime.input_mut().set_mouse_pos(pos);
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let b = map_mouse_button(button);
                match btn_state {
                    ElementState::Pressed => state.runtime.input_mut().press_mouse(b),
                    ElementState::Released => state.runtime.input_mut().release_mouse(b),
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let v = match delta {
                    MouseScrollDelta::LineDelta(x, y) => glam::Vec2::new(x, y),
                    MouseScrollDelta::PixelDelta(p) => {
                        glam::Vec2::new(p.x as f32 / 100.0, p.y as f32 / 100.0)
                    }
                };
                state.runtime.input_mut().add_scroll(v);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if let Some(k) = map_key(code) {
                    match key_state {
                        ElementState::Pressed => state.runtime.input_mut().press_key(k),
                        ElementState::Released => state.runtime.input_mut().release_key(k),
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now - state.last_frame;
                state.last_frame = now;
                state.runtime.tick(dt);
                if let Err(e) = state.renderer.render(&state.gpu, &mut state.runtime) {
                    log::warn!("render: {e}");
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn map_mouse_button(b: WMouseButton) -> MouseButton {
    match b {
        WMouseButton::Left => MouseButton::Left,
        WMouseButton::Right => MouseButton::Right,
        WMouseButton::Middle => MouseButton::Middle,
        WMouseButton::Other(n) => MouseButton::Other(n),
        WMouseButton::Back => MouseButton::Other(4),
        WMouseButton::Forward => MouseButton::Other(5),
    }
}

fn map_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyA => Key::A, KeyCode::KeyB => Key::B, KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D, KeyCode::KeyE => Key::E, KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G, KeyCode::KeyH => Key::H, KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J, KeyCode::KeyK => Key::K, KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M, KeyCode::KeyN => Key::N, KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P, KeyCode::KeyQ => Key::Q, KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S, KeyCode::KeyT => Key::T, KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V, KeyCode::KeyW => Key::W, KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y, KeyCode::KeyZ => Key::Z,
        KeyCode::Digit0 => Key::Num0, KeyCode::Digit1 => Key::Num1,
        KeyCode::Digit2 => Key::Num2, KeyCode::Digit3 => Key::Num3,
        KeyCode::Digit4 => Key::Num4, KeyCode::Digit5 => Key::Num5,
        KeyCode::Digit6 => Key::Num6, KeyCode::Digit7 => Key::Num7,
        KeyCode::Digit8 => Key::Num8, KeyCode::Digit9 => Key::Num9,
        KeyCode::ArrowUp => Key::Up, KeyCode::ArrowDown => Key::Down,
        KeyCode::ArrowLeft => Key::Left, KeyCode::ArrowRight => Key::Right,
        KeyCode::Space => Key::Space, KeyCode::Enter => Key::Enter,
        KeyCode::Escape => Key::Escape, KeyCode::Tab => Key::Tab,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::ShiftLeft => Key::LShift, KeyCode::ShiftRight => Key::RShift,
        KeyCode::ControlLeft => Key::LCtrl, KeyCode::ControlRight => Key::RCtrl,
        KeyCode::AltLeft => Key::LAlt, KeyCode::AltRight => Key::RAlt,
        _ => return None,
    })
}

// ─────────────────────────── GPU host ───────────────────────────

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    surface_format: wgpu::TextureFormat,
}

impl Gpu {
    async fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .context("create_surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("no suitable wgpu adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ocarina.device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .context("request_device")?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|p| matches!(p, wgpu::PresentMode::Mailbox))
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            surface_format,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }
}

// ─────────────────────────── Renderer ───────────────────────────

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 1024;
const INSTANCE_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
    light_dir: Vec4,
    base_color: Vec4,
}
const CAMERA_UNIFORM_SIZE: u64 = std::mem::size_of::<CameraUniform>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    model: Mat4,
    tint: Vec4,
}
const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
};
struct Instance {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> instance: Instance;

struct VsIn { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, };
struct VsOut { @builtin(position) clip: vec4<f32>, @location(0) world_normal: vec3<f32>, };

@vertex fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = instance.model * vec4<f32>(in.position, 1.0);
    out.clip = camera.view_proj * world;
    out.world_normal = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(camera.light_dir.xyz);
    let lambert = max(dot(n, l), 0.0);
    let up = max(n.y, 0.0);
    let down = max(-n.y, 0.0);
    let ambient = vec3<f32>(0.18, 0.22, 0.30) * up + vec3<f32>(0.10, 0.08, 0.06) * down + vec3<f32>(0.06);
    let base = camera.base_color.rgb * instance.tint.rgb;
    let color = base * (ambient + vec3<f32>(lambert * 0.85));
    return vec4<f32>(color, 1.0);
}
"#;

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_bg: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl Renderer {
    fn new(gpu: &Gpu) -> Result<Self> {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ocarina.lit"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        // Bind group 0: camera.
        let camera_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera.bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE),
                    },
                    count: None,
                }],
            });
        let camera_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera.ubo"),
            size: CAMERA_UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera.bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Bind group 1: per-instance, dynamic offset.
        let instance_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("instance.bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(INSTANCE_UNIFORM_SIZE),
                        },
                        count: None,
                    }],
                });
        let instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance.ubo"),
            size: INSTANCE_STRIDE * MAX_INSTANCES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance.bg"),
            layout: &instance_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &instance_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(INSTANCE_UNIFORM_SIZE),
                }),
            }],
        });

        // Cube mesh.
        let (vertices, indices) = build_unit_cube();
        let vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube.vb"),
            size: (std::mem::size_of_val(&vertices[..])) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        let index_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube.ib"),
            size: (std::mem::size_of_val(&indices[..])) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&indices));

        // Pipeline.
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lit.layout"),
                bind_group_layouts: &[&camera_layout, &instance_layout],
                push_constant_ranges: &[],
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lit.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let depth_view = create_depth(gpu, gpu.config.width, gpu.config.height);

        Ok(Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            camera_buffer,
            camera_bg,
            instance_buffer,
            instance_bg,
            depth_view,
            width: gpu.config.width,
            height: gpu.config.height,
        })
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.depth_view = create_depth(gpu, width, height);
    }

    fn render(&mut self, gpu: &Gpu, runtime: &mut Runtime) -> Result<()> {
        // Camera: a fixed orbit looking at the world centroid. Real
        // games place a Camera entity; we'll honour that once we have
        // an `ActiveCamera` resource.
        let world_xforms = runtime.world_transforms();
        let centroid = if world_xforms.is_empty() {
            Vec3::ZERO
        } else {
            let sum: Vec3 = world_xforms
                .values()
                .map(|m| m.col(3).truncate())
                .sum();
            sum / world_xforms.len() as f32
        };
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let proj = Mat4::perspective_rh(55f32.to_radians(), aspect, 0.05, 100.0);
        let cam_pos = centroid + Vec3::new(3.0, 2.5, 5.0);
        let view = Mat4::look_at_rh(cam_pos, centroid, Vec3::Y);
        let cam_uniform = CameraUniform {
            view_proj: proj * view,
            light_dir: Vec4::new(0.35, 1.0, 0.85, 0.0).normalize(),
            base_color: Vec4::ONE,
        };
        gpu.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        // Instance batch: one cube per entity-with-Transform. Tint
        // honours BasicMesh.tint when present, otherwise a neutral
        // grey so empty entities are still visible.
        let mut entries: Vec<(bevy_ecs::entity::Entity, Mat4)> = world_xforms.into_iter().collect();
        entries.sort_by_key(|(e, _)| *e);
        let world = runtime.world();
        let mut instance_bytes: Vec<u8> =
            vec![0; (entries.len() * INSTANCE_STRIDE as usize).max(1)];
        for (i, (entity, model)) in entries.iter().enumerate().take(MAX_INSTANCES as usize) {
            let tint = world
                .get::<BasicMesh>(*entity)
                .map(|m| m.tint)
                .unwrap_or(Vec4::new(0.78, 0.82, 0.90, 1.0));
            let uniform = InstanceUniform {
                model: *model,
                tint,
            };
            let off = i * INSTANCE_STRIDE as usize;
            instance_bytes[off..off + INSTANCE_UNIFORM_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if !entries.is_empty() {
            gpu.queue
                .write_buffer(&self.instance_buffer, 0, &instance_bytes);
        }

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(e) => {
                log::warn!("get_current_texture: {e}");
                return Ok(());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocarina.scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.07, g: 0.09, b: 0.12, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for i in 0..entries.len().min(MAX_INSTANCES as usize) {
                let off = (i as u64 * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, &self.instance_bg, &[off]);
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn create_depth(gpu: &Gpu, width: u32, height: u32) -> wgpu::TextureView {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ocarina.depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn build_unit_cube() -> ([Vertex; 24], [u32; 36]) {
    let h = 0.5;
    // 6 faces × 4 verts each, flat-shaded (one normal per face).
    let mut v = [Vertex {
        position: [0.0; 3],
        normal: [0.0; 3],
    }; 24];
    let mut idx = [0u32; 36];

    let faces: [([Vec3; 4], [f32; 3]); 6] = [
        // +X
        ([
            Vec3::new(h, -h, -h), Vec3::new(h, h, -h),
            Vec3::new(h, h, h), Vec3::new(h, -h, h),
        ], [1.0, 0.0, 0.0]),
        // -X
        ([
            Vec3::new(-h, -h, h), Vec3::new(-h, h, h),
            Vec3::new(-h, h, -h), Vec3::new(-h, -h, -h),
        ], [-1.0, 0.0, 0.0]),
        // +Y
        ([
            Vec3::new(-h, h, h), Vec3::new(h, h, h),
            Vec3::new(h, h, -h), Vec3::new(-h, h, -h),
        ], [0.0, 1.0, 0.0]),
        // -Y
        ([
            Vec3::new(-h, -h, -h), Vec3::new(h, -h, -h),
            Vec3::new(h, -h, h), Vec3::new(-h, -h, h),
        ], [0.0, -1.0, 0.0]),
        // +Z
        ([
            Vec3::new(-h, -h, h), Vec3::new(h, -h, h),
            Vec3::new(h, h, h), Vec3::new(-h, h, h),
        ], [0.0, 0.0, 1.0]),
        // -Z
        ([
            Vec3::new(h, -h, -h), Vec3::new(-h, -h, -h),
            Vec3::new(-h, h, -h), Vec3::new(h, h, -h),
        ], [0.0, 0.0, -1.0]),
    ];

    for (i, (corners, normal)) in faces.iter().enumerate() {
        let base = (i * 4) as u32;
        for (j, c) in corners.iter().enumerate() {
            v[i * 4 + j] = Vertex {
                position: [c.x, c.y, c.z],
                normal: *normal,
            };
        }
        let ix = i * 6;
        idx[ix..ix + 6]
            .copy_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (v, idx)
}
