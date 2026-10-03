//! Windowed runtime: opens a winit window, renders the project's
//! entities through the **RHI** (today `ocarina-rhi-wgpu`; tomorrow
//! a hypothetical `ocarina-rhi-vulkan` slots in by changing one dep
//! and zero call sites), and ticks scripts every frame.
//!
//! Resources (default scene pipeline, meshes, UBOs) and the per-frame command
//! list both go through the RHI. The runtime never imports `wgpu`
//! directly. That's the load-bearing invariant the engine made
//! itself when it adopted an RHI: bypassing it would lock the
//! engine to one backend and defeat the abstraction.
//!
//! Editor-only render features (wireframe, unlit, RenderConfig
//! toggles) live in `ocarina-render` behind its `editor` feature.
//! This crate doesn't enable that feature, so none of that code
//! reaches the shipped runtime binary.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use bevy_ecs::entity::Entity;
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton as WMouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use ocarina_core::components::{Material, Mesh, Sky};
use ocarina_core::{Key, MouseButton};
use ocarina_render::{
    collect_lights_from_world, EntityInstance, RenderSettings, SceneArgs, SceneRenderer,
    ViewportCamera,
};
use ocarina_rhi::{CommandList, PresentMode, Rhi, SwapchainHandle};
use ocarina_rhi_wgpu::WgpuRhi;

use crate::Runtime;

/// Open a window, render the project, and tick the runtime each
/// frame until the user closes the window.
///
/// Hosts (the binary `ocarina build` produces) call this from
/// `main` and that's the whole entry point.
pub fn run(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = std::sync::Arc::new(project_root.as_ref().to_path_buf());
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
    project_root: std::sync::Arc<std::path::PathBuf>,
    state: Option<RunState>,
}

struct RunState {
    window: Arc<Window>,
    rhi: WgpuRhi,
    swapchain: SwapchainHandle,
    /// Single shared rendering core. Owns all pipelines, the depth
    /// + scene_color attachments, the post-process compositor + chain
    /// textures, and the asset registries. Identical instance shape
    /// between editor and runtime.
    scene: SceneRenderer,
    runtime: Runtime,
    last_frame: Instant,
    width: u32,
    height: u32,

    // ── Perf instrumentation ──
    frame_count: u64,
    total_tick_ms: f64,
    total_collect_ms: f64,
    total_upload_ms: f64,
    total_record_ms: f64,
    total_present_ms: f64,
    total_frame_ms: f64,
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

        let (mut rhi, swapchain) = match WgpuRhi::new_for_window(
            window.clone(),
            width,
            height,
            PresentMode::Fifo,
        ) {
            Ok(p) => p,
            Err(e) => {
                log::error!("rhi init: {e}");
                event_loop.exit();
                return;
            }
        };
        let surface_format = match rhi.swapchain_format(swapchain) {
            Some(f) => f,
            None => {
                log::error!("rhi: swapchain has no format");
                event_loop.exit();
                return;
            }
        };

        let mut scene = match SceneRenderer::new(&mut rhi, surface_format) {
            Ok(s) => s,
            Err(e) => {
                log::error!("scene renderer init: {e}");
                event_loop.exit();
                return;
            }
        };
        scene.set_project_root(self.project_root.clone());

        let (runtime, report) = Runtime::from_project(self.project_root.as_path());
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
            rhi,
            swapchain,
            scene,
            runtime,
            last_frame: Instant::now(),
            width,
            height,
            frame_count: 0,
            total_tick_ms: 0.0,
            total_collect_ms: 0.0,
            total_upload_ms: 0.0,
            total_record_ms: 0.0,
            total_present_ms: 0.0,
            total_frame_ms: 0.0,
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
                if let Err(e) =
                    state.rhi.resize_swapchain(state.swapchain, state.width, state.height)
                {
                    log::warn!("resize swapchain: {e}");
                }
                // SceneRenderer's `ensure_targets` reallocates depth +
                // scene_color + chain textures to match the new size
                // on the next `record_scene` call.
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = glam::Vec2::new(position.x as f32, position.y as f32);
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
                let frame_start = Instant::now();
                let dt = frame_start - state.last_frame;
                state.last_frame = frame_start;

                let tick_start = Instant::now();
                state.runtime.tick(dt);
                let tick_ms = tick_start.elapsed().as_secs_f64() * 1000.0;
                state.total_tick_ms += tick_ms;

                if let Err(e) = render_frame(state) {
                    log::warn!("render: {e}");
                }

                let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
                state.total_frame_ms += frame_ms;
                state.frame_count += 1;
                if state.frame_count >= 60 {
                    let n = state.frame_count as f64;
                    let avg = state.total_frame_ms / n;
                    log::info!(
                        "frame {:.2} ms ({:.1} fps) [tick {:.2} | collect {:.2} | upload {:.2} | record {:.2} | present {:.2}]",
                        avg,
                        1000.0 / avg,
                        state.total_tick_ms / n,
                        state.total_collect_ms / n,
                        state.total_upload_ms / n,
                        state.total_record_ms / n,
                        state.total_present_ms / n,
                    );
                    state.frame_count = 0;
                    state.total_frame_ms = 0.0;
                    state.total_tick_ms = 0.0;
                    state.total_collect_ms = 0.0;
                    state.total_upload_ms = 0.0;
                    state.total_record_ms = 0.0;
                    state.total_present_ms = 0.0;
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

// ─────────────────────────── Frame render ───────────────────────────

fn render_frame(state: &mut RunState) -> Result<()> {
    let collect_start = Instant::now();
    let aspect = state.width as f32 / state.height.max(1) as f32;

    // Active camera (clone -- Camera owns an AssetPath for
    // post_process so it isn't Copy).
    let active_camera: Option<(Entity, ocarina_core::components::Camera)> = {
        let world_mut = state.runtime.world_mut();
        let mut q = world_mut.query::<(Entity, &ocarina_core::components::Camera)>();
        q.iter(world_mut).next().map(|(e, c)| (e, c.clone()))
    };

    let world_xforms = state.runtime.world_transforms();

    let (view_proj, eye_pos) = if let Some((entity, cam)) = active_camera
        .as_ref()
        .filter(|(e, _)| world_xforms.contains_key(e))
    {
        let cam_world = world_xforms[entity];
        let view = cam_world.inverse();
        let proj = Mat4::perspective_rh(cam.fov_y.to_radians(), aspect, cam.near, cam.far);
        (proj * view, cam_world.col(3).truncate())
    } else {
        let centroid = if world_xforms.is_empty() {
            Vec3::ZERO
        } else {
            let sum: Vec3 = world_xforms.values().map(|m| m.col(3).truncate()).sum();
            sum / world_xforms.len() as f32
        };
        let cam_pos = centroid + Vec3::new(3.0, 2.5, 5.0);
        let view = Mat4::look_at_rh(cam_pos, centroid, Vec3::Y);
        let proj = Mat4::perspective_rh(55f32.to_radians(), aspect, 0.05, 100.0);
        (proj * view, cam_pos)
    };

    // Lights + sky from world.
    let (lights, sky) = {
        let world_mut = state.runtime.world_mut();
        let lights = collect_lights_from_world(world_mut, &world_xforms);
        let sky = {
            let mut q = world_mut.query::<(Entity, &Sky)>();
            q.iter(world_mut).next().map(|(_, s)| *s)
        };
        (lights, sky)
    };

    // Per-camera scene_shader, post_process. Falls back to
    // RenderSettings::default_shader, then engine default.
    let world = state.runtime.world();
    let scene_shader = match active_camera
        .as_ref()
        .map(|(_, c)| c.scene_shader.clone())
        .filter(|p| !p.is_empty())
    {
        Some(per_camera) => std::sync::Arc::new(per_camera),
        None => world
            .get_resource::<RenderSettings>()
            .map(|s| s.default_shader.clone())
            .unwrap_or_else(ocarina_render::engine_default_shader_path),
    };
    let post_process = active_camera
        .as_ref()
        .map(|(_, c)| c.post_process.clone())
        .unwrap_or_default();

    // Per-entity data.
    let mut entities: Vec<EntityInstance> = Vec::new();
    for (entity, world_xform) in &world_xforms {
        let Some(mesh) = world.get::<Mesh>(*entity) else {
            continue;
        };
        if mesh.source.is_empty() {
            continue;
        }
        let material = world
            .get::<Material>(*entity)
            .cloned()
            .unwrap_or_default();
        entities.push(EntityInstance {
            entity: *entity,
            world_xform: *world_xform,
            mesh: Some(mesh.source.clone()),
            material,
        });
    }
    entities.sort_by_key(|e| e.entity);

    state.total_collect_ms += collect_start.elapsed().as_secs_f64() * 1000.0;
    let upload_start = Instant::now();

    // Acquire swapchain target.
    let frame_target = state
        .rhi
        .acquire_next_image(state.swapchain)
        .map_err(|e| anyhow!("acquire image: {e}"))?;

    let camera = ViewportCamera {
        view_proj,
        eye_pos,
    };
    let args = SceneArgs {
        target: frame_target,
        target_size: (state.width, state.height),
        rect: (0.0, 0.0, state.width as f32, state.height as f32),
        camera,
        entities: &entities,
        lights: &lights,
        sky,
        scene_shader: &scene_shader,
        post_process: &post_process,
    };

    state.total_upload_ms += upload_start.elapsed().as_secs_f64() * 1000.0;
    let record_start = Instant::now();

    let mut list = CommandList::new();
    state.scene.record_scene(&mut state.rhi, &args, &mut list);
    state.scene.record_compositor(&mut state.rhi, &args, &mut list);

    state.total_record_ms += record_start.elapsed().as_secs_f64() * 1000.0;
    let present_start = Instant::now();

    state
        .rhi
        .submit(list)
        .map_err(|e| anyhow!("submit: {e}"))?;
    state
        .rhi
        .present(state.swapchain)
        .map_err(|e| anyhow!("present: {e}"))?;

    state.total_present_ms += present_start.elapsed().as_secs_f64() * 1000.0;
    Ok(())
}

// ─────────────────────────── Input mapping ───────────────────────────

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
