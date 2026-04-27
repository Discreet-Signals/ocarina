//! Windowed runtime: opens a winit window, renders the project's
//! entities through the **RHI** (today `ocarina-rhi-wgpu`; tomorrow
//! a hypothetical `ocarina-rhi-vulkan` slots in by changing one dep
//! and zero call sites), and ticks scripts every frame.
//!
//! Resources (lit pipeline, meshes, UBOs) and the per-frame command
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
use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton as WMouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use ocarina_core::components::{Material, Mesh};
use ocarina_core::{AssetPath, Key, MouseButton};
use ocarina_render::{
    CameraUniform, InstanceUniform, LitPipeline, MeshHandle, MeshRegistry, INSTANCE_STRIDE,
    MAX_INSTANCES,
};
use ocarina_rhi::{
    ColorAttachment, CommandList, DepthStencilAttachment, IndexFormat, LoadOp, PresentMode,
    RenderPassDesc, Rhi, RhiResult, StoreOp, SwapchainHandle, TextureDesc, TextureDimension,
    TextureFormat, TextureHandle, TextureUsage,
};
use ocarina_rhi_wgpu::WgpuRhi;

use crate::Runtime;

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// Open a window, render the project, and tick the runtime each
/// frame until the user closes the window.
///
/// Hosts (the binary `ocarina build` produces) call this from
/// `main` and that's the whole entry point.
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
    rhi: WgpuRhi,
    swapchain: SwapchainHandle,
    lit: LitPipeline,
    meshes: MeshRegistry,
    depth: DepthTarget,
    runtime: Runtime,
    last_frame: Instant,
    width: u32,
    height: u32,
}

struct DepthTarget {
    handle: TextureHandle,
    width: u32,
    height: u32,
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

        let lit = match ocarina_render::lit::build_lit_pipeline(
            &mut rhi,
            surface_format,
            DEPTH_FORMAT,
        ) {
            Ok(p) => p,
            Err(e) => {
                log::error!("lit pipeline init: {e}");
                event_loop.exit();
                return;
            }
        };

        let mut meshes = MeshRegistry::new();
        meshes.set_project_root(self.project_root.clone());
        if let Err(e) = meshes.register_builtins(&mut rhi) {
            log::error!("mesh builtins: {e}");
            event_loop.exit();
            return;
        }

        let depth = match make_depth(&mut rhi, width, height) {
            Ok(d) => d,
            Err(e) => {
                log::error!("depth target: {e}");
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
            rhi,
            swapchain,
            lit,
            meshes,
            depth,
            runtime,
            last_frame: Instant::now(),
            width,
            height,
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
                if let Ok(d) = make_depth(&mut state.rhi, state.width, state.height) {
                    state.rhi.destroy_texture(state.depth.handle);
                    state.depth = d;
                }
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
                let now = Instant::now();
                let dt = now - state.last_frame;
                state.last_frame = now;
                state.runtime.tick(dt);
                if let Err(e) = render_frame(state) {
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

// ─────────────────────────── Frame render ───────────────────────────

fn render_frame(state: &mut RunState) -> Result<()> {
    let aspect = state.width as f32 / state.height.max(1) as f32;

    // Active camera (and its entity) BEFORE we take any other
    // borrows on the world. First entity with a `Camera` component
    // wins; if none exist we fall back to a fixed orbit view in the
    // matrix block below.
    let active_camera_entity: Option<(Entity, ocarina_core::components::Camera)> = {
        let world_mut = state.runtime.world_mut();
        let mut q = world_mut.query::<(Entity, &ocarina_core::components::Camera)>();
        q.iter(world_mut).next().map(|(e, c)| (e, *c))
    };

    // Compose world transforms once. Used both for the camera
    // matrix lookup and for the per-entity draw matrices.
    let world_xforms = state.runtime.world_transforms();
    let world = state.runtime.world();

    let (view, proj) = if let Some((entity, cam)) = active_camera_entity
        .filter(|(e, _)| world_xforms.contains_key(e))
    {
        let cam_world = world_xforms[&entity];
        (
            cam_world.inverse(),
            Mat4::perspective_rh(cam.fov_y.to_radians(), aspect, cam.near, cam.far),
        )
    } else {
        let centroid = if world_xforms.is_empty() {
            Vec3::ZERO
        } else {
            let sum: Vec3 = world_xforms.values().map(|m| m.col(3).truncate()).sum();
            sum / world_xforms.len() as f32
        };
        let cam_pos = centroid + Vec3::new(3.0, 2.5, 5.0);
        (
            Mat4::look_at_rh(cam_pos, centroid, Vec3::Y),
            Mat4::perspective_rh(55f32.to_radians(), aspect, 0.05, 100.0),
        )
    };

    let cam_uniform = CameraUniform {
        view_proj: proj * view,
        light_dir: Vec4::new(0.35, 1.0, 0.85, 0.0).normalize(),
        base_color: Vec4::ONE,
        viewport_size: Vec4::new(state.width as f32, state.height as f32, 0.0, 0.0),
    };
    state
        .rhi
        .write_buffer(state.lit.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

    // Build per-instance UBO + draw call list. Skip entities
    // without a `Mesh` component -- those are cameras, audio
    // listeners, pure-script entities that have no visible body.
    let mut entries: Vec<(Entity, Mat4, AssetPath, glam::Vec4)> = Vec::new();
    for (entity, world_xform) in &world_xforms {
        let Some(mesh) = world.get::<Mesh>(*entity) else {
            continue;
        };
        if mesh.source.is_empty() {
            continue;
        }
        let tint = world
            .get::<Material>(*entity)
            .map(|m| m.tint)
            .unwrap_or_else(|| Vec4::new(0.78, 0.82, 0.90, 1.0));
        entries.push((*entity, *world_xform, mesh.source.clone(), tint));
    }
    entries.sort_by_key(|(e, ..)| *e);

    let mut draw_calls: Vec<(MeshHandle, u32)> = Vec::with_capacity(entries.len());
    let mut instance_bytes: Vec<u8> = vec![0; entries.len() * INSTANCE_STRIDE as usize];
    for (slot, (_entity, world_xform, source, tint)) in
        entries.iter().enumerate().take(MAX_INSTANCES as usize)
    {
        let Some(handle) = state.meshes.ensure(&mut state.rhi, source) else {
            continue;
        };
        let uniform = InstanceUniform {
            model: *world_xform,
            tint: *tint,
        };
        let off = slot * INSTANCE_STRIDE as usize;
        instance_bytes[off..off + std::mem::size_of::<InstanceUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&uniform));
        draw_calls.push((handle, slot as u32));
    }
    if !instance_bytes.is_empty() {
        state
            .rhi
            .write_buffer(state.lit.instance_buffer, 0, &instance_bytes);
    }

    // ── Acquire swapchain image + record render pass ──
    let frame_color = state
        .rhi
        .acquire_next_image(state.swapchain)
        .map_err(|e| anyhow!("acquire image: {e}"))?;

    let mut list = CommandList::new();
    {
        let color_attachments = [Some(ColorAttachment {
            view: frame_color,
            resolve_target: None,
            load: LoadOp::Clear([0.07, 0.09, 0.12, 1.0]),
            store: StoreOp::Store,
        })];
        let depth_attachment = DepthStencilAttachment {
            view: state.depth.handle,
            depth_load: LoadOp::Clear(1.0),
            depth_store: StoreOp::Store,
            stencil_load: LoadOp::Clear(0),
            stencil_store: StoreOp::Discard,
            read_only_depth: false,
            read_only_stencil: false,
        };
        let mut pass = list.begin_render_pass(&RenderPassDesc {
            label: Some("ocarina.runtime.scene"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(depth_attachment),
        });
        pass.set_pipeline(state.lit.pipeline);
        pass.set_bind_group(0, state.lit.camera_bg, &[]);
        for (mesh_handle, slot) in &draw_calls {
            let dyn_off = (*slot as u64 * INSTANCE_STRIDE) as u32;
            pass.set_vertex_buffer(0, mesh_handle.vertex_buffer, 0, None);
            pass.set_index_buffer(mesh_handle.index_buffer, 0, None, IndexFormat::Uint32);
            pass.set_bind_group(1, state.lit.instance_bg, &[dyn_off]);
            pass.draw_indexed(0..mesh_handle.index_count, 0, 0..1);
        }
    }
    state
        .rhi
        .submit(list)
        .map_err(|e| anyhow!("submit: {e}"))?;
    state
        .rhi
        .present(state.swapchain)
        .map_err(|e| anyhow!("present: {e}"))?;
    Ok(())
}

fn make_depth(rhi: &mut WgpuRhi, width: u32, height: u32) -> RhiResult<DepthTarget> {
    let handle = rhi.create_texture(&TextureDesc {
        label: Some("ocarina.runtime.depth"),
        dimension: TextureDimension::D2 {
            width: width.max(1),
            height: height.max(1),
            array_layers: 1,
        },
        format: DEPTH_FORMAT,
        usage: TextureUsage::RENDER_ATTACHMENT,
        mip_levels: 1,
        sample_count: 1,
    })?;
    Ok(DepthTarget {
        handle,
        width,
        height,
    })
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
