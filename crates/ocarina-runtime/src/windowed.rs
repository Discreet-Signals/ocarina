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
use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton as WMouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use std::collections::HashMap;

use ocarina_core::components::{Material, Mesh, Sky};
use ocarina_core::{AssetPath, Key, MouseButton};
use ocarina_render::shader::ShaderRegistry;
use ocarina_render::{
    build_lights_uniform, build_material_textures_bg, build_sky_pipeline,
    collect_lights_from_world, CameraUniform, ColorSpace, DefaultPipeline, InstanceUniform,
    MaterialTextureKey, MaterialUniform, MeshHandle, MeshRegistry, RenderSettings, SkyPipeline,
    SkyUniform, TextureRegistry, ENGINE_SHADER_DEFAULT, INSTANCE_STRIDE, MATERIAL_STRIDE,
    MATERIAL_UNIFORM_SIZE, MAX_INSTANCES,
};
use ocarina_rhi::{
    BindGroupHandle, ColorAttachment, CommandList, DepthStencilAttachment, IndexFormat, LoadOp,
    PresentMode, RenderPassDesc, Rhi, RhiResult, StoreOp, SwapchainHandle, TextureDesc,
    TextureDimension, TextureFormat, TextureHandle, TextureUsage,
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
    default_pipeline: DefaultPipeline,
    sky_pipeline: SkyPipeline,
    meshes: MeshRegistry,
    textures: TextureRegistry,
    shader_registry: ShaderRegistry,
    material_bg_cache: HashMap<MaterialTextureKey, BindGroupHandle>,
    depth: DepthTarget,
    runtime: Runtime,
    last_frame: Instant,
    width: u32,
    height: u32,

    // ── Perf instrumentation ──
    /// Per-phase elapsed accumulators. Drained + logged every 60 frames
    /// so a stable average covers vsync jitter without being too noisy.
    frame_count: u64,
    total_tick_ms: f64,
    total_collect_ms: f64,
    total_upload_ms: f64,
    total_record_ms: f64,
    total_present_ms: f64,
    total_frame_ms: f64,
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

        let default_pipeline = match ocarina_render::default::build_default_pipeline(
            &mut rhi,
            surface_format,
            DEPTH_FORMAT,
        ) {
            Ok(p) => p,
            Err(e) => {
                log::error!("default scene pipeline init: {e}");
                event_loop.exit();
                return;
            }
        };

        let sky_pipeline = match build_sky_pipeline(&mut rhi, surface_format, DEPTH_FORMAT) {
            Ok(p) => p,
            Err(e) => {
                log::error!("sky pipeline init: {e}");
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

        let mut textures = match TextureRegistry::new(&mut rhi) {
            Ok(t) => t,
            Err(e) => {
                log::error!("texture registry init: {e}");
                event_loop.exit();
                return;
            }
        };
        textures.set_project_root(self.project_root.clone());

        let mut shader_registry = ShaderRegistry::new();
        shader_registry.set_project_root(self.project_root.clone());

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
            default_pipeline,
            sky_pipeline,
            meshes,
            textures,
            shader_registry,
            material_bg_cache: HashMap::new(),
            depth,
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
    // `world` is taken AFTER the lights collection below: that block
    // grabs `world_mut` to build query states, and an immutable borrow
    // here would conflict.

    let (view, proj, eye_pos) = if let Some((entity, cam)) = active_camera_entity
        .filter(|(e, _)| world_xforms.contains_key(e))
    {
        let cam_world = world_xforms[&entity];
        (
            cam_world.inverse(),
            Mat4::perspective_rh(cam.fov_y.to_radians(), aspect, cam.near, cam.far),
            cam_world.col(3).truncate(),
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
            cam_pos,
        )
    };

    state.total_collect_ms += collect_start.elapsed().as_secs_f64() * 1000.0;
    let upload_start = Instant::now();

    let cam_uniform = CameraUniform {
        view_proj: proj * view,
        eye_pos: Vec4::new(eye_pos.x, eye_pos.y, eye_pos.z, 0.0),
        viewport_size: Vec4::new(state.width as f32, state.height as f32, 0.0, 0.0),
    };
    state.rhi.write_buffer(
        state.default_pipeline.camera_buffer,
        0,
        bytemuck::bytes_of(&cam_uniform),
    );

    // Pack scene lights from `DirectionalLight` / `PointLight` /
    // `SpotLight` components and upload to the lights UBO. While
    // we have a mut World handle, also pull the first `Sky`
    // component (if any) for the sky pass below.
    let (lights, sky) = {
        let world_mut = state.runtime.world_mut();
        let lights = collect_lights_from_world(world_mut, &world_xforms);
        let sky = {
            let mut q = world_mut.query::<(Entity, &Sky)>();
            q.iter(world_mut).next().map(|(_, s)| *s)
        };
        (lights, sky)
    };
    let lights_uniform = build_lights_uniform(&lights);
    state.rhi.write_buffer(
        state.default_pipeline.lights_buffer,
        0,
        bytemuck::bytes_of(&lights_uniform),
    );

    // Sky uniform upload, conditional on a Sky component existing.
    if let Some(sky) = sky {
        let sky_uniform = SkyUniform::build(&sky, proj * view, eye_pos);
        state.rhi.write_buffer(
            state.sky_pipeline.buffer,
            0,
            bytemuck::bytes_of(&sky_uniform),
        );
    }
    let do_sky = sky.is_some();

    // All mut borrows on `state.runtime` are now done. Take a fresh
    // immut `world` ref for the resource lookup + entry-data loop.
    let world = state.runtime.world();

    // Look up the active scene shader from `RenderSettings`. This
    // is the gameplay-side handle for swapping the scene shader at
    // runtime; `DefaultPipeline::pipeline_for` resolves it through
    // the cache, falling back to `engine/shaders/default` on miss.
    let scene_shader = world
        .get_resource::<RenderSettings>()
        .map(|s| s.default_shader.clone())
        .unwrap_or_else(|| AssetPath::new(ENGINE_SHADER_DEFAULT));

    // Build per-entity data: (entity, world transform, mesh path,
    // material). Skip entities without a `Mesh` component -- those
    // are cameras, audio listeners, pure-script entities that have
    // no visible body.
    struct EntryData {
        world_xform: Mat4,
        mesh_path: AssetPath,
        material: Material,
        entity: Entity,
    }
    let mut entries: Vec<EntryData> = Vec::new();
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
        entries.push(EntryData {
            entity: *entity,
            world_xform: *world_xform,
            mesh_path: mesh.source.clone(),
            material,
        });
    }
    entries.sort_by_key(|e| e.entity);

    struct DrawCall {
        mesh: MeshHandle,
        slot: u32,
        material_bg: Option<BindGroupHandle>,
    }
    let mut draw_calls: Vec<DrawCall> = Vec::with_capacity(entries.len());
    let mut instance_bytes: Vec<u8> = vec![0; entries.len() * INSTANCE_STRIDE as usize];
    let mut material_bytes: Vec<u8> = vec![0; entries.len() * MATERIAL_STRIDE as usize];
    for (slot, entry) in entries.iter().enumerate().take(MAX_INSTANCES as usize) {
        let Some(handle) = state.meshes.ensure(&mut state.rhi, &entry.mesh_path) else {
            continue;
        };

        // Mirror albedo into the instance UBO's `tint` so simple
        // pipelines (none used by the runtime today, but parity with
        // the editor matters for shared shaders) and the scene
        // shader land on the same display colour.
        let instance = InstanceUniform {
            model: entry.world_xform,
            tint: entry.material.albedo,
        };
        let material = MaterialUniform::from_component(&entry.material);

        let inst_off = slot * INSTANCE_STRIDE as usize;
        instance_bytes[inst_off..inst_off + std::mem::size_of::<InstanceUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&instance));
        let mat_off = slot * MATERIAL_STRIDE as usize;
        material_bytes[mat_off..mat_off + MATERIAL_UNIFORM_SIZE as usize]
            .copy_from_slice(bytemuck::bytes_of(&material));

        let material_bg = ensure_material_bg(
            &mut state.rhi,
            &mut state.textures,
            &mut state.material_bg_cache,
            state.default_pipeline.material_textures_bgl,
            &entry.material,
        );

        draw_calls.push(DrawCall {
            mesh: handle,
            slot: slot as u32,
            material_bg,
        });
    }
    if !instance_bytes.is_empty() {
        state
            .rhi
            .write_buffer(state.default_pipeline.instance_buffer, 0, &instance_bytes);
        state
            .rhi
            .write_buffer(state.default_pipeline.material_buffer, 0, &material_bytes);
    }

    // Resolve the active scene pipeline. Done up front so any
    // pipeline-cache build (Stage 9) lands before we acquire
    // immutable RHI borrows for the render pass.
    let pipeline_handle = state.default_pipeline.pipeline_for(
        &mut state.rhi,
        &mut state.shader_registry,
        &scene_shader,
    );

    state.total_upload_ms += upload_start.elapsed().as_secs_f64() * 1000.0;
    let record_start = Instant::now();

    // ── Acquire swapchain image + record render pass ──
    let frame_color = state
        .rhi
        .acquire_next_image(state.swapchain)
        .map_err(|e| anyhow!("acquire image: {e}"))?;

    let mut list = CommandList::new();

    // ─── Sky pass ───
    // Runs first when a Sky component exists; replaces the lit
    // pass's solid clear with a per-pixel world-space gradient.
    if do_sky {
        let sky_color_attachments = [Some(ColorAttachment {
            view: frame_color,
            resolve_target: None,
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
        })];
        let mut pass = list.begin_render_pass(&RenderPassDesc {
            label: Some("ocarina.runtime.sky"),
            color_attachments: &sky_color_attachments,
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(state.sky_pipeline.pipeline);
        pass.set_bind_group(0, state.sky_pipeline.bg, &[]);
        pass.draw(0..6, 0..1);
    }

    // ─── Lit pass ───
    {
        let lit_color_load = if do_sky {
            LoadOp::Load
        } else {
            LoadOp::Clear([0.07, 0.09, 0.12, 1.0])
        };
        let color_attachments = [Some(ColorAttachment {
            view: frame_color,
            resolve_target: None,
            load: lit_color_load,
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
        pass.set_pipeline(pipeline_handle);
        pass.set_bind_group(0, state.default_pipeline.frame_bg, &[]);
        for d in &draw_calls {
            let inst_off = (u64::from(d.slot) * INSTANCE_STRIDE) as u32;
            let mat_off = (u64::from(d.slot) * MATERIAL_STRIDE) as u32;
            pass.set_vertex_buffer(0, d.mesh.vertex_buffer, 0, None);
            pass.set_index_buffer(d.mesh.index_buffer, 0, None, IndexFormat::Uint32);
            pass.set_bind_group(
                1,
                state.default_pipeline.instance_bg,
                &[inst_off, mat_off],
            );
            if let Some(bg) = d.material_bg {
                pass.set_bind_group(2, bg, &[]);
            }
            pass.draw_indexed(0..d.mesh.index_count, 0, 0..1);
        }
    }
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

/// Resolve `material`'s four texture slots through the registry,
/// look up (or build) the cached group-2 bind group, and return its
/// handle. Two entities with identical texture sets share one bind
/// group; lookup is the hot path inside the entity loop.
fn ensure_material_bg(
    rhi: &mut WgpuRhi,
    textures: &mut TextureRegistry,
    cache: &mut HashMap<MaterialTextureKey, BindGroupHandle>,
    layout: ocarina_rhi::BindGroupLayoutHandle,
    material: &Material,
) -> Option<BindGroupHandle> {
    let albedo =
        textures.ensure_or_white(rhi, &material.albedo_texture, ColorSpace::Srgb);
    let normal = textures.ensure_or_flat_normal(rhi, &material.normal_texture);
    let metallic_roughness = textures.ensure_or_white(
        rhi,
        &material.metallic_roughness_texture,
        ColorSpace::Linear,
    );
    let emission =
        textures.ensure_or_white(rhi, &material.emission_texture, ColorSpace::Srgb);
    let key = MaterialTextureKey {
        albedo,
        normal,
        metallic_roughness,
        emission,
    };
    if let Some(h) = cache.get(&key) {
        return Some(*h);
    }
    match build_material_textures_bg(rhi, layout, key) {
        Ok(handle) => {
            cache.insert(key, handle);
            Some(handle)
        }
        Err(e) => {
            log::warn!("runtime: material bind-group create failed: {e}");
            None
        }
    }
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
