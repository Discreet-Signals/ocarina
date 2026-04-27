//! Ocarina viewport — Floem panel that renders the live ECS scene
//! through `ocarina-rhi-wgpu`, drawing directly into the editor's
//! swapchain backbuffer via `floem::Renderer::submit_wgpu_pass`.
//!
//! # Architecture
//!
//! The crate is split into two layers:
//!
//! - [`ViewportRenderer`] (in `renderer.rs`) — the renderer **core**.
//!   Stateless across "edit vs play" mode: takes a flat list of
//!   [`EntityInstance`]s, a [`ViewportCamera`], and a [`RenderMode`],
//!   and renders into a caller-supplied target. Reusable both by the
//!   editor's panel and (future) the shipped runtime's player window.
//!
//! - [`viewport_panel`] (in this file) — the **Floem binding**. Owns
//!   an [`OrbitCamera`] (edit-mode only — play-mode reads the
//!   gameplay camera from the ECS), translates pointer events into
//!   camera deltas, walks the [`Scene`] each frame to build the
//!   instance list, and hands that off to the renderer in a closure
//!   scheduled via `cx.submit_wgpu_pass(...)`.
//!
//! When `ocarina-runtime` (Phase 6) lands, the runtime instantiates
//! `ViewportRenderer` itself for shipped builds with no editor deps.
//! Same renderer code, no divergence between editor preview and
//! shipped game.

mod camera;
mod gizmo;
mod mesh;
mod renderer;
mod transform_io;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

// Force-link ocarina-physics so its `#[derive(Component)]`
// submissions (Collider / RigidBody / Velocity) reach the linker's
// inventory section. Without this `use`, the editor binary that
// pulls ocarina-viewport in would still see no physics components
// registered — the loader would reject `[collider]` / `[rigid_body]`
// sections and the surrounding entity would lose its Transform and
// vanish from the viewport.
use ocarina_physics as _;

use bevy_ecs::entity::Entity;
use floem::{
    View,
    action::exec_after,
    event::{Event, EventListener},
    kurbo::{Point, Rect, Size},
    reactive::{RwSignal, SignalGet, SignalUpdate, SignalWith},
    style::CursorStyle,
    views::{Decorators, canvas},
};
use floem_renderer::WgpuPassCtx;
use glam::{EulerRot, Mat4, Quat, Vec3, Vec4};
use ocarina_core::components::{Material, Mesh, Sky, Transform};
use ocarina_core::AssetPath;
use ocarina_scene::Scene;

pub use camera::OrbitCamera;
pub use gizmo::Axis;
pub use renderer::{EntityInstance, GizmoOverlay, RenderMode, ViewportCamera, ViewportRenderer};

use gizmo::GizmoDrag;

/// Reactive handle to the loaded scene. `None` means no project is
/// open. Stored in `RwSignal` so panels (viewport, file explorer,
/// lens) re-render reactively when the scene mutates.
pub type SceneSignal = RwSignal<Option<Rc<RefCell<Scene>>>>;

/// Reactive handle to the currently selected entity + the folder it
/// corresponds to on disk. **One source of truth** shared between
/// the file explorer and the viewport — selecting in either updates
/// this signal, and both panels read it for highlight rendering.
pub type SelectionSignal = RwSignal<Option<ViewportSelection>>;

/// A selection that knows both its ECS identity (for the viewport
/// to highlight) and its on-disk identity (for the file explorer to
/// highlight + scroll-to). Filesystem-as-truth: if either drifts
/// from the other, the signal-set is out of sync.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportSelection {
    pub entity: Entity,
    pub folder: PathBuf,
}

/// Construct an empty signal pair on workspace start. Caller wires
/// these into both the viewport panel (via `viewport_panel`) and the
/// file explorer (via the selection-sync hook).
#[must_use]
pub fn make_signals() -> (SceneSignal, SelectionSignal) {
    // `new` (UnsyncStorage default) because the contents
    // (Rc<RefCell<Scene>>) are !Send, and the panel only ever
    // reads/writes them on the UI thread.
    (RwSignal::new(None), RwSignal::new(None))
}

/// Build a Floem `View` for the viewport panel. Renders entities from
/// `scene_signal`'s scene (if any) with selection from
/// `selection_signal` highlighting the picked entity. Pointer events
/// drive the orbit camera; scroll wheel zooms.
pub fn viewport_panel(
    scene_signal: SceneSignal,
    selection_signal: SelectionSignal,
) -> impl View {
    let state = Rc::new(RefCell::new(PanelState::new()));

    // Drive the viewport at ~60 fps. Floem only repaints a canvas when
    // a tracked signal it accesses inside `paint` changes — and our
    // scene/selection signals only mutate on user actions, so without
    // a continuous tick the panel sits idle. We spin a 16 ms timer that
    // bumps `frame_tick`, the canvas tracks it, and Floem repaints.
    //
    // Filesystem hot-reload also relies on this tick: the scene's
    // contents mutate through interior mutability (`Rc<RefCell<Scene>>`),
    // so the signal *value* doesn't change when files reload — only a
    // continuous repaint surfaces the new state.
    let frame_tick = RwSignal::new(0u64);
    spawn_repaint_loop(state.clone(), frame_tick);

    let canvas_view = canvas({
        let state = state.clone();
        move |cx, size| {
            // Tracked read: any frame_tick.update() triggers another
            // paint. The recursive timer below makes that happen at
            // ~60 fps for as long as the panel exists.
            frame_tick.get();

            // Tracked reads: when the scene or selection signal value
            // itself changes (e.g. workspace open swaps in a new Scene
            // Rc), repaint immediately rather than waiting on the tick.
            let scene = scene_signal.get();
            let selected_entity = selection_signal.get().map(|s| s.entity);

            // Cache the panel's current logical size so the click-pick
            // handler (which only sees pointer position, not size) can
            // convert clicks to UV.
            state.borrow_mut().last_size = size;

            let local_rect = Rect::ZERO.with_size(size);
            let state = state.clone();
            let pass: floem_renderer::WgpuPass = Box::new(move |ctx: WgpuPassCtx<'_>| {
                let mut state = state.borrow_mut();
                state.render_frame(&ctx, scene.as_ref(), selected_entity);
            });
            // Pre-paint: the viewport draws *under* the UI so
            // overlays (palettes, dropdowns, popovers) paint cleanly
            // on top instead of being overdrawn by the 3D scene.
            // The first pre-paint pass each frame is responsible for
            // clearing the swapchain (the viewport's lit pass uses
            // `LoadOp::Clear` for this).
            cx.submit_pre_paint_wgpu_pass(local_rect, pass);
        }
    });

    // Capture the canvas's view id so the PointerDown handler can
    // make a synchronous `request_focus` call (belt-and-suspenders
    // alongside Floem's auto-focus on click for `keyboard_navigable`
    // views).
    let canvas_id = canvas_view.id();

    canvas_view
    // `keyboard_navigable()` is the load-bearing piece here:
    //   - It registers our id in `app_state.keyboard_navigable`.
    //   - During PointerDown dispatch, Floem's `update_focus` auto-
    //     focuses the deepest keyboard_navigable view at the click
    //     point — first claim wins (because focus is taken to None
    //     at the start of PointerDown). Our canvas is a leaf, so it
    //     wins against any ancestor focus claims (editor, terminal).
    //   - Without this flag, KeyDown events won't route here even if
    //     `id.request_focus()` is called.
    .keyboard_navigable()
    .style(|s| s.size_full().cursor(CursorStyle::Default))
    .on_event_stop(EventListener::PointerDown, {
        let state = state.clone();
        move |e: &Event| {
            let Event::PointerDown(pe) = e else { return };

            // Belt-and-suspenders: ask for focus explicitly too.
            // Floem's auto-focus already does this for navigable
            // views, but if any wrapper view above us also claims
            // navigable status, this ensures our id wins regardless.
            canvas_id.request_focus();

            // Gizmo handle takes precedence over orbit. Try to start
            // a gizmo drag first; if it lands, suppress the orbit
            // drag_mode so PointerMove drives the entity instead of
            // the camera.
            if pe.button.mouse_button() == floem::pointer::MouseButton::Primary {
                let (camera, size) = {
                    let s = state.borrow();
                    (clone_camera(&s.camera), s.last_size)
                };
                if let Some(sel) = selection_signal.get_untracked() {
                    if let Some(scene_rc) = scene_signal.get_untracked() {
                        let mut scene = scene_rc.borrow_mut();
                        if let Some(drag) = try_start_gizmo_drag(
                            pe.pos, size, &camera, &mut scene, sel.entity,
                        ) {
                            drop(scene);
                            let mut s = state.borrow_mut();
                            s.last_pointer = Some(pe.pos);
                            s.dragged = false;
                            s.drag_mode = None;
                            s.gizmo_drag = Some(drag);
                            return;
                        }
                    }
                }
            }

            let mut s = state.borrow_mut();
            s.last_pointer = Some(pe.pos);
            s.dragged = false;
            s.drag_mode = match pe.button.mouse_button() {
                floem::pointer::MouseButton::Primary => Some(DragMode::Orbit),
                floem::pointer::MouseButton::Secondary
                | floem::pointer::MouseButton::Auxiliary => Some(DragMode::Pan),
                _ => None,
            };
        }
    })
    .on_event_stop(EventListener::PointerMove, {
        let state = state.clone();
        move |e: &Event| {
            let Event::PointerMove(pm) = e else { return };

            // Gizmo drag wins over orbit. Pull the snapshot we need
            // before re-borrowing the scene.
            let (drag, camera, size, sel) = {
                let s = state.borrow();
                (
                    s.gizmo_drag.clone(),
                    clone_camera(&s.camera),
                    s.last_size,
                    selection_signal.get_untracked(),
                )
            };
            if let (Some(drag), Some(sel)) = (drag, sel) {
                if let Some(scene_rc) = scene_signal.get_untracked() {
                    let mut scene = scene_rc.borrow_mut();
                    apply_gizmo_drag(&drag, pm.pos, size, &camera, &mut scene, sel.entity);
                }
                state.borrow_mut().last_pointer = Some(pm.pos);
                return;
            }

            let mut s = state.borrow_mut();
            if let (Some(prev), Some(mode)) = (s.last_pointer, s.drag_mode) {
                let dx = (pm.pos.x - prev.x) as f32;
                let dy = (pm.pos.y - prev.y) as f32;
                // 2 px deadzone before a press counts as a drag —
                // anything tighter classifies a steady-hand click as
                // an orbit and never picks.
                if dx.abs() + dy.abs() > 2.0 {
                    s.dragged = true;
                }
                match mode {
                    DragMode::Orbit => s.camera.orbit(dx, dy),
                    DragMode::Pan => s.camera.pan(dx, dy),
                }
            }
            s.last_pointer = Some(pm.pos);
        }
    })
    .on_event_stop(EventListener::PointerUp, {
        let state = state.clone();
        move |e: &Event| {
            let Event::PointerUp(pe) = e else { return };

            // Releasing a gizmo drag: persist the entity's current
            // Transform to disk, clear drag state, and skip the
            // click-pick path entirely.
            let was_gizmo_drag = state.borrow().gizmo_drag.is_some();
            if was_gizmo_drag {
                let sel = selection_signal.get_untracked();
                if let (Some(sel), Some(scene_rc)) = (sel, scene_signal.get_untracked()) {
                    let scene = scene_rc.borrow();
                    let transform = scene
                        .world
                        .get::<Transform>(sel.entity)
                        .copied();
                    drop(scene);
                    if let Some(t) = transform {
                        persist_gizmo_drag(&sel.folder, t);
                    }
                }
                state.borrow_mut().gizmo_drag = None;
                return;
            }

            let was_primary_click = pe.button.mouse_button()
                == floem::pointer::MouseButton::Primary
                && !state.borrow().dragged;
            // Always release the drag handles even if we end up picking.
            state.borrow_mut().drag_mode = None;
            if !was_primary_click {
                return;
            }
            // Resolve scene + camera + size, then run the pick out of
            // band — `pick_at` borrow_mut's the scene and we don't want
            // to nest that under PanelState's borrow.
            let scene_rc = scene_signal.get_untracked();
            let (camera_state, size) = {
                let s = state.borrow();
                (clone_camera(&s.camera), s.last_size)
            };
            let Some(scene_rc) = scene_rc else { return };
            let mut scene = scene_rc.borrow_mut();
            let pick = pick_at(pe.pos, size, &camera_state, &mut scene);
            drop(scene);
            match pick {
                Some((entity, folder)) => {
                    selection_signal.set(Some(ViewportSelection { entity, folder }));
                }
                None => {
                    // Click on empty space clears the selection.
                    selection_signal.set(None);
                }
            }
        }
    })
    .on_event_stop(EventListener::PointerWheel, {
        let state = state.clone();
        move |e: &Event| {
            let Event::PointerWheel(pw) = e else { return };
            // PointerWheelEvent.delta is a Vec2 (logical pixels for
            // pixel-delta wheels, ~1 unit per detent for line-delta).
            // Sum-then-clamp gives a stable feel across both modes.
            let dy = pw.delta.y as f32;
            if dy.abs() > 1e-3 {
                state.borrow_mut().camera.zoom(dy);
            }
        }
    })
    .on_event_stop(EventListener::KeyDown, {
        let state = state.clone();
        move |e: &Event| {
            let Event::KeyDown(ke) = e else { return };
            // KeyDown only flips the held-state flag; `spawn_repaint_loop`
            // does the actual integration each frame so multiple held
            // keys compose into one direction vector and movement
            // doesn't depend on the OS key-repeat rate.
            if let Some(slot) = held_key_slot(&mut state.borrow_mut().held_keys, ke) {
                *slot = true;
            }
        }
    })
    .on_event_stop(EventListener::KeyUp, {
        let state = state.clone();
        move |e: &Event| {
            let Event::KeyUp(ke) = e else { return };
            if let Some(slot) = held_key_slot(&mut state.borrow_mut().held_keys, ke) {
                *slot = false;
            }
        }
    })
}

/// Map a key event to the corresponding `HeldKeys` field. Returns
/// `None` for keys we don't track.
fn held_key_slot<'a>(
    keys: &'a mut HeldKeys,
    ke: &floem::keyboard::KeyEvent,
) -> Option<&'a mut bool> {
    use floem::keyboard::{KeyCode, PhysicalKey};
    match ke.key.physical_key {
        PhysicalKey::Code(KeyCode::KeyW) => Some(&mut keys.w),
        PhysicalKey::Code(KeyCode::KeyA) => Some(&mut keys.a),
        PhysicalKey::Code(KeyCode::KeyS) => Some(&mut keys.s),
        PhysicalKey::Code(KeyCode::KeyD) => Some(&mut keys.d),
        PhysicalKey::Code(KeyCode::KeyQ) => Some(&mut keys.q),
        PhysicalKey::Code(KeyCode::KeyE) => Some(&mut keys.e),
        _ => None,
    }
}

/// Per-frame loop. Recursively schedules itself every ~16 ms via
/// `exec_after` (one-shot timer that re-arms each tick).
///
/// Two responsibilities:
/// 1. Bump `frame_tick` so the canvas repaints (the canvas reads it
///    as a tracked signal — see `viewport_panel`).
/// 2. Sample held WASD/QE keys and integrate camera movement. This
///    is the *real* input loop that game engines use: held keys are
///    polled at frame rate, normalized into a direction vector, and
///    applied with `speed × dt`. KeyDown/KeyUp only flip flags; the
///    actual motion happens here, so multiple keys compose cleanly
///    (W+D = diagonal, no key-repeat artefacts, no OS dependency).
fn spawn_repaint_loop(state: Rc<RefCell<PanelState>>, frame_tick: RwSignal<u64>) {
    const FRAME_DT: f32 = 0.016; // ~60 Hz timer cadence
    exec_after(Duration::from_millis(16), move |_| {
        // ── Movement integration ──────────────────────────────────
        {
            let mut s = state.borrow_mut();
            let keys = s.held_keys;
            let dir_right = (keys.d as i32 - keys.a as i32) as f32;
            let dir_up = (keys.e as i32 - keys.q as i32) as f32;
            let dir_forward = (keys.w as i32 - keys.s as i32) as f32;
            if dir_right != 0.0 || dir_up != 0.0 || dir_forward != 0.0 {
                // Normalize so diagonal isn't faster than axis-aligned
                // motion. Scale speed by camera distance so a
                // far-out viewport zooms across the scene at the
                // same on-screen rate as a close-in one.
                let dir = Vec3::new(dir_right, dir_up, dir_forward).normalize_or_zero();
                let speed = (s.camera.distance * 1.5).max(1.0); // world units / sec
                let v = dir * speed * FRAME_DT;
                s.camera.move_target(v.z, v.x, v.y);
            }
        }

        // ── Repaint trigger ───────────────────────────────────────
        frame_tick.update(|v| *v = v.wrapping_add(1));
        spawn_repaint_loop(state.clone(), frame_tick);
    });
}

/// Per-panel mutable state. Cloned-Rc-shared between paint() (which
/// reads camera + scene) and the wgpu render closure (which reads
/// renderer state and updates depth target on resize).
struct PanelState {
    /// Lazily initialized on the first wgpu render closure (when we
    /// first see a wgpu Device + Queue + swapchain format).
    renderer: Option<ViewportRenderer>,
    /// Edit-mode orbit camera. Pointer events update it.
    camera: OrbitCamera,
    /// Current drag (if any) — primary = orbit, middle/secondary = pan.
    drag_mode: Option<DragMode>,
    /// Last pointer position (logical pixels) for delta computation.
    last_pointer: Option<Point>,
    /// Panel size in logical pixels, refreshed each paint. Read by the
    /// click-pick path (which needs to convert a pointer position into
    /// a normalized UV for the camera ray).
    last_size: Size,
    /// Set true the first time the pointer moves with a `drag_mode`
    /// active. Distinguishes "primary click" (no movement → pick) from
    /// "primary drag" (orbit). Reset on every PointerDown.
    dragged: bool,
    /// Active drag on a gizmo handle, or `None` if not currently
    /// manipulating. While `Some`, pointer events drive the entity's
    /// Transform instead of the orbit camera.
    gizmo_drag: Option<GizmoDrag>,
    /// Currently-held navigation keys. Set on KeyDown, cleared on
    /// KeyUp; sampled each frame by `spawn_repaint_loop` to drive
    /// continuous camera movement (FPS-style: hold to move, release
    /// to stop, multiple held simultaneously sum into one direction).
    held_keys: HeldKeys,
    /// Editor render toggles (wireframe, lighting, custom shaders).
    /// Defaults to "everything looks like the shipped game"; the
    /// editor surfaces UI to flip individual flags.
    render_config: ocarina_render::editor::RenderConfig,
}

#[derive(Default, Clone, Copy)]
struct HeldKeys {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    q: bool,
    e: bool,
}

#[derive(Clone, Copy)]
enum DragMode {
    Orbit,
    Pan,
}

impl PanelState {
    fn new() -> Self {
        Self {
            renderer: None,
            camera: OrbitCamera::new(glam::Vec3::ZERO, 6.0),
            drag_mode: None,
            last_pointer: None,
            last_size: Size::ZERO,
            dragged: false,
            gizmo_drag: None,
            held_keys: HeldKeys::default(),
            render_config: ocarina_render::editor::RenderConfig::default(),
        }
    }

    fn ensure_renderer(&mut self, ctx: &WgpuPassCtx<'_>) {
        if let Some(renderer) = &self.renderer {
            if renderer.matches_target_format(ctx.format) {
                return;
            }
            // Format changed (rare): drop old renderer; will be
            // rebuilt below.
            self.renderer = None;
        }
        match ViewportRenderer::from_existing(
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.adapter,
            ctx.format,
        ) {
            Ok(r) => self.renderer = Some(r),
            Err(e) => {
                log::error!("ocarina-viewport: renderer init failed: {e}");
            }
        }
    }

    fn render_frame(
        &mut self,
        ctx: &WgpuPassCtx<'_>,
        scene: Option<&Rc<RefCell<Scene>>>,
        selected_entity: Option<Entity>,
    ) {
        self.ensure_renderer(ctx);
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        // Build entity instance list + scene lights + sky + active
        // scene shader + gizmo overlay if anything is selected. All
        // five walk the same world / world_transforms so we share
        // the borrow.
        let (entities, lights, sky, scene_shader, gizmo_origin, project_root) = scene
            .map(|scene| {
                let mut scene = scene.borrow_mut();
                let xforms = scene.world_transforms();
                let origin = selected_entity
                    .and_then(|e| xforms.get(&e).copied())
                    .map(|m| m.w_axis.truncate());
                let project_root = scene.root_dir.clone();
                let world = &mut scene.world;
                let entities = build_entity_instances_from(&xforms, world);
                let lights =
                    ocarina_render::collect_lights_from_world(world, &xforms);
                // First entity with a Sky component wins. There's no
                // ordering rule beyond entity-id; if the project
                // ships multiple Sky entities the renderer just
                // picks one and ignores the rest.
                let sky = {
                    use bevy_ecs::entity::Entity;
                    let mut q = world.query::<(Entity, &Sky)>();
                    q.iter(world).next().map(|(_, s)| *s)
                };
                let scene_shader = world
                    .get_resource::<ocarina_render::RenderSettings>()
                    .map(|s| s.default_shader.clone())
                    .unwrap_or_else(|| {
                        AssetPath::new(ocarina_render::ENGINE_SHADER_DEFAULT)
                    });
                (entities, lights, sky, scene_shader, origin, Some(project_root))
            })
            .unwrap_or_else(|| {
                (
                    Vec::new(),
                    Vec::new(),
                    None,
                    AssetPath::new(ocarina_render::ENGINE_SHADER_DEFAULT),
                    None,
                    None,
                )
            });

        // Keep the mesh registry's project root in sync with the
        // current scene so user OBJs resolve to the right files.
        if let Some(root) = project_root {
            renderer.set_project_root(root);
        }

        // View-projection from the orbit camera, aspect from the
        // panel rect (which `WgpuPassCtx::rect` carries in pixels).
        let aspect = (ctx.rect.width() / ctx.rect.height().max(1.0)) as f32;
        let view_proj = self.camera.view_projection(aspect);
        let camera = ViewportCamera {
            view_proj,
            eye_pos: self.camera.position(),
        };

        let gizmo = gizmo_origin.map(|origin| GizmoOverlay {
            origin,
            scale_world: gizmo_scale(&self.camera),
            active_axis: self.gizmo_drag.as_ref().map(|d| d.axis()),
        });

        let mode = RenderMode::Edit { selected_entity };
        renderer.render(
            &entities,
            &lights,
            sky,
            &scene_shader,
            ctx,
            camera,
            mode,
            gizmo,
            self.render_config,
        );
    }
}

/// World-space gizmo size derived from the camera's distance to the
/// orbit target — keeps the handles roughly constant on-screen as the
/// camera dollies in / out. Clamped at the bottom so handles don't
/// disappear when zoomed in close.
fn gizmo_scale(camera: &OrbitCamera) -> f32 {
    (camera.distance * 0.18).max(0.25)
}

/// Snapshot the orbit camera's mutable state so the pick path can
/// reuse it without borrowing PanelState across the scene mutation.
fn clone_camera(c: &OrbitCamera) -> OrbitCamera {
    OrbitCamera {
        target: c.target,
        distance: c.distance,
        yaw: c.yaw,
        pitch: c.pitch,
        fov_y: c.fov_y,
        near: c.near,
        far: c.far,
    }
}

/// Build a world-space pick ray through `pos_logical` (panel pixels)
/// using the current camera and panel size. Returns `(origin, dir)`.
fn pointer_ray(
    pos_logical: Point,
    size_logical: Size,
    camera: &OrbitCamera,
) -> Option<(Vec3, Vec3)> {
    if size_logical.width <= 0.0 || size_logical.height <= 0.0 {
        return None;
    }
    let aspect = (size_logical.width / size_logical.height) as f32;
    let u = (pos_logical.x / size_logical.width).clamp(0.0, 1.0) as f32;
    let v = (pos_logical.y / size_logical.height).clamp(0.0, 1.0) as f32;
    Some(camera.ray_through_uv(u, v, aspect))
}

fn euler_deg_to_quat(deg: Vec3) -> Quat {
    Quat::from_euler(
        EulerRot::XYZ,
        deg.x.to_radians(),
        deg.y.to_radians(),
        deg.z.to_radians(),
    )
}

fn quat_to_euler_deg(q: Quat) -> Vec3 {
    let (x, y, z) = q.to_euler(EulerRot::XYZ);
    Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
}

/// Try to start a gizmo drag: ray-test against both the translate
/// axes and the rotate rings around the selected entity's world
/// position. Returns the drag state for whichever handle was hit, or
/// `None` if the click missed every handle (caller falls back to
/// orbit / scene click-pick).
///
/// Translate handles are checked first; ties go to translate since
/// the user usually grabs the central axis bar before reaching the
/// surrounding ring.
fn try_start_gizmo_drag(
    pos: Point,
    size: Size,
    camera: &OrbitCamera,
    scene: &mut Scene,
    entity: Entity,
) -> Option<GizmoDrag> {
    let (ray_o, ray_d) = pointer_ray(pos, size, camera)?;
    let xforms = scene.world_transforms();
    let world_xform = xforms.get(&entity).copied()?;
    let origin = world_xform.w_axis.truncate();
    let scale = gizmo_scale(camera);
    let transform = scene.world.get::<Transform>(entity).copied()?;

    if let Some(axis) = gizmo::pick_translate_axis(origin, ray_o, ray_d, scale) {
        let axis_param =
            gizmo::closest_point_on_axis(origin, axis.dir(), ray_o, ray_d)?;
        return Some(GizmoDrag::Translate {
            axis,
            start_position: transform.position,
            start_rotation: transform.rotation,
            start_scale: transform.scale,
            start_axis_param: axis_param,
        });
    }

    if let Some(axis) = gizmo::pick_rotate_axis(origin, ray_o, ray_d, scale) {
        let angle = gizmo::rotate_drag_angle(origin, axis, ray_o, ray_d)?;
        return Some(GizmoDrag::Rotate {
            axis,
            start_position: transform.position,
            start_quat: euler_deg_to_quat(transform.rotation),
            start_scale: transform.scale,
            start_angle: angle,
        });
    }

    None
}

/// Apply an in-progress gizmo drag to the entity's local Transform.
/// Mutates only `Transform.position` (translate) or `Transform.rotation`
/// (rotate); other fields remain at their drag-start snapshot.
///
/// Operates in *world-axis* space. For an entity with a non-identity
/// parent rotation/scale, the axis bars / rings still align to world
/// X/Y/Z; the change is then expressed as if the parent were identity.
/// Good enough for v1 — the common case is unrotated parents.
fn apply_gizmo_drag(
    drag: &GizmoDrag,
    pos: Point,
    size: Size,
    camera: &OrbitCamera,
    scene: &mut Scene,
    entity: Entity,
) {
    let Some((ray_o, ray_d)) = pointer_ray(pos, size, camera) else {
        return;
    };
    // Origin = world-space center of the gizmo. Capture once for the
    // duration of the drag — re-reading from `world_transforms` mid-
    // drag would shift as we mutate the entity.
    let xforms = scene.world_transforms();
    let Some(world_xform) = xforms.get(&entity).copied() else {
        return;
    };
    let origin = world_xform.w_axis.truncate();

    match drag {
        GizmoDrag::Translate {
            axis,
            start_position,
            start_axis_param,
            ..
        } => {
            let Some(param) =
                gizmo::closest_point_on_axis(origin, axis.dir(), ray_o, ray_d)
            else {
                return;
            };
            let delta = param - *start_axis_param;
            let new_position = *start_position + axis.dir() * delta;
            if let Some(mut t) = scene.world.get_mut::<Transform>(entity) {
                t.position = new_position;
            }
        }
        GizmoDrag::Rotate {
            axis,
            start_quat,
            start_angle,
            ..
        } => {
            let Some(angle) = gizmo::rotate_drag_angle(origin, *axis, ray_o, ray_d) else {
                return;
            };
            let delta = angle - *start_angle;
            let new_quat = Quat::from_axis_angle(axis.dir(), delta) * *start_quat;
            if let Some(mut t) = scene.world.get_mut::<Transform>(entity) {
                t.rotation = quat_to_euler_deg(new_quat);
            }
        }
    }
}

/// Persist the entity's current Transform to disk. Looks up the
/// `[transform]`-bearing file in the entity's folder; if none exists,
/// silently skips (the user dragged a transform that isn't backed by
/// a single file — currently uncommon, but valid).
fn persist_gizmo_drag(folder: &std::path::Path, transform: Transform) {
    let Some(path) = transform_io::find_transform_file(folder) else {
        log::warn!(
            "ocarina-viewport: no [transform] file in {} — gizmo edit not saved",
            folder.display()
        );
        return;
    };
    if let Err(e) = transform_io::write_transform(
        &path,
        transform.position,
        transform.rotation,
        transform.scale,
    ) {
        log::warn!("ocarina-viewport: write_transform({}): {e}", path.display());
    }
}

/// Cast a ray through `pos_logical` and return the closest hit entity
/// + its folder path, if any. Hit-tests every entity that has a world
/// transform against the unit cube `[-0.5, 0.5]^3` in its local frame
/// — matches the cube placeholder the renderer draws for each entity.
fn pick_at(
    pos_logical: Point,
    size_logical: Size,
    camera: &OrbitCamera,
    scene: &mut Scene,
) -> Option<(Entity, PathBuf)> {
    if size_logical.width <= 0.0 || size_logical.height <= 0.0 {
        return None;
    }
    let aspect = (size_logical.width / size_logical.height) as f32;
    let u = (pos_logical.x / size_logical.width).clamp(0.0, 1.0) as f32;
    let v = (pos_logical.y / size_logical.height).clamp(0.0, 1.0) as f32;
    let (origin, dir) = camera.ray_through_uv(u, v, aspect);

    let mut best: Option<(f32, Entity)> = None;
    for (entity, world_xform) in scene.world_transforms() {
        let inv = world_xform.inverse();
        let local_origin = inv.transform_point3(origin);
        let local_dir = inv.transform_vector3(dir);
        // ray_aabb's t is in units of |local_dir|; since we transformed
        // both origin and dir through the same matrix, that t is the
        // same parameter as for the world ray and is comparable across
        // entities without renormalizing.
        if let Some(t) = ray_aabb(local_origin, local_dir, Vec3::splat(-0.5), Vec3::splat(0.5)) {
            if t < 0.0 {
                continue;
            }
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, entity));
            }
        }
    }

    let (_, entity) = best?;
    let folder = scene
        .dir_entity
        .iter()
        .find_map(|(p, &e)| (e == entity).then(|| p.clone()))?;
    Some((entity, folder))
}

/// Slab-method ray vs axis-aligned box. Returns the entry distance `t`
/// (in units of `ray_dir`) if the ray hits, otherwise `None`. Negative
/// `t` means the box is behind the ray origin.
fn ray_aabb(ray_origin: Vec3, ray_dir: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv = Vec3::new(
        if ray_dir.x.abs() > 1e-8 { 1.0 / ray_dir.x } else { f32::INFINITY },
        if ray_dir.y.abs() > 1e-8 { 1.0 / ray_dir.y } else { f32::INFINITY },
        if ray_dir.z.abs() > 1e-8 { 1.0 / ray_dir.z } else { f32::INFINITY },
    );
    let t1 = (min - ray_origin) * inv;
    let t2 = (max - ray_origin) * inv;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let near = tmin.x.max(tmin.y).max(tmin.z);
    let far = tmax.x.min(tmax.y).min(tmax.z);
    if far < near.max(0.0) {
        None
    } else {
        Some(near.max(0.0))
    }
}

/// Walk the scene's entities (folding parent transforms via
/// `Scene::world_transforms`), pair each with a renderer-friendly
/// `EntityInstance`. The instance's `mesh` field is `None` for
/// entities that have no `Mesh` component (cameras, audio
/// listeners): the renderer skips them in the lit pass and (in the
/// editor) draws a small selection marker if they're selected.
///
/// Takes the precomputed `world_xforms` map + a `&World` because
/// the caller (render_frame) needs the same map for gizmo-origin
/// lookup, and `Scene::world_transforms` would otherwise be called
/// twice and require &mut Scene each time.
fn build_entity_instances_from(
    world_xforms: &std::collections::HashMap<Entity, Mat4>,
    world: &bevy_ecs::world::World,
) -> Vec<EntityInstance> {
    const DEFAULT_TINT: Vec4 = Vec4::new(0.78, 0.82, 0.90, 1.0);

    let mut instances = Vec::with_capacity(world_xforms.len());
    for (entity, world_xform) in world_xforms {
        let mesh_path = world
            .get::<Mesh>(*entity)
            .map(|m| m.source.clone())
            .filter(|p| !p.is_empty());
        let material = world
            .get::<Material>(*entity)
            .cloned()
            .unwrap_or_default();
        // Mirror albedo into `tint` so the simple-pipeline shaders
        // (mask / gizmo / unlit) and the scene shader land on the
        // same display colour. `DEFAULT_TINT` is the fallback for
        // entities with no Material at all -- a soft cool grey that
        // reads as "unstyled object" rather than blown-out white.
        let tint = if world.get::<Material>(*entity).is_some() {
            material.albedo
        } else {
            DEFAULT_TINT
        };
        instances.push(EntityInstance {
            entity: *entity,
            world_xform: *world_xform,
            tint,
            mesh: mesh_path,
            material,
        });
    }
    instances.sort_by_key(|i| i.entity);
    instances
}

// ─────────────────────────── Camera viewport panel ───────────────────────────

/// Build a Floem `View` that renders the scene from a specific
/// scene-camera entity's perspective. Used by the editor's
/// "per-camera viewport tabs" feature: each `Camera` component in
/// the scene gets its own panel tab, and clicking that tab swaps in
/// this view.
///
/// Locked-down compared to [`viewport_panel`]:
///   - No orbit / pan / scroll camera.
///   - No mesh picking.
///   - No selection markers / gizmo overlay.
///   - No keyboard movement.
///   - Renders in `RenderMode::Play` so the editor highlight tweaks
///     don't apply.
///
/// `entity_bits` is the camera entity's `Entity::to_bits()` value.
/// If the entity has been removed (the parent sync system normally
/// drops the panel tab in that case, but mid-frame races are
/// possible), the canvas paints nothing for that frame.
pub fn camera_viewport_panel(
    scene_signal: SceneSignal,
    entity_bits: u64,
) -> impl View {
    let state = Rc::new(RefCell::new(CameraPanelState::new(entity_bits)));

    let frame_tick = RwSignal::new(0u64);
    spawn_camera_repaint_loop(frame_tick);

    canvas({
        let state = state.clone();
        move |cx, size| {
            frame_tick.get();
            let scene = scene_signal.get();

            state.borrow_mut().last_size = size;

            let local_rect = Rect::ZERO.with_size(size);
            let state = state.clone();
            let pass: floem_renderer::WgpuPass = Box::new(move |ctx: WgpuPassCtx<'_>| {
                let mut state = state.borrow_mut();
                state.render_frame(&ctx, scene.as_ref());
            });
            cx.submit_pre_paint_wgpu_pass(local_rect, pass);
        }
    })
    .style(|s| s.size_full().cursor(CursorStyle::Default))
}

/// Per-tab state for [`camera_viewport_panel`]. Stripped-down
/// compared to [`PanelState`] since there's no editor input to track.
struct CameraPanelState {
    entity_bits: u64,
    renderer: Option<ViewportRenderer>,
    last_size: Size,
}

impl CameraPanelState {
    fn new(entity_bits: u64) -> Self {
        Self {
            entity_bits,
            renderer: None,
            last_size: Size::ZERO,
        }
    }

    fn ensure_renderer(&mut self, ctx: &WgpuPassCtx<'_>) {
        if let Some(renderer) = &self.renderer {
            if renderer.matches_target_format(ctx.format) {
                return;
            }
            self.renderer = None;
        }
        match ViewportRenderer::from_existing(
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.adapter,
            ctx.format,
        ) {
            Ok(r) => self.renderer = Some(r),
            Err(e) => {
                log::error!("ocarina-viewport: camera renderer init failed: {e}");
            }
        }
    }

    fn render_frame(
        &mut self,
        ctx: &WgpuPassCtx<'_>,
        scene: Option<&Rc<RefCell<Scene>>>,
    ) {
        self.ensure_renderer(ctx);
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(scene) = scene else { return };

        // Resolve the camera entity from its bits, gather the scene
        // payload + the camera's world transform / Camera component.
        let target_entity = bevy_ecs::entity::Entity::from_bits(self.entity_bits);
        let (entities, lights, sky, scene_shader, view_proj, eye_pos, project_root) = {
            use ocarina_core::components::{Camera as CameraComp, Transform as TransformComp};
            let mut scene = scene.borrow_mut();
            let xforms = scene.world_transforms();
            let project_root = scene.root_dir.clone();
            let world = &mut scene.world;

            // Resolve the camera entity. The world's `entities()`
            // table is the authoritative source of validity; we
            // use `world.get_entity(...)` to confirm liveness, then
            // pull components.
            let cam_xform = xforms.get(&target_entity).copied();
            let _ = world.get::<TransformComp>(target_entity); // touch transform component import path
            let cam_comp = world.get::<CameraComp>(target_entity).copied();
            let (cam_xform, cam_comp) = match (cam_xform, cam_comp) {
                (Some(x), Some(c)) => (x, c),
                _ => return,
            };

            let aspect = (ctx.rect.width() / ctx.rect.height().max(1.0)) as f32;
            let view = cam_xform.inverse();
            let proj = glam::Mat4::perspective_rh(
                cam_comp.fov_y.to_radians(),
                aspect,
                cam_comp.near,
                cam_comp.far,
            );
            let view_proj = proj * view;
            let eye_pos = cam_xform.col(3).truncate();

            let entities = build_entity_instances_from(&xforms, world);
            let lights = ocarina_render::collect_lights_from_world(world, &xforms);
            let sky = {
                use bevy_ecs::entity::Entity;
                let mut q = world.query::<(Entity, &Sky)>();
                q.iter(world).next().map(|(_, s)| *s)
            };
            let scene_shader = world
                .get_resource::<ocarina_render::RenderSettings>()
                .map(|s| s.default_shader.clone())
                .unwrap_or_else(|| AssetPath::new(ocarina_render::ENGINE_SHADER_DEFAULT));
            (
                entities,
                lights,
                sky,
                scene_shader,
                view_proj,
                eye_pos,
                Some(project_root),
            )
        };

        if let Some(root) = project_root {
            renderer.set_project_root(root);
        }

        let camera = ViewportCamera { view_proj, eye_pos };
        // Editor toggles intentionally never reach this path; we
        // always render the "shipped game look" through the scene
        // pipeline. No gizmo overlay either -- there's nothing to
        // pick / select / move from this view.
        let render_config = ocarina_render::editor::RenderConfig::default();
        renderer.render(
            &entities,
            &lights,
            sky,
            &scene_shader,
            ctx,
            camera,
            RenderMode::Play,
            None,
            render_config,
        );
    }
}

/// Recursive 16 ms tick that drives the camera viewport's repaints.
/// No movement / input integration -- the camera viewport is a pure
/// observer of scene state.
fn spawn_camera_repaint_loop(frame_tick: RwSignal<u64>) {
    exec_after(Duration::from_millis(16), move |_| {
        frame_tick.update(|v| *v = v.wrapping_add(1));
        spawn_camera_repaint_loop(frame_tick);
    });
}

/// Walk the scene's `World` for `Camera` component entities and
/// return their `Entity::to_bits()` ids (sorted, stable). Used by
/// the editor's per-camera panel-tab sync to know which `CameraView`
/// tabs should currently be present.
///
/// Returns an empty vec when no scene is loaded.
pub fn collect_camera_entity_bits(scene_signal: &SceneSignal) -> Vec<u64> {
    use ocarina_core::components::Camera;
    scene_signal.with_untracked(|maybe_scene| {
        maybe_scene
            .as_ref()
            .map(|scene| {
                let mut scene = scene.borrow_mut();
                let world = &mut scene.world;
                let mut q = world.query::<(Entity, &Camera)>();
                let mut bits: Vec<u64> =
                    q.iter(world).map(|(e, _)| e.to_bits()).collect();
                bits.sort_unstable();
                bits
            })
            .unwrap_or_default()
    })
}
