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
mod mesh;
mod renderer;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use bevy_ecs::entity::Entity;
use floem::{
    View,
    action::exec_after,
    event::{Event, EventListener},
    kurbo::{Point, Rect, Size},
    reactive::{RwSignal, SignalGet, SignalUpdate},
    style::CursorStyle,
    views::{Decorators, canvas},
};
use floem_renderer::WgpuPassCtx;
use glam::{Vec3, Vec4};
use ocarina_core::components::BasicMesh;
use ocarina_scene::Scene;

pub use camera::OrbitCamera;
pub use renderer::{EntityInstance, RenderMode, ViewportCamera, ViewportRenderer};

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
    spawn_repaint_loop(frame_tick);

    canvas({
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
            // PaintCx::submit_wgpu_pass converts the local logical rect
            // to framebuffer pixels (world transform × scale) before
            // forwarding to the renderer.
            cx.submit_wgpu_pass(local_rect, pass);
        }
    })
    .style(|s| s.size_full().cursor(CursorStyle::Default))
    .on_event_stop(EventListener::PointerDown, {
        let state = state.clone();
        move |e: &Event| {
            let Event::PointerDown(pe) = e else { return };
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
}

/// Recursively schedule a `frame_tick.update(...)` every ~16 ms,
/// driving canvas repaints at ~60 fps. `exec_after` is one-shot, so
/// the closure re-arms itself.
///
/// When the surrounding scope is disposed (workspace close, app exit)
/// `frame_tick`'s underlying signal is gone; `update` becomes a no-op
/// internally, so the chain quietly idles instead of misbehaving.
fn spawn_repaint_loop(frame_tick: RwSignal<u64>) {
    exec_after(Duration::from_millis(16), move |_| {
        frame_tick.update(|v| *v = v.wrapping_add(1));
        spawn_repaint_loop(frame_tick);
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

        // Build entity instance list. No scene = render an empty
        // viewport (still clears depth so panel BG shows through).
        let entities: Vec<EntityInstance> = scene
            .map(|scene| {
                let mut scene = scene.borrow_mut();
                build_entity_instances(&mut scene)
            })
            .unwrap_or_default();

        // View-projection from the orbit camera, aspect from the
        // panel rect (which `WgpuPassCtx::rect` carries in pixels).
        let aspect = (ctx.rect.width() / ctx.rect.height().max(1.0)) as f32;
        let view_proj = self.camera.view_projection(aspect);
        let camera = ViewportCamera::standard(view_proj);

        let mode = RenderMode::Edit { selected_entity };
        renderer.render(&entities, ctx, camera, mode);
    }
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
/// `Scene::world_transforms`), pair each with its `BasicMesh.tint` if
/// present, otherwise the default neutral cool grey.
fn build_entity_instances(scene: &mut Scene) -> Vec<EntityInstance> {
    const DEFAULT_TINT: Vec4 = Vec4::new(0.78, 0.82, 0.90, 1.0);

    let world_xforms = scene.world_transforms();

    // Pull tints up front so we don't borrow `scene.world` across
    // the iteration (world_transforms already returned its borrow).
    let world = &mut scene.world;
    let mut instances = Vec::with_capacity(world_xforms.len());
    for (entity, world_xform) in world_xforms {
        let tint = world
            .get::<BasicMesh>(entity)
            .map_or(DEFAULT_TINT, |m| m.tint);
        instances.push(EntityInstance {
            entity,
            world_xform,
            tint,
        });
    }
    instances.sort_by_key(|i| i.entity);
    instances
}
