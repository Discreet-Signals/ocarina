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

use bevy_ecs::entity::Entity;
use floem::{
    IntoView,
    event::listener,
    reactive::{RwSignal, SignalGet},
    style::CursorStyle,
    views::{Decorators, canvas},
};
use floem::ui_events::{ScrollDelta, pointer::PointerButton};
// Bring the `Renderer` trait into scope so the `submit_wgpu_pass`
// method we call on `&mut PaintCx` (which derefs to the renderer
// wrapper) is visible.
use floem_renderer::{Renderer as _, WgpuPassCtx};
use glam::Vec4;
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
) -> impl IntoView {
    let state = Rc::new(RefCell::new(PanelState::new()));

    canvas({
        let state = state.clone();
        move |cx, _size| {
            // Compute the panel rect in framebuffer pixels. world_transform
            // includes window scaling per Floem's Renderer trait docs.
            let world_rect = cx.world_transform.transform_rect_bbox(cx.layout_rect_local);

            // Snapshot reactive state BEFORE the closure (which runs
            // later in finish()). The closure owns these values.
            let scene = scene_signal.get_untracked();
            let selected_entity = selection_signal
                .get_untracked()
                .map(|s| s.entity);

            let state = state.clone();
            let pass: floem_renderer::WgpuPass = Box::new(move |ctx: WgpuPassCtx<'_>| {
                let mut state = state.borrow_mut();
                state.render_frame(&ctx, scene.as_ref(), selected_entity);
            });
            cx.submit_wgpu_pass(world_rect, pass);
        }
    })
    .style(|s| s.size_full().cursor(CursorStyle::Default))
    .on_event_stop(listener::PointerDown, {
        let state = state.clone();
        move |_cx, pbe| {
            let mut s = state.borrow_mut();
            s.last_pointer = Some(pbe.state.logical_point());
            s.drag_mode = match pbe.button {
                Some(PointerButton::Primary) => Some(DragMode::Orbit),
                Some(PointerButton::Secondary) | Some(PointerButton::Auxiliary) => {
                    Some(DragMode::Pan)
                }
                _ => None,
            };
        }
    })
    .on_event_stop(listener::PointerMove, {
        let state = state.clone();
        move |_cx, pu| {
            let pos = pu.current.logical_point();
            let mut s = state.borrow_mut();
            if let (Some(prev), Some(mode)) = (s.last_pointer, s.drag_mode) {
                let dx = (pos.x - prev.x) as f32;
                let dy = (pos.y - prev.y) as f32;
                match mode {
                    DragMode::Orbit => s.camera.orbit(dx, dy),
                    DragMode::Pan => s.camera.pan(dx, dy),
                }
            }
            s.last_pointer = Some(pos);
        }
    })
    .on_event_stop(listener::PointerUp, {
        let state = state.clone();
        move |_cx, _pbe| {
            state.borrow_mut().drag_mode = None;
        }
    })
    .on_event_stop(listener::PointerWheel, {
        let state = state.clone();
        move |_cx, pse| {
            let dy = match pse.delta {
                ScrollDelta::LineDelta(_, y) => y,
                ScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                ScrollDelta::PageDelta(_, y) => y * 10.0,
            };
            if dy.abs() > 1e-3 {
                state.borrow_mut().camera.zoom(dy);
            }
        }
    })
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
    last_pointer: Option<floem::kurbo::Point>,
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
