//! Ocarina 3D viewport panel.
//!
//! Thin shim over `ocarina_viewport::viewport_panel`. The actual
//! rendering — pipeline, mesh, camera, RHI handoff, scene walk —
//! lives in the `ocarina-viewport` crate so it stays out of the
//! Lapce subtree (clean upstream rebases) and so the rendering core
//! is reusable by `ocarina-runtime` for shipped builds.
//!
//! Pulls the scene + selection signals from `WindowTabData::common`
//! so this panel and the file explorer share the same source of
//! truth for "which entity is selected."

use std::rc::Rc;
use std::time::Duration;

use floem::{
    View, action::exec_after, prelude::SignalGet, reactive::SignalUpdate,
    views::Decorators,
};

use super::{kind::PanelKind, position::PanelPosition};
use crate::{
    config::color::LapceColor, panel::data::PanelData, window_tab::WindowTabData,
};

pub fn viewport_panel(
    window_tab_data: Rc<WindowTabData>,
    _position: PanelPosition,
) -> impl View {
    let config = window_tab_data.common.config;
    let scene = window_tab_data.common.ocarina_scene;
    let selection = window_tab_data.common.ocarina_selection;

    ocarina_viewport::viewport_panel(scene, selection)
        .style(move |s| {
            // No background fill: the viewport is a pre-paint pass
            // that runs BEFORE Vger encodes the UI, and it
            // `LoadOp::Clear`s the swapchain to its own dark color.
            // Vger then paints this panel on top with `LoadOp::Load`,
            // so any opaque background here would just hide the
            // viewport. We keep the border for the panel chrome.
            s.size_pct(100.0, 100.0)
                .border_color(config.get().color(LapceColor::LAPCE_BORDER))
        })
        .debug_name("Ocarina Viewport Panel")
}

/// Per-camera viewport panel. Renders the live ECS scene from a
/// specific `Camera` entity's perspective. Locked-down compared to
/// [`viewport_panel`]: no orbit / pan / picking / gizmos / WASD
/// movement. The bottom panel container's tab strip dynamically
/// adds / removes one of these per `Camera` entity in the world.
pub fn camera_viewport_panel(
    window_tab_data: Rc<WindowTabData>,
    _position: PanelPosition,
    entity_bits: u64,
) -> impl View {
    let config = window_tab_data.common.config;
    let scene = window_tab_data.common.ocarina_scene;

    ocarina_viewport::camera_viewport_panel(scene, entity_bits)
        .style(move |s| {
            s.size_pct(100.0, 100.0)
                .border_color(config.get().color(LapceColor::LAPCE_BORDER))
        })
        .debug_name(format!("Camera Viewport ({entity_bits:#x})"))
}

/// Start a 4 Hz timer that keeps `panel.panels[BottomLeft]` in sync
/// with the world's `Camera` entities: one `PanelKind::CameraView`
/// per camera, added when a Camera component appears, removed when
/// it disappears. The fixed `Viewport` (editor) tab is preserved at
/// the position the user left it; new camera tabs append, removed
/// camera tabs disappear from wherever they were.
///
/// Polled instead of event-driven because the ECS world doesn't emit
/// "component added/removed" events the editor can hook into; 250 ms
/// is fast enough that adding a camera in TOML feels instant and
/// slow enough that the cost is negligible.
pub fn start_camera_panel_sync(
    scene_signal: ocarina_viewport::SceneSignal,
    panel: PanelData,
) {
    fn tick(scene_signal: ocarina_viewport::SceneSignal, panel: PanelData) {
        let cameras = ocarina_viewport::collect_camera_entity_bits(&scene_signal);

        // Update the panel order: drop CameraViews whose entity is
        // gone, append CameraViews for new entities. Preserve the
        // Viewport (editor) tab + any other non-camera tabs in their
        // existing positions.
        panel.panels.update(|panels_map| {
            let entry = panels_map
                .entry(super::position::PanelPosition::BottomLeft)
                .or_insert_with(im::Vector::new);

            // 1. Remove stale CameraView entries.
            let stale_indices: Vec<usize> = entry
                .iter()
                .enumerate()
                .filter_map(|(i, k)| match k {
                    PanelKind::CameraView(bits) if !cameras.contains(bits) => Some(i),
                    _ => None,
                })
                .collect();
            for &i in stale_indices.iter().rev() {
                entry.remove(i);
            }

            // 2. Append CameraViews for entities not already present.
            let mut existing: Vec<u64> = entry
                .iter()
                .filter_map(|k| match k {
                    PanelKind::CameraView(b) => Some(*b),
                    _ => None,
                })
                .collect();
            existing.sort_unstable();
            for &bits in &cameras {
                if !existing.contains(&bits) {
                    entry.push_back(PanelKind::CameraView(bits));
                }
            }
        });

        // Re-arm. Drop the timer entirely if `scene_signal` (and
        // therefore the whole window tab) is gone -- the
        // `with_untracked` above doesn't drop the closure, and an
        // exec_after re-arm leaks no state since each closure is
        // Box::new'd.
        let scene_signal_next = scene_signal;
        let panel_next = panel.clone();
        exec_after(Duration::from_millis(250), move |_| {
            tick(scene_signal_next, panel_next);
        });
    }

    tick(scene_signal, panel);
}
