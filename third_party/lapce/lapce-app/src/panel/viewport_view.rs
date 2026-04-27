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

use floem::{View, prelude::SignalGet, views::Decorators};

use super::position::PanelPosition;
use crate::{
    config::color::LapceColor,
    window_tab::WindowTabData,
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
