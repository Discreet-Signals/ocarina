//! Ocarina 3D viewport panel.
//!
//! v0 stub: paints a placeholder background + label so the panel system
//! plumbing is verified end-to-end. The real viewport — driven by
//! `ocarina-rhi-wgpu` rendering through Floem's shared `wgpu::Device` —
//! lands in a follow-up that wires up `WgpuRhi::from_existing(...)` from
//! a `Canvas` paint callback.

use std::rc::Rc;

use floem::{
    View,
    peniko::Color,
    reactive::SignalGet,
    views::{Decorators, container, label, stack},
};

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
    container(
        stack((
            label(|| "Ocarina Viewport".to_string()).style(|s| {
                s.font_size(14.0)
                    .color(Color::from_rgba8(220, 220, 220, 255))
                    .padding(12.0)
            }),
            label(|| "wgpu device sharing wired; RHI handoff pending".to_string())
                .style(|s| {
                    s.font_size(11.0)
                        .color(Color::from_rgba8(140, 140, 150, 255))
                        .padding_horiz(12.0)
                }),
        ))
        .style(|s| s.flex_col()),
    )
    .style(move |s| {
        s.size_pct(100.0, 100.0)
            .background(config.get().color(LapceColor::EDITOR_BACKGROUND))
            .border_color(config.get().color(LapceColor::LAPCE_BORDER))
    })
    .debug_name("Ocarina Viewport Panel")
}
