use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

use super::{data::PanelOrder, position::PanelPosition};
use crate::config::icon::LapceIcons;

#[derive(
    Clone, Copy, PartialEq, Serialize, Deserialize, Hash, Eq, Debug, EnumIter,
)]
pub enum PanelKind {
    Terminal,
    FileExplorer,
    SourceControl,
    Plugin,
    Search,
    Problem,
    Debug,
    CallHierarchy,
    DocumentSymbol,
    References,
    Implementation,
    /// Ocarina 3D viewport -- renders the live ECS scene through the
    /// shared wgpu device.
    Viewport,
    /// Ocarina per-camera viewport. The `u64` payload is the
    /// scene-camera entity's `Entity::to_bits()` value; the renderer
    /// re-resolves the entity each frame so a stale id (camera was
    /// removed mid-frame) just renders nothing for one frame before
    /// the sync drops the tab. Tab lifetime is tied to the entity's
    /// `Camera` component: added when the component appears, removed
    /// when it disappears.
    ///
    /// `#[strum(disabled)]` -- `EnumIter` doesn't work for tuple
    /// variants without `Default` and we don't want this to show up
    /// in the "auto-insert all panels" path anyway.
    #[strum(disabled)]
    CameraView(u64),
    ClaudeAgent,
    CodexAgent,
    OpencodeAgent,
}

impl PanelKind {
    pub fn svg_name(&self) -> &'static str {
        match &self {
            PanelKind::Terminal => LapceIcons::TERMINAL,
            PanelKind::FileExplorer => LapceIcons::FILE_EXPLORER,
            PanelKind::SourceControl => LapceIcons::SCM,
            PanelKind::Plugin => LapceIcons::EXTENSIONS,
            PanelKind::Search => LapceIcons::SEARCH,
            PanelKind::Problem => LapceIcons::PROBLEM,
            PanelKind::Debug => LapceIcons::DEBUG,
            PanelKind::CallHierarchy => LapceIcons::TYPE_HIERARCHY,
            PanelKind::DocumentSymbol => LapceIcons::DOCUMENT_SYMBOL,
            PanelKind::References => LapceIcons::REFERENCES,
            PanelKind::Implementation => LapceIcons::IMPLEMENTATION,
            PanelKind::Viewport => LapceIcons::START,
            PanelKind::CameraView(_) => LapceIcons::START,
            PanelKind::ClaudeAgent => LapceIcons::AGENT_CLAUDE,
            PanelKind::CodexAgent => LapceIcons::AGENT_OPENAI,
            PanelKind::OpencodeAgent => LapceIcons::AGENT_OPENCODE,
        }
    }

    pub fn position(&self, order: &PanelOrder) -> Option<(usize, PanelPosition)> {
        for (pos, panels) in order.iter() {
            let index = panels.iter().position(|k| k == self);
            if let Some(index) = index {
                return Some((index, *pos));
            }
        }
        None
    }

    pub fn default_position(&self) -> PanelPosition {
        match self {
            // Terminal moves to the right side per Ocarina default layout.
            PanelKind::Terminal => PanelPosition::RightTop,
            PanelKind::FileExplorer => PanelPosition::LeftTop,
            PanelKind::SourceControl => PanelPosition::LeftTop,
            PanelKind::Plugin => PanelPosition::LeftTop,
            PanelKind::Search => PanelPosition::BottomLeft,
            PanelKind::Problem => PanelPosition::BottomLeft,
            PanelKind::Debug => PanelPosition::LeftTop,
            PanelKind::CallHierarchy => PanelPosition::BottomLeft,
            PanelKind::DocumentSymbol => PanelPosition::RightTop,
            PanelKind::References => PanelPosition::BottomLeft,
            PanelKind::Implementation => PanelPosition::BottomLeft,
            // Viewport sits in the bottom area as a wide 3D scene panel.
            PanelKind::Viewport => PanelPosition::BottomLeft,
            PanelKind::CameraView(_) => PanelPosition::BottomLeft,
            PanelKind::ClaudeAgent => PanelPosition::RightTop,
            PanelKind::CodexAgent => PanelPosition::RightTop,
            PanelKind::OpencodeAgent => PanelPosition::RightTop,
        }
    }
}
