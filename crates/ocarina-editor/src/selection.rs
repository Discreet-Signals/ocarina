//! Editor-wide selection state.
//!
//! The viewport selects entities via clicks; other panes (lens panel,
//! file browser highlighting) want to read this selection. Held on the
//! [`OcarinaApp`] so panes can borrow it.

use std::path::PathBuf;

use bevy_ecs::entity::Entity;

#[derive(Debug, Default, Clone)]
pub struct Selection {
    /// Entity currently picked in the viewport, if any.
    pub entity: Option<Entity>,
    /// Best-effort path to the .toml file declaring this entity's
    /// `[transform]`. Cached when an entity is selected so the gizmo
    /// can splice without scanning every drag-end.
    pub transform_file: Option<PathBuf>,
}

impl Selection {
    pub fn clear(&mut self) {
        self.entity = None;
        self.transform_file = None;
    }
}
