//! Per-frame time information.
//!
//! Stored as a `bevy_ecs::Resource` on the runtime's world. Scripts
//! read it through `Context::time`. Both the editor's play mode and
//! the shipped runtime populate it before each tick.

use std::time::Duration;

use bevy_ecs::resource::Resource;

#[derive(Debug, Clone, Copy, Resource)]
pub struct Time {
    /// Time since the previous tick. The first tick reports
    /// `Duration::ZERO` so scripts don't see a giant first-frame jump.
    pub delta: Duration,
    /// Cumulative time since the runtime started.
    pub total: Duration,
    /// Monotonic frame counter. Useful as a cheap "first frame?" check.
    pub frame: u64,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            delta: Duration::ZERO,
            total: Duration::ZERO,
            frame: 0,
        }
    }
}

impl Time {
    /// Convenience: delta in seconds as f32. Most game-tick math wants
    /// this rather than `Duration`.
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    /// Cumulative seconds as f32.
    #[must_use]
    pub fn elapsed(&self) -> f32 {
        self.total.as_secs_f32()
    }

    /// Advance by `delta`. Called by the runtime once per frame; not
    /// meant to be called from scripts.
    pub fn advance(&mut self, delta: Duration) {
        self.delta = delta;
        self.total += delta;
        self.frame += 1;
    }
}
