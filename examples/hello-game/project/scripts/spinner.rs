//! Spinner: rotates its host entity around the Y axis at a constant
//! rate. Demonstrates the script-authoring shape devs work in:
//!
//! - `#[derive(OcarinaScript)]` registers the type so the engine can
//!   construct it when an entity's `[[script]] path = "..."` matches.
//! - `Script::start` runs once when the script is first attached.
//! - `Script::tick` runs every frame; `Context::time.dt()` is the
//!   per-frame delta.
//!
//! Authored at `project/scripts/spinner.rs` — `build.rs` discovers
//! every `*.rs` file in this folder and compiles them into the host
//! binary as modules. The dev never edits engine code; they edit
//! TOML and these scripts.

use ocarina_gameplay_api::{Context, Entity, Name, OcarinaScript, Script, Transform};

#[derive(OcarinaScript, Default)]
#[ocarina_script(path = "scripts/spinner.rs")]
pub struct Spinner {
    /// Total degrees rotated since `start`. Useful for verifying we
    /// keep state across frames; not surfaced to TOML.
    pub accumulated: f32,
}

impl Script for Spinner {
    fn start(&mut self, ctx: &mut Context<'_>, entity: Entity) {
        if let Some(name) = ctx.get::<Name>(entity) {
            log::info!("Spinner started on `{}`", name.value);
        }
    }

    fn tick(&mut self, ctx: &mut Context<'_>, entity: Entity) {
        const DEGREES_PER_SECOND: f32 = 990.0;
        let delta_deg = DEGREES_PER_SECOND * ctx.time.dt();
        self.accumulated += delta_deg;
        if let Some(mut t) = ctx.get_mut::<Transform>(entity) {
            t.rotation.y = (t.rotation.y + delta_deg).rem_euclid(360.0);
        }
    }
}
