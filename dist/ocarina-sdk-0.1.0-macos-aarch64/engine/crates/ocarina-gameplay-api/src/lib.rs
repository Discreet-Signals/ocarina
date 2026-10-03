//! Public API for Ocarina gameplay code.
//!
//! Two layers:
//!
//! - **Script API** (this module's contents): the high-level surface
//!   gameplay devs write against. Define a struct, `impl Script for
//!   …`, and `#[derive(OcarinaScript)]` with `path = "..."` to attach
//!   it to entities via `[script]` TOML. This is what 99 % of game
//!   code touches.
//! - **FFI** ([`ffi`] submodule): the older C ABI used by the editor's
//!   cdylib hot-reload path. Kept for compatibility with editor
//!   tooling that loads gameplay as a dylib. Static-link consumers
//!   (the runtime, integration tests) ignore it.

pub mod ffi;

use std::any::Any;

pub use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

pub use ocarina_core::{AssetPath, Input, Key, MouseButton, Time};

// Re-export so the `OcarinaScript` derive can reach inventory through
// the same crate path user code already imports.
pub use inventory;

// Re-export the derive so user code only needs one import:
//   use ocarina_gameplay_api::*;
//   #[derive(OcarinaScript)] #[ocarina_script(path = "...")] ...
pub use ocarina_derive::OcarinaScript;

// ─────────────────────────── Script trait ───────────────────────────

/// The single trait every gameplay script implements.
///
/// `start` runs once when the script is first attached to an entity
/// (after the entity is spawned and components are loaded). `tick`
/// runs every frame while the script remains attached. Both default
/// to no-ops so each script can override only what it needs.
///
/// Scripts hold their own per-instance state in the struct's fields.
/// Cross-script state lives on components in the world.
pub trait Script: Any + Send + Sync + 'static {
    /// Called once, after the script is constructed and attached. Use
    /// this to read initial values off the entity or spawn helper
    /// entities.
    fn start(&mut self, _ctx: &mut Context<'_>, _entity: Entity) {}

    /// Called every frame. The bulk of gameplay lives here.
    fn tick(&mut self, _ctx: &mut Context<'_>, _entity: Entity) {}
}

// ─────────────────────────── Context ───────────────────────────

/// What every script call receives. Holds mutable access to the world
/// plus per-frame `Time` and `Input`. Lifetimes are scoped to a single
/// `start` / `tick` call.
pub struct Context<'a> {
    pub world: &'a mut World,
    pub time: &'a Time,
    pub input: &'a Input,
}

impl<'a> Context<'a> {
    pub fn new(world: &'a mut World, time: &'a Time, input: &'a Input) -> Self {
        Self {
            world,
            time,
            input,
        }
    }

    // ── Convenience: components ──

    /// Borrow a component on `entity`. Returns `None` if the entity
    /// doesn't carry that component.
    #[must_use]
    pub fn get<T: bevy_ecs::component::Component>(&self, entity: Entity) -> Option<&T> {
        self.world.get::<T>(entity)
    }

    /// Mutably borrow a component on `entity`. Triggers change
    /// detection just by being called -- same semantics as
    /// `World::get_mut`.
    pub fn get_mut<T>(&mut self, entity: Entity) -> Option<bevy_ecs::change_detection::Mut<'_, T>>
    where
        T: bevy_ecs::component::Component<Mutability = bevy_ecs::component::Mutable>,
    {
        self.world.get_mut::<T>(entity)
    }

    /// Spawn a fresh entity carrying `bundle`'s components. Returns
    /// the new entity id.
    pub fn spawn<B: bevy_ecs::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        self.world.spawn(bundle).id()
    }

    /// Despawn `entity` if it exists. No-op if it's already gone.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.world.despawn(entity)
    }

    // ── Convenience: lookups ──

    /// First entity whose `Name` matches. O(N) scan; fine for the
    /// "find the player on startup" case, not for per-frame queries
    /// (cache the entity in your script's state instead).
    pub fn find_by_name(&mut self, name: &str) -> Option<Entity> {
        let mut q = self
            .world
            .query::<(Entity, &ocarina_core::components::Name)>();
        q.iter(self.world)
            .find(|(_, n)| n.value == name)
            .map(|(e, _)| e)
    }
}

// ─────────────────────────── Registration ───────────────────────────

/// One row in the script registry. `#[derive(OcarinaScript)]` emits
/// these via `inventory::submit!`. The runtime picks them up and maps
/// `[script] path = "..."` references to the right constructor.
pub struct ScriptRegistration {
    /// Project-relative path the `[script]` TOML field references --
    /// e.g. `"scripts/spinner.rs"`. Match is exact.
    pub path: &'static str,
    /// Fully qualified Rust type path (informational; surfaced in
    /// editor diagnostics).
    pub rust_type: &'static str,
    /// Build a fresh instance of the script. Called once per attach.
    pub construct: fn() -> Box<dyn Script>,
}

inventory::collect!(ScriptRegistration);

/// Iterate every registered script. Used by the runtime at startup
/// and by tooling that wants to enumerate available scripts.
pub fn iter_scripts() -> impl Iterator<Item = &'static ScriptRegistration> {
    inventory::iter::<ScriptRegistration>.into_iter()
}

/// Look up a registration by path. Returns `None` if no script with
/// that path is linked into the current binary.
#[must_use]
pub fn find_script(path: &str) -> Option<&'static ScriptRegistration> {
    iter_scripts().find(|r| r.path == path)
}

// ─────────────────────────── Re-exports ───────────────────────────
//
// Gameplay code that writes `use ocarina_gameplay_api::*;` should be
// able to reach the most common engine types without a second `use`.

pub use ocarina_core::components::{Camera, Material, Mesh, Name, Transform};
pub use ocarina_core::math::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
