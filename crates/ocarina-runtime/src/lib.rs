//! Shippable runtime engine.
//!
//! Owns the live ECS world plus the running script instances. Hosts
//! (the editor in play mode, or the binary produced by `ocarina build`)
//! drive [`Runtime::tick`] each frame and feed input via the
//! `Runtime::input_mut()` accessor.
//!
//! This crate **must not** depend on editor crates -- its dependency
//! tree defines the ship-binary dep surface.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use glam::Mat4;

use ocarina_core::components::Scripts;
use ocarina_core::{Input, Time};
use ocarina_gameplay_api::{find_script, Context, Script};
use ocarina_physics::Physics;
use ocarina_scene::Scene;

pub use ocarina_scene::LoadReport;

#[cfg(feature = "windowed")]
pub mod windowed;

/// One tick of game state. Holds the full `Scene` (so it can compose
/// hierarchical transforms), per-frame `Time` / `Input` resources,
/// and the live script instances keyed by their owning entity.
///
/// Hosts construct this once via [`Runtime::from_project`] (or
/// [`Runtime::from_scene`]) and call [`tick`](Self::tick) every frame.
pub struct Runtime {
    scene: Scene,
    time: Time,
    input: Input,
    physics: Physics,
    /// One vec per entity, parallel to the entity's `Scripts.items`
    /// order. We track them outside the ECS because `Box<dyn Script>`
    /// isn't a component (no Send + Sync + 'static guarantees we want
    /// to make automatic) and instances carry per-frame state. The
    /// `bool` is a "poisoned" flag set if the script panicked in
    /// `start` or `tick`; poisoned scripts are skipped on subsequent
    /// frames so a single buggy script can't bring down the editor or
    /// flood the log loop forever.
    instances: HashMap<Entity, Vec<ScriptSlot>>,
    /// Entities whose script `start()` we've already run.
    started: std::collections::HashSet<Entity>,
}

/// One running script instance plus a poisoned flag. We can't put the
/// flag inside `Box<dyn Script>` (it would force an extension to the
/// public trait), so we wrap.
struct ScriptSlot {
    script: Box<dyn Script>,
    /// Identifier the user wrote in TOML; carried so a panic message
    /// can name the script that broke instead of just "<unknown>".
    path: String,
    /// Set after the first panic from `start` or `tick`. Skipped on
    /// subsequent calls.
    poisoned: bool,
}

impl Runtime {
    /// Build a runtime from an already-loaded scene. Takes ownership
    /// of the scene.
    #[must_use]
    pub fn from_scene(scene: Scene) -> Self {
        let mut rt = Self {
            scene,
            time: Time::default(),
            input: Input::default(),
            physics: Physics::new(),
            instances: HashMap::new(),
            started: Default::default(),
        };
        rt.spawn_script_instances();
        rt
    }

    /// Convenience: open a project directory, load it via
    /// `ocarina-scene`, build a runtime around the resulting world.
    /// Errors on the load are returned in the report; the runtime
    /// is still constructed (with whatever loaded successfully).
    pub fn from_project(root: impl AsRef<Path>) -> (Self, LoadReport) {
        let (scene, report) = ocarina_scene::loader::open(root.as_ref());
        (Self::from_scene(scene), report)
    }

    /// Look up scripts for every entity carrying `Scripts` and
    /// instantiate them via the global registry. Called once at
    /// startup; the editor's play-mode toggle will eventually call
    /// this again on entry.
    fn spawn_script_instances(&mut self) {
        let mut to_construct: Vec<(Entity, Vec<String>)> = Vec::new();
        {
            let world = &mut self.scene.world;
            let mut q = world.query::<(Entity, &Scripts)>();
            for (entity, scripts) in q.iter(world) {
                let paths = scripts
                    .items
                    .iter()
                    .map(|s| s.path.0.clone())
                    .collect::<Vec<_>>();
                if !paths.is_empty() {
                    to_construct.push((entity, paths));
                }
            }
        }

        for (entity, paths) in to_construct {
            let mut built: Vec<ScriptSlot> = Vec::with_capacity(paths.len());
            for path in &paths {
                match find_script(path) {
                    Some(reg) => {
                        // The constructor itself can panic for
                        // misbehaving user scripts. Catch it so one
                        // bad script doesn't take down the editor.
                        let constructed = std::panic::catch_unwind(
                            std::panic::AssertUnwindSafe(reg.construct),
                        );
                        match constructed {
                            Ok(s) => built.push(ScriptSlot {
                                script: s,
                                path: path.clone(),
                                poisoned: false,
                            }),
                            Err(_) => log::error!(
                                "script `{path}` panicked during construction on entity {entity:?} -- disabling for this session"
                            ),
                        }
                    }
                    None => log::warn!(
                        "no script registered for path `{path}` on entity {entity:?} -- skipping"
                    ),
                }
            }
            if !built.is_empty() {
                self.instances.insert(entity, built);
            }
        }
    }

    /// Advance the simulation by `delta`. Runs each script's `start`
    /// the first time we see it, then `tick` for every script every
    /// frame. Input "just" deltas are rolled forward at the end of
    /// the frame.
    pub fn tick(&mut self, delta: Duration) {
        self.time.advance(delta);

        // Scripts run BEFORE physics so a controller can write
        // `Velocity` (or apply impulses, once we have them) and have
        // the next physics step integrate from those values. Reading
        // input + writing intent → physics integration → render is
        // the standard frame order.
        let entities: Vec<Entity> = self.instances.keys().copied().collect();
        for entity in entities {
            let Some(mut scripts) = self.instances.remove(&entity) else {
                continue;
            };
            let started = self.started.contains(&entity);
            // Wrap each `start` / `tick` invocation in `catch_unwind`
            // so a panicking user script logs once and gets poisoned
            // (skipped from then on) instead of bringing down the
            // host. The world it touched may be in a partial state,
            // but that beats process death.
            for s in &mut scripts {
                if s.poisoned {
                    continue;
                }
                if !started {
                    let path = s.path.clone();
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut ctx = Context::new(
                            &mut self.scene.world,
                            &self.time,
                            &mut self.input,
                        );
                        s.script.start(&mut ctx, entity);
                    }));
                    if res.is_err() {
                        log::error!(
                            "script `{path}` panicked in start() on entity {entity:?} -- disabling for this session"
                        );
                        s.poisoned = true;
                        continue;
                    }
                }
                let path = s.path.clone();
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut ctx = Context::new(
                        &mut self.scene.world,
                        &self.time,
                        &mut self.input,
                    );
                    s.script.tick(&mut ctx, entity);
                }));
                if res.is_err() {
                    log::error!(
                        "script `{path}` panicked in tick() on entity {entity:?} -- disabling for this session"
                    );
                    s.poisoned = true;
                }
            }
            if !started {
                self.started.insert(entity);
            }
            self.instances.insert(entity, scripts);
        }

        // Physics integrates one step using whatever scripts just
        // wrote (Velocity, kinematic Transform, etc.) and writes the
        // post-step state back onto the components.
        self.physics.step(&mut self.scene.world, delta.as_secs_f32());

        self.input.end_frame();
    }

    /// World-space transform matrices for every entity that carries a
    /// `Transform` (composed through parent folders). Cheap enough to
    /// call once per frame at editor / shipping-game scales.
    pub fn world_transforms(&mut self) -> HashMap<Entity, Mat4> {
        self.scene.world_transforms()
    }

    // ── Accessors ──

    pub fn world(&self) -> &World {
        &self.scene.world
    }
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.scene.world
    }
    pub fn scene(&self) -> &Scene {
        &self.scene
    }
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }
    pub fn time(&self) -> &Time {
        &self.time
    }
    pub fn input(&self) -> &Input {
        &self.input
    }
    /// Mutable access for hosts to push input events before each tick.
    pub fn input_mut(&mut self) -> &mut Input {
        &mut self.input
    }
    pub fn physics(&self) -> &Physics {
        &self.physics
    }
    pub fn physics_mut(&mut self) -> &mut Physics {
        &mut self.physics
    }
    pub fn entity_count(&self) -> usize {
        self.scene
            .world
            .archetypes()
            .iter()
            .map(|a| a.len() as usize)
            .sum()
    }
    pub fn script_count(&self) -> usize {
        self.instances.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::component::Component;
    use ocarina_core::components::{ScriptRef, Scripts, Transform};
    use ocarina_gameplay_api::{Context, Script, ScriptRegistration};

    /// A test script that ticks a counter on its own component.
    #[derive(Default)]
    struct CountTicks;

    #[derive(Component, Default)]
    struct Counter(u32);

    impl Script for CountTicks {
        fn start(&mut self, ctx: &mut Context<'_>, entity: Entity) {
            // Ensure a Counter exists on first tick.
            if ctx.get::<Counter>(entity).is_none() {
                ctx.world.entity_mut(entity).insert(Counter(0));
            }
        }
        fn tick(&mut self, ctx: &mut Context<'_>, entity: Entity) {
            if let Some(mut c) = ctx.get_mut::<Counter>(entity) {
                c.0 += 1;
            }
        }
    }

    fn construct_count_ticks() -> Box<dyn Script> {
        Box::new(CountTicks)
    }

    ocarina_gameplay_api::inventory::submit! {
        ScriptRegistration {
            path: "scripts/test_count.rs",
            rust_type: "CountTicks",
            construct: construct_count_ticks,
        }
    }

    #[test]
    fn tick_advances_time_and_runs_scripts() {
        let mut world = World::new();
        let entity = world
            .spawn((
                Transform::default(),
                Scripts {
                    items: vec![ScriptRef {
                        path: ocarina_core::AssetPath::new("scripts/test_count.rs"),
                    }],
                },
            ))
            .id();

        // Build a Scene around the world. Scene::empty seeds its own
        // World; we replace it with the test-built one and re-anchor
        // the root to our entity.
        let mut scene = Scene::empty(std::env::temp_dir().join("ocarina-runtime-tick-test"));
        scene.world = world;
        scene.dir_entity.clear();
        scene.dir_entity.insert((*scene.root_dir).clone(), entity);
        scene.root = entity;

        let mut rt = Runtime::from_scene(scene);
        assert_eq!(rt.script_count(), 1);

        rt.tick(Duration::from_millis(16));
        rt.tick(Duration::from_millis(16));
        rt.tick(Duration::from_millis(16));

        let counter = rt.world().get::<Counter>(entity).expect("Counter present");
        assert_eq!(counter.0, 3);
        assert!(rt.time().total > Duration::ZERO);
        assert_eq!(rt.time().frame, 3);
    }
}
