//! `rapier3d`-backed physics simulation.
//!
//! [`Physics`] owns rapier's body/collider sets and pipeline state.
//! Each call to [`Physics::step`] reconciles the ECS world with
//! rapier (creating handles for newly-seen entities, copying
//! kinematic transforms in), advances the simulation by `dt`, and
//! copies the solver's results back onto entity `Transform` and
//! `Velocity` components.
//!
//! Engine code never imports from `rapier3d` directly -- all the
//! rapier types are kept inside this module. Components in
//! [`crate::components`] use plain glam types and our own enums, so
//! the rest of the codebase stays decoupled from the solver crate.
//!
//! `rapier3d` 0.32+ uses `glam` natively (via the `glamx` crate) for
//! its math types, so position / velocity exchange is a direct field
//! copy with no per-frame conversion.

use std::collections::HashMap;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use glam::{Quat, Vec3};
use rapier3d::math::{Pose, Vector};
use rapier3d::prelude::*;

use ocarina_core::components::Transform as EcsTransform;

use crate::components::{
    Collider as EcsCollider, ColliderShape, RigidBody as EcsRigidBody, RigidBodyKind, Velocity,
};

/// All rapier state, plus the entity ↔ handle maps that bridge it
/// back to the ECS. One instance lives on the runtime; the editor
/// will own its own at "play" time when that lands.
pub struct Physics {
    pub gravity: Vec3,
    pub integration_parameters: IntegrationParameters,

    rigid_bodies: RigidBodySet,
    colliders: ColliderSet,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pipeline: PhysicsPipeline,

    body_for_entity: HashMap<Entity, RigidBodyHandle>,
    collider_for_entity: HashMap<Entity, ColliderHandle>,
}

impl Default for Physics {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            integration_parameters: IntegrationParameters::default(),
            rigid_bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            pipeline: PhysicsPipeline::new(),
            body_for_entity: HashMap::new(),
            collider_for_entity: HashMap::new(),
        }
    }

    /// Advance the simulation by `dt` seconds. Reconciles ECS state
    /// with rapier first (lazy-creating handles for new entities,
    /// pushing kinematic poses in), then steps, then writes the
    /// solver's result back into `Transform` / `Velocity`.
    pub fn step(&mut self, world: &mut World, dt: f32) {
        self.integration_parameters.dt = dt.max(1.0e-5);

        self.sync_ecs_to_rapier(world);

        let physics_hooks = ();
        let event_handler = ();
        let gravity: Vector = self.gravity;
        self.pipeline.step(
            gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &physics_hooks,
            &event_handler,
        );

        self.sync_rapier_to_ecs(world);
    }

    /// Discard all rapier state. Used by the editor's "stop play"
    /// path so a future "play" rebuilds bodies from the on-disk
    /// world rather than carrying over runtime mutations.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    // ──────────────────────── Internals ────────────────────────

    fn sync_ecs_to_rapier(&mut self, world: &mut World) {
        // 1. Spawn rapier handles for entities we haven't seen yet.
        let mut new_bodies: Vec<(
            Entity,
            EcsTransform,
            Option<EcsRigidBody>,
            EcsCollider,
            Option<Velocity>,
        )> = Vec::new();
        {
            let mut q = world.query::<(
                Entity,
                &EcsTransform,
                Option<&EcsRigidBody>,
                &EcsCollider,
                Option<&Velocity>,
            )>();
            for (entity, transform, body, collider, velocity) in q.iter(world) {
                if self.body_for_entity.contains_key(&entity) {
                    continue;
                }
                new_bodies.push((
                    entity,
                    *transform,
                    body.cloned(),
                    collider.clone(),
                    velocity.cloned(),
                ));
            }
        }
        for (entity, transform, body, collider, velocity) in new_bodies {
            self.create_handle(entity, &transform, body.as_ref(), &collider, velocity.as_ref());
        }

        // 2. Push the ECS view of velocity (and pose, for kinematic
        //    bodies) into rapier so changes scripts wrote this
        //    frame are honored by the next step. Static bodies are
        //    skipped -- they don't move.
        //
        //    For dynamic bodies this is the script-driven control
        //    path: write `Velocity` from a script, the next physics
        //    step integrates from there. The post-step value gets
        //    copied back in `sync_rapier_to_ecs`, so leaving the
        //    component untouched next frame is a no-op (we round-trip
        //    the same value).
        let updates: Vec<(Entity, EcsTransform, RigidBodyKind, Option<Velocity>)> = {
            let mut q = world.query::<(
                Entity,
                &EcsTransform,
                &EcsRigidBody,
                Option<&Velocity>,
            )>();
            q.iter(world)
                .filter(|(_, _, b, _)| b.kind != RigidBodyKind::Static)
                .map(|(e, t, b, v)| (e, *t, b.kind, v.cloned()))
                .collect()
        };
        for (entity, transform, kind, velocity) in updates {
            let Some(handle) = self.body_for_entity.get(&entity) else {
                continue;
            };
            let Some(body) = self.rigid_bodies.get_mut(*handle) else {
                continue;
            };
            if kind == RigidBodyKind::Kinematic {
                body.set_next_kinematic_position(transform_pose(&transform));
            }
            if let Some(v) = velocity {
                body.set_linvel(v.linear, true);
                body.set_angvel(v.angular, true);
            }
        }
    }

    fn create_handle(
        &mut self,
        entity: Entity,
        transform: &EcsTransform,
        body: Option<&EcsRigidBody>,
        collider: &EcsCollider,
        velocity: Option<&Velocity>,
    ) {
        let body = body.cloned().unwrap_or_else(|| EcsRigidBody {
            kind: RigidBodyKind::Static,
            mass: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
        });

        let mut rb_builder = match body.kind {
            RigidBodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            RigidBodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            RigidBodyKind::Static => RigidBodyBuilder::fixed(),
        };
        rb_builder = rb_builder
            .pose(transform_pose(transform))
            .gravity_scale(body.gravity_scale)
            .linear_damping(body.linear_damping)
            .angular_damping(body.angular_damping);
        if body.lock_rotation_x || body.lock_rotation_y || body.lock_rotation_z {
            rb_builder = rb_builder.enabled_rotations(
                !body.lock_rotation_x,
                !body.lock_rotation_y,
                !body.lock_rotation_z,
            );
        }
        if let Some(v) = velocity {
            rb_builder = rb_builder.linvel(v.linear).angvel(v.angular);
        }
        if body.mass > 0.0 {
            rb_builder = rb_builder.additional_mass(body.mass);
        }
        let rb_handle = self.rigid_bodies.insert(rb_builder.build());

        let mut col_builder = collider_builder(&collider.shape)
            .friction(collider.friction)
            .restitution(collider.restitution)
            .sensor(collider.sensor);
        if collider.offset != Vec3::ZERO {
            col_builder = col_builder.translation(collider.offset);
        }
        let col_handle = self.colliders.insert_with_parent(
            col_builder.build(),
            rb_handle,
            &mut self.rigid_bodies,
        );

        self.body_for_entity.insert(entity, rb_handle);
        self.collider_for_entity.insert(entity, col_handle);
    }

    fn sync_rapier_to_ecs(&self, world: &mut World) {
        let mut q = world.query::<(
            Entity,
            &mut EcsTransform,
            Option<&mut Velocity>,
            Option<&EcsRigidBody>,
        )>();
        for (entity, mut transform, velocity, body_kind) in q.iter_mut(world) {
            let Some(handle) = self.body_for_entity.get(&entity) else {
                continue;
            };
            let Some(body) = self.rigid_bodies.get(*handle) else {
                continue;
            };
            // Static bodies don't move; skip the readback to avoid
            // perturbing the authored transform.
            if matches!(body_kind.map(|b| b.kind), Some(RigidBodyKind::Static)) {
                continue;
            }

            let pose = body.position();
            transform.position = pose.translation;
            // YXZ matches Transform::matrix(): yaw (Y) is extracted
            // first so it stays full-range past 90 degrees. XYZ
            // would asin-clamp Y and "flip" the decomposition,
            // which broke controller scripts reading rotation.y.
            let q = pose.rotation;
            let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
            transform.rotation = Vec3::new(
                pitch.to_degrees(),
                yaw.to_degrees(),
                roll.to_degrees(),
            );

            if let Some(mut v) = velocity {
                v.linear = body.linvel();
                v.angular = body.angvel();
            }
        }
    }
}

// ─────────────────── Helpers ───────────────────

fn transform_pose(t: &EcsTransform) -> Pose {
    let rotation = Quat::from_euler(
        glam::EulerRot::YXZ,
        t.rotation.y.to_radians(),
        t.rotation.x.to_radians(),
        t.rotation.z.to_radians(),
    );
    Pose {
        translation: t.position,
        rotation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;
    use ocarina_core::components::Transform as EcsTransform;

    #[test]
    fn dynamic_ball_falls_under_gravity() {
        let mut world = World::new();
        let ball = world
            .spawn((
                EcsTransform {
                    position: Vec3::new(0.0, 5.0, 0.0),
                    rotation: Vec3::ZERO,
                    scale: Vec3::ONE,
                },
                EcsCollider {
                    shape: ColliderShape::Sphere { radius: 0.5 },
                    ..Default::default()
                },
                EcsRigidBody {
                    kind: RigidBodyKind::Dynamic,
                    mass: 1.0,
                    ..Default::default()
                },
            ))
            .id();

        let mut physics = Physics::new();
        for _ in 0..30 {
            physics.step(&mut world, 1.0 / 60.0);
        }
        let final_y = world.get::<EcsTransform>(ball).unwrap().position.y;
        // 0.5 second of -9.81 m/s² gravity gives ~1.23 m of drop, so
        // the ball should be well below its 5 m start. We don't
        // pin a precise number -- solver dynamics drift across rapier
        // versions -- but it must have fallen meaningfully.
        assert!(
            final_y < 4.0,
            "expected ball to fall meaningfully; final_y = {final_y}"
        );
    }

    #[test]
    fn static_floor_stays_put() {
        let mut world = World::new();
        let floor = world
            .spawn((
                EcsTransform {
                    position: Vec3::new(0.0, -0.25, 0.0),
                    rotation: Vec3::ZERO,
                    scale: Vec3::ONE,
                },
                EcsCollider {
                    shape: ColliderShape::Box {
                        size: Vec3::new(10.0, 0.5, 10.0),
                    },
                    ..Default::default()
                },
                EcsRigidBody {
                    kind: RigidBodyKind::Static,
                    ..Default::default()
                },
            ))
            .id();

        let mut physics = Physics::new();
        for _ in 0..60 {
            physics.step(&mut world, 1.0 / 60.0);
        }
        let pos = world.get::<EcsTransform>(floor).unwrap().position;
        assert!(
            (pos.y - (-0.25)).abs() < 1e-4,
            "static floor moved: y = {}",
            pos.y
        );
    }
}

fn collider_builder(shape: &ColliderShape) -> ColliderBuilder {
    match shape {
        ColliderShape::Box { size } => {
            ColliderBuilder::cuboid(size.x * 0.5, size.y * 0.5, size.z * 0.5)
        }
        ColliderShape::Sphere { radius } => ColliderBuilder::ball(*radius),
        ColliderShape::Capsule {
            half_height,
            radius,
        } => ColliderBuilder::capsule_y(*half_height, *radius),
        ColliderShape::HalfSpace => {
            ColliderBuilder::halfspace(rapier3d::na::Unit::new_unchecked(Vec3::Y))
        }
    }
}
