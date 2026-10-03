//! Character controller.
//!
//! - WASD moves the host entity in its **facing** XZ frame: W
//!   pushes along the cube's forward direction (its local -Z),
//!   D pushes right (local +X), and so on.
//! - Mouse X delta drives the Y angular velocity. Moving the mouse
//!   right rotates the cube clockwise (viewed from above), so the
//!   camera child rides along and the world appears to swing past.
//! - Space jumps when the body is roughly stationary vertically.
//!
//! Movement is *velocity*, not impulse -- keys held = constant speed,
//! keys released = instant stop. Rotation is angular velocity scaled
//! by mouse delta and divided by `dt`, so per-pixel rotation stays
//! frame-rate independent.

use ocarina_gameplay_api::{Context, Entity, Key, OcarinaScript, Script, Transform, Vec3};
use ocarina_physics::Velocity;

#[derive(OcarinaScript, Default)]
#[ocarina_script(path = "scripts/controller.rs")]
pub struct Controller;

impl Script for Controller {
    fn tick(&mut self, ctx: &mut Context<'_>, entity: Entity) {
        const MOVE_SPEED: f32 = 4.0;
        const JUMP_SPEED: f32 = 6.0;
        const GROUNDED_VEL_EPS: f32 = 0.5;
        // radians of yaw per pixel of mouse delta. Higher = twitchier.
        const MOUSE_SENSITIVITY: f32 = 0.005;

        let input = &*ctx.input;
        let dt = ctx.time.dt().max(1.0e-3);

        // Read current facing from Transform.rotation.y (degrees → rad).
        let yaw_rad = ctx
            .get::<Transform>(entity)
            .map(|t| t.rotation.y.to_radians())
            .unwrap_or(0.0);
        // Forward = entity's local -Z rotated by yaw around Y.
        // Right   = entity's local +X rotated by yaw around Y.
        let (sin, cos) = yaw_rad.sin_cos();
        let forward = Vec3::new(-sin, 0.0, -cos);
        let right = Vec3::new(cos, 0.0, -sin);

        let mut wish = Vec3::ZERO;
        if input.key_held(Key::W) {
            wish += forward;
        }
        if input.key_held(Key::S) {
            wish -= forward;
        }
        if input.key_held(Key::D) {
            wish += right;
        }
        if input.key_held(Key::A) {
            wish -= right;
        }
        // Normalise so diagonals don't move √2 faster.
        let mag = wish.length();
        if mag > 1.0 {
            wish /= mag;
        }

        let want_jump = input.key_just_pressed(Key::Space);

        // Yaw rate from mouse-X. Negative so dragging right turns the
        // cube clockwise (and the world appears to swing left), which
        // is the FPS-game convention.
        let mouse_dx = input.mouse_delta().x;
        let yaw_rate = -mouse_dx * MOUSE_SENSITIVITY / dt;

        // Read existing Y velocity so jump doesn't kill gravity.
        let current_vy = ctx
            .get::<Velocity>(entity)
            .map(|v| v.linear.y)
            .unwrap_or(0.0);
        let new_vy = if want_jump && current_vy.abs() < GROUNDED_VEL_EPS {
            JUMP_SPEED
        } else {
            current_vy
        };

        if let Some(mut v) = ctx.get_mut::<Velocity>(entity) {
            v.linear.x = wish.x * MOVE_SPEED;
            v.linear.z = wish.z * MOVE_SPEED;
            v.linear.y = new_vy;
            v.angular.y = yaw_rate;
            // Leave x/z angular alone -- they're locked anyway, but
            // explicit is friendly to future-us.
            v.angular.x = 0.0;
            v.angular.z = 0.0;
        }
    }
}
