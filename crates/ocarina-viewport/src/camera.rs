//! Orbit camera for the viewport pane.
//!
//! Yaw/pitch/distance around a target. Perspective projection. View and
//! view-projection matrices exposed in a form ready to upload to a uniform
//! buffer. Input handling (drag to orbit, scroll to zoom) is the caller's
//! responsibility — the camera just consumes deltas.

use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl OrbitCamera {
    #[must_use]
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.7,
            pitch: 0.55,
            fov_y: 55f32.to_radians(),
            near: 0.05,
            far: 100.0,
        }
    }

    #[must_use]
    pub fn position(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let offset = Vec3::new(cp * sy, sp, cp * cy) * self.distance;
        self.target + offset
    }

    #[must_use]
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    #[must_use]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(0.01), self.near, self.far)
    }

    #[must_use]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// Drag delta in pixels → rotation in radians. Multiplier tuned for feel.
    /// Pitch follows the cursor's vertical motion (drag down → look down)
    /// to match the convention the rest of the editor uses.
    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        const PIXELS_PER_RAD: f32 = 200.0;
        self.yaw -= dx_px / PIXELS_PER_RAD;
        self.pitch = (self.pitch + dy_px / PIXELS_PER_RAD).clamp(-1.45, 1.45);
    }

    /// Drag delta in pixels → lateral target translation in world space.
    /// Right is camera-right, up is camera-up. Translation magnitude is
    /// proportional to `distance` so movement feels consistent at any
    /// zoom level (1 px ≈ 0.0025 world units per unit of distance).
    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        let forward = (self.target - self.position()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance * 0.0025;
        self.target += (-right * dx_px + up * dy_px) * scale;
    }

    /// Scroll delta → distance change. Exponential to keep zoom feel even at
    /// any distance scale.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.08)).clamp(0.3, 50.0);
    }

    /// Move the orbit target along the camera's forward / right basis
    /// vectors plus world-up — used by WASD (+Q/E) navigation. All
    /// inputs are in *world units*; the caller scales by speed * dt.
    ///
    /// "Forward" is **horizontal** (the camera's forward projected onto
    /// the XZ plane) so pitching the view doesn't make `W` fly into
    /// the floor. Vertical motion is on world-up, decoupled from
    /// camera pitch — Minecraft / Blender conventions.
    pub fn move_target(&mut self, forward_units: f32, right_units: f32, up_units: f32) {
        let view = (self.target - self.position()).normalize_or_zero();
        // Forward projected onto the horizontal plane so W moves the
        // camera *along the ground*, regardless of pitch.
        let forward_flat = Vec3::new(view.x, 0.0, view.z).normalize_or_zero();
        let right = forward_flat.cross(Vec3::Y).normalize_or_zero();
        self.target += forward_flat * forward_units + right * right_units + Vec3::Y * up_units;
    }

    /// Build a world-space ray (origin + normalized direction) through
    /// the given UV in the viewport (0..1, with v measured from the top).
    /// Used by entity picking and by the gizmo's drag math.
    #[must_use]
    pub fn ray_through_uv(&self, u: f32, v: f32, aspect: f32) -> (Vec3, Vec3) {
        let ndc_x = u * 2.0 - 1.0;
        let ndc_y = 1.0 - v * 2.0;
        let tan_half_fov = (self.fov_y * 0.5).tan();
        let cam_x = ndc_x * tan_half_fov * aspect;
        let cam_y = ndc_y * tan_half_fov;

        let position = self.position();
        let forward = (self.target - position).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        let world_dir = (right * cam_x + up * cam_y + forward).normalize_or_zero();
        (position, world_dir)
    }
}
