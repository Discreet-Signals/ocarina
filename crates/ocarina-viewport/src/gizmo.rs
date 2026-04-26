//! Translate + rotate gizmo math.
//!
//! Geometry and shading live in the renderer; this module owns the
//! axis enum and the closed-form drag math: ray-axis closest-point for
//! translate, ray-plane intersection + atan2 for rotate. Hit testing
//! is also here so the panel handler stays focused on event routing.

use glam::{Quat, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    #[must_use]
    pub fn dir(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }

    /// Tint used to shade the axis bar / ring. RGB roughly matches
    /// the convention every editor uses (Maya, Blender, Unity, …).
    #[must_use]
    pub fn tint(self) -> [f32; 4] {
        match self {
            Axis::X => [1.00, 0.30, 0.25, 1.0],
            Axis::Y => [0.30, 0.95, 0.35, 1.0],
            Axis::Z => [0.30, 0.55, 1.00, 1.0],
        }
    }

    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
}

/// Active drag state. Captured on the PointerDown that hits a gizmo
/// handle; the move/up handlers consume it to drive the entity's
/// transform and write back on release.
///
/// The full Transform snapshot (position + rotation + scale at drag
/// start) is captured so we can restore non-affected fields cleanly
/// when writing back; only the field for the active mode actually
/// changes during the drag.
#[derive(Debug, Clone)]
#[allow(dead_code)] // start_rotation/start_scale are kept for future parent-rotation handling
pub enum GizmoDrag {
    Translate {
        axis: Axis,
        /// Entity local position at drag-start.
        start_position: Vec3,
        start_rotation: Vec3,
        start_scale: Vec3,
        /// Axis-line parameter where the cursor ray was closest to
        /// the axis at PointerDown — zero point for subsequent moves.
        start_axis_param: f32,
    },
    Rotate {
        axis: Axis,
        start_position: Vec3,
        /// Entity rotation (as a quat) when drag began.
        start_quat: Quat,
        start_scale: Vec3,
        /// Angle in the rotation plane where the click landed —
        /// reference for delta angle.
        start_angle: f32,
    },
}

impl GizmoDrag {
    pub fn axis(&self) -> Axis {
        match self {
            GizmoDrag::Translate { axis, .. } | GizmoDrag::Rotate { axis, .. } => *axis,
        }
    }
}

/// Bounds of the translate axis as a world-space AABB. Used for ray
/// hit-testing — generous padding around the visible 2D line so the
/// user can click roughly in its vicinity.
#[must_use]
pub fn translate_bar_aabb(origin: Vec3, axis: Axis, scale_world: f32) -> (Vec3, Vec3) {
    let along = scale_world * 1.5;
    let thick = scale_world * 0.04;
    // Padding makes the bar easier to click — the visual stays the
    // same, but the AABB used for hit-testing is generous.
    let pad = scale_world * 0.06;
    let half_thick = thick * 0.5 + pad;
    match axis {
        Axis::X => (
            origin + Vec3::new(0.0, -half_thick, -half_thick),
            origin + Vec3::new(along + pad, half_thick, half_thick),
        ),
        Axis::Y => (
            origin + Vec3::new(-half_thick, 0.0, -half_thick),
            origin + Vec3::new(half_thick, along + pad, half_thick),
        ),
        Axis::Z => (
            origin + Vec3::new(-half_thick, -half_thick, 0.0),
            origin + Vec3::new(half_thick, half_thick, along + pad),
        ),
    }
}

/// Hit-test a ray against the three translate bars. Returns the
/// closest-hit axis if any.
#[must_use]
pub fn pick_translate_axis(
    origin: Vec3,
    ray_origin: Vec3,
    ray_dir: Vec3,
    scale_world: f32,
) -> Option<Axis> {
    let mut best: Option<(f32, Axis)> = None;
    for axis in Axis::ALL {
        let (min, max) = translate_bar_aabb(origin, axis, scale_world);
        if let Some(t) = ray_aabb(ray_origin, ray_dir, min, max) {
            if t < 0.0 {
                continue;
            }
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, axis));
            }
        }
    }
    best.map(|(_, axis)| axis)
}

/// Project the cursor ray onto an axis line through `origin` and
/// return the line parameter. `None` if ray and axis are parallel.
#[must_use]
pub fn closest_point_on_axis(
    origin: Vec3,
    axis_dir: Vec3,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Option<f32> {
    let a = axis_dir.dot(axis_dir);
    let b = axis_dir.dot(ray_dir);
    let e = ray_dir.dot(ray_dir);
    let d = a * e - b * b;
    if d.abs() < 1e-6 {
        return None;
    }
    let r = origin - ray_origin;
    let c = axis_dir.dot(r);
    let f = ray_dir.dot(r);
    Some((b * f - c * e) / d)
}

/// Hit-test a ray against the three rotate rings. Returns the axis of
/// the ring whose disk the ray crosses closest to its rim — within a
/// tolerance band so glancing clicks just outside still register.
#[must_use]
pub fn pick_rotate_axis(
    origin: Vec3,
    ray_origin: Vec3,
    ray_dir: Vec3,
    scale_world: f32,
) -> Option<Axis> {
    let radius = scale_world * 1.0;
    let band = scale_world * 0.12;
    let mut best: Option<(f32, Axis)> = None;
    for axis in Axis::ALL {
        let n = axis.dir();
        // Ray-plane: t = ((origin - ray_origin) · n) / (ray_dir · n)
        let denom = ray_dir.dot(n);
        if denom.abs() < 1e-6 {
            continue;
        }
        let t = (origin - ray_origin).dot(n) / denom;
        if t < 0.0 {
            continue;
        }
        let hit = ray_origin + ray_dir * t;
        let dist = (hit - origin).length();
        if (dist - radius).abs() <= band {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, axis));
            }
        }
    }
    best.map(|(_, axis)| axis)
}

/// Angle (in radians) of the cursor's intersection with the rotation
/// plane, measured around `axis`. Used both for the initial reference
/// angle on PointerDown and for incremental deltas during drag. Returns
/// `None` if the ray is parallel to the plane.
#[must_use]
pub fn rotate_drag_angle(
    origin: Vec3,
    axis: Axis,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Option<f32> {
    let n = axis.dir();
    let denom = ray_dir.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray_origin).dot(n) / denom;
    let hit = ray_origin + ray_dir * t;
    let local = hit - origin;
    // Two perpendicular reference axes that span the rotation plane.
    // X-rotate plane is YZ → use (Y, Z); Y-rotate plane is XZ → use
    // (Z, X) so a counter-clockwise drag (when looking down +Y) reads
    // as a positive angle; Z-rotate plane is XY → use (X, Y).
    let (u, v) = match axis {
        Axis::X => (Vec3::Y, Vec3::Z),
        Axis::Y => (Vec3::Z, Vec3::X),
        Axis::Z => (Vec3::X, Vec3::Y),
    };
    Some(local.dot(v).atan2(local.dot(u)))
}

/// Slab-method ray vs axis-aligned box. Returns the entry distance
/// `t` (in units of `ray_dir`) if the ray hits, otherwise `None`.
fn ray_aabb(ray_origin: Vec3, ray_dir: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv = Vec3::new(
        if ray_dir.x.abs() > 1e-8 { 1.0 / ray_dir.x } else { f32::INFINITY },
        if ray_dir.y.abs() > 1e-8 { 1.0 / ray_dir.y } else { f32::INFINITY },
        if ray_dir.z.abs() > 1e-8 { 1.0 / ray_dir.z } else { f32::INFINITY },
    );
    let t1 = (min - ray_origin) * inv;
    let t2 = (max - ray_origin) * inv;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let near = tmin.x.max(tmin.y).max(tmin.z);
    let far = tmax.x.min(tmax.y).min(tmax.z);
    if far < near.max(0.0) {
        None
    } else {
        Some(near.max(0.0))
    }
}
