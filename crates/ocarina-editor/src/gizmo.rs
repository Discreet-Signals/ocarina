//! Translate gizmo math.
//!
//! The gizmo itself is rendered as three tinted unit cubes scaled into
//! thin axis-aligned bars at the selected entity's position. Hit testing
//! and drag math live here so the viewport's render loop stays focused
//! on GPU recording.
//!
//! ## Drag math
//!
//! Given an axis (world direction) and the cursor's world ray, find the
//! point on the axis closest to the ray. The classic two-ray closest
//! point: solve for `s` in `axis_origin + s·axis_dir` such that the
//! distance to the cursor ray is minimal. We use that point's projection
//! along the axis as the drag amount.

use glam::Vec3;

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

    #[must_use]
    pub fn tint(self) -> [f32; 4] {
        match self {
            Axis::X => [1.0, 0.30, 0.25, 1.0],
            Axis::Y => [0.30, 0.95, 0.35, 1.0],
            Axis::Z => [0.30, 0.55, 1.00, 1.0],
        }
    }
}

/// Find the parameter `s` along the axis line `origin + s·dir` such that
/// the corresponding point is closest to the ray `ray_origin + t·ray_dir`.
/// Returns `None` if the lines are essentially parallel.
#[must_use]
pub fn closest_point_on_axis(
    axis_origin: Vec3,
    axis_dir: Vec3,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Option<f32> {
    // Standard closest-points-between-two-lines, returning only the
    // axis-line parameter. See Realtime Rendering §22.4.
    let a = axis_dir.dot(axis_dir);
    let b = axis_dir.dot(ray_dir);
    let e = ray_dir.dot(ray_dir);
    let d = a * e - b * b;
    if d.abs() < 1e-6 {
        return None;
    }
    let r = axis_origin - ray_origin;
    let c = axis_dir.dot(r);
    let f = ray_dir.dot(r);
    Some((b * f - c * e) / d)
}

/// Slab/AABB ray test. Returns the entry distance `t` if the ray hits
/// the box bounded by `min..max` at a non-negative `t`.
#[must_use]
pub fn ray_aabb(
    ray_origin: Vec3,
    ray_dir: Vec3,
    min: Vec3,
    max: Vec3,
) -> Option<f32> {
    let inv = Vec3::new(
        if ray_dir.x.abs() > 1e-8 { 1.0 / ray_dir.x } else { f32::INFINITY },
        if ray_dir.y.abs() > 1e-8 { 1.0 / ray_dir.y } else { f32::INFINITY },
        if ray_dir.z.abs() > 1e-8 { 1.0 / ray_dir.z } else { f32::INFINITY },
    );
    let t1 = (min - ray_origin) * inv;
    let t2 = (max - ray_origin) * inv;
    let tmin_v = t1.min(t2);
    let tmax_v = t1.max(t2);
    let tmin = tmin_v.x.max(tmin_v.y).max(tmin_v.z);
    let tmax = tmax_v.x.min(tmax_v.y).min(tmax_v.z);
    if tmax >= tmin.max(0.0) {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_hit() {
        // Ray from (0,0,5) toward -Z hits unit cube centered at origin.
        let t = ray_aabb(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::splat(-0.5),
            Vec3::splat(0.5),
        );
        assert!(t.is_some());
        let t = t.unwrap();
        assert!((t - 4.5).abs() < 1e-3, "t = {t}");
    }

    #[test]
    fn aabb_miss() {
        // Ray pointing away from the cube — no hit.
        let t = ray_aabb(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::splat(-0.5),
            Vec3::splat(0.5),
        );
        assert!(t.is_none());
    }

    #[test]
    fn axis_closest_point_perpendicular() {
        // Axis along X at origin; ray going through (3, 1, 0) along Y.
        // Closest point on X to that ray is x = 3.
        let s = closest_point_on_axis(
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::Y,
        )
        .unwrap();
        assert!((s - 3.0).abs() < 1e-3, "s = {s}");
    }
}
