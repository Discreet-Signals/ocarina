//! CPU-side mesh data and procedural geometry.
//!
//! Phase 4 renders every entity with `Transform` as a unit cube; the
//! Triforce builder remains as a kept-for-reference procedural mesh and
//! is kept opt-in (it isn't called by the live editor). The asset
//! pipeline that replaces these placeholder shapes lands in Phase 6.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// Interleaved vertex: position / normal / uv.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

impl Mesh {
    #[allow(dead_code)] // used by callers that focus the camera on a specific mesh
    pub fn center(&self) -> Vec3 {
        (self.aabb_min + self.aabb_max) * 0.5
    }

    #[allow(dead_code)]
    pub fn radius(&self) -> f32 {
        ((self.aabb_max - self.aabb_min) * 0.5).length()
    }
}

// ────────────────────────── Triforce ──────────────────────────

/// Build the Triforce.
///
/// Three equilateral triangles tile a larger equilateral triangle, with a
/// downward-pointing equilateral gap in the middle. Each piece is a
/// triangular prism (flat triangle extruded a little on Z) so the shape
/// reads as real geometry when the camera orbits.
///
/// The whole mesh is centered at the origin.
#[allow(dead_code)] // kept for reference; the live editor renders unit cubes
pub fn build_triforce() -> Mesh {
    let sqrt3 = 3f32.sqrt();

    // Outer triangle spans (-1..1) horizontally and (0..sqrt3) vertically,
    // then we shift so the centroid is at the origin.
    let cy = sqrt3 / 3.0; // centroid Y of the outer equilateral triangle.
    let a = Vec2::new(-1.0, 0.0 - cy);
    let b = Vec2::new(1.0, 0.0 - cy);
    let c = Vec2::new(0.0, sqrt3 - cy);
    let m_ab = Vec2::new(0.0, 0.0 - cy);
    let m_bc = Vec2::new(0.5, sqrt3 * 0.5 - cy);
    let m_ca = Vec2::new(-0.5, sqrt3 * 0.5 - cy);

    // Each piece is one equilateral triangle pointing up; CCW when viewed
    // from +Z (so the front cap faces the camera by default).
    let pieces: [[Vec2; 3]; 3] = [
        [a, m_ab, m_ca],   // bottom-left
        [m_ab, b, m_bc],   // bottom-right
        [m_ca, m_bc, c],   // top
    ];

    let thickness = 0.18;
    let half_t = thickness * 0.5;

    let mut vertices: Vec<Vertex> = Vec::with_capacity(3 * (2 * 3 + 3 * 4));
    let mut indices: Vec<u32> = Vec::with_capacity(3 * (2 * 3 + 3 * 6));

    for piece in &pieces {
        // Lift piece into 3D, one copy for each cap.
        let f = |i: usize, z: f32| Vec3::new(piece[i].x, piece[i].y, z);
        let f0 = f(0, half_t);
        let f1 = f(1, half_t);
        let f2 = f(2, half_t);
        let b0 = f(0, -half_t);
        let b1 = f(1, -half_t);
        let b2 = f(2, -half_t);

        // Front cap — CCW from +Z → normal points +Z.
        push_triangle(&mut vertices, &mut indices, f0, f1, f2);
        // Back cap — reversed winding so CCW from -Z → normal points -Z.
        push_triangle(&mut vertices, &mut indices, b0, b2, b1);

        // Three side quads. For each CCW edge (pi → pj) on the front,
        // the outward-facing quad is (fi, fj, bj, bi).
        push_quad(&mut vertices, &mut indices, f0, f1, b1, b0);
        push_quad(&mut vertices, &mut indices, f1, f2, b2, b1);
        push_quad(&mut vertices, &mut indices, f2, f0, b0, b2);
    }

    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
    for v in &vertices {
        let p = Vec3::from(v.position);
        aabb_min = aabb_min.min(p);
        aabb_max = aabb_max.max(p);
    }

    Mesh {
        vertices,
        indices,
        aabb_min,
        aabb_max,
    }
}

#[allow(dead_code)]
fn push_triangle(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, a: Vec3, b: Vec3, c: Vec3) {
    let normal = (b - a).cross(c - a).normalize_or_zero();
    let base = vertices.len() as u32;
    for v in [a, b, c] {
        vertices.push(Vertex {
            position: [v.x, v.y, v.z],
            normal: [normal.x, normal.y, normal.z],
            uv: [0.0, 0.0],
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

// ────────────────────────── Unit cube ──────────────────────────

/// Build a unit cube centered at the origin (corners at ±0.5). Used as
/// the placeholder mesh for every entity with a Transform until the
/// Phase 6 asset pipeline lands. Flat-shaded — six normals, six quads.
pub fn build_unit_cube() -> Mesh {
    let h = 0.5;
    let mut vertices: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    // (corner_offsets, axis-aligned outward normal). Corners listed CCW
    // from outside so push_quad's first-triangle normal points outward.
    let faces: [([Vec3; 4], Vec3); 6] = [
        ( // +X
            [
                Vec3::new( h, -h, -h),
                Vec3::new( h,  h, -h),
                Vec3::new( h,  h,  h),
                Vec3::new( h, -h,  h),
            ],
            Vec3::X,
        ),
        ( // -X
            [
                Vec3::new(-h, -h,  h),
                Vec3::new(-h,  h,  h),
                Vec3::new(-h,  h, -h),
                Vec3::new(-h, -h, -h),
            ],
            -Vec3::X,
        ),
        ( // +Y
            [
                Vec3::new(-h,  h,  h),
                Vec3::new( h,  h,  h),
                Vec3::new( h,  h, -h),
                Vec3::new(-h,  h, -h),
            ],
            Vec3::Y,
        ),
        ( // -Y
            [
                Vec3::new(-h, -h, -h),
                Vec3::new( h, -h, -h),
                Vec3::new( h, -h,  h),
                Vec3::new(-h, -h,  h),
            ],
            -Vec3::Y,
        ),
        ( // +Z
            [
                Vec3::new(-h, -h,  h),
                Vec3::new( h, -h,  h),
                Vec3::new( h,  h,  h),
                Vec3::new(-h,  h,  h),
            ],
            Vec3::Z,
        ),
        ( // -Z
            [
                Vec3::new( h, -h, -h),
                Vec3::new(-h, -h, -h),
                Vec3::new(-h,  h, -h),
                Vec3::new( h,  h, -h),
            ],
            -Vec3::Z,
        ),
    ];

    for (corners, normal) in &faces {
        let base = vertices.len() as u32;
        for v in corners {
            vertices.push(Vertex {
                position: [v.x, v.y, v.z],
                normal: [normal.x, normal.y, normal.z],
                uv: [0.0, 0.0],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    Mesh {
        vertices,
        indices,
        aabb_min: Vec3::splat(-h),
        aabb_max: Vec3::splat(h),
    }
}

#[allow(dead_code)]
fn push_quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
    // Wound a→b→c→d CCW from outside. Normal from the first triangle.
    let normal = (b - a).cross(c - a).normalize_or_zero();
    let base = vertices.len() as u32;
    for v in [a, b, c, d] {
        vertices.push(Vertex {
            position: [v.x, v.y, v.z],
            normal: [normal.x, normal.y, normal.z],
            uv: [0.0, 0.0],
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
