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

// ────────────────────────── 2D gizmo geometry ──────────────────────────
//
// Gizmo vertices carry an *anchor* world position, a *neighbor* point
// along the same line, and a signed pixel-offset. The gizmo shader
// expands each pair (anchor, neighbor) into a screen-space-thick line
// at draw time; arrow tips collapse their offset to 0 so the shape
// pinches to a point. This produces constant-thickness lines on
// screen no matter the camera angle or distance.

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub neighbor: [f32; 3],
    /// `x` = signed pixel offset perpendicular to the line. `y` reserved
    /// for future use (e.g. dashed-line phase).
    pub extras: [f32; 2],
}

impl GizmoVertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

pub struct GizmoMesh {
    pub vertices: Vec<GizmoVertex>,
    pub indices: Vec<u32>,
}

/// Pixel thickness for the translate axis line.
const LINE_THICKNESS_PX: f32 = 16.0;
/// Pixel half-width of the arrow base (so total width is `2 * this`).
const ARROW_HALF_PX: f32 = 24.0;

/// Build the translate-axis handle: a thin line + arrowhead pointing
/// along `+X` in local space. The renderer rotates this base mesh
/// into the Y / Z directions via instance model matrices, so we only
/// author one orientation here.
///
/// Geometry layout (local frame):
/// - Line shaft: from `(0, 0, 0)` to `(line_end, 0, 0)`, expanded
///   perpendicular to ±LINE_THICKNESS_PX/2 pixels in screen space.
/// - Arrow base: square at `(line_end, 0, 0)`, 2 verts at ±ARROW_HALF
/// - Arrow tip:  point at `(arrow_tip, 0, 0)`, collapses to a single
///   pixel (offset = 0).
///
/// `length` is the world-space length of the axis from origin to
/// arrow tip. Caller scales it by camera distance to keep gizmos
/// roughly screen-constant.
pub fn build_translate_axis(length: f32) -> GizmoMesh {
    let line_end = length * 0.85;
    let arrow_tip = length;

    // Anchor positions in local space. The "neighbor" of a line vert
    // is the OTHER endpoint of the line (so the shader can compute
    // the line's direction in screen space). For arrow base verts,
    // neighbor is the tip — so the arrow widens *perpendicular* to
    // the axis direction, same as the line.
    let p0 = [0.0_f32, 0.0, 0.0];
    let p1 = [line_end, 0.0, 0.0];
    let p_tip = [arrow_tip, 0.0, 0.0];

    let line_half = LINE_THICKNESS_PX * 0.5;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(7);
    // Line quad: 4 verts (start ±, end ±). `extras.y` carries the
    // local half-thickness — constant across the quad and so
    // constant after interpolation; the FS uses it to feather alpha
    // at `|signed_offset| ≈ extras.y` for AA.
    vertices.push(GizmoVertex { position: p0, neighbor: p1, extras: [-line_half, line_half] });
    vertices.push(GizmoVertex { position: p0, neighbor: p1, extras: [ line_half, line_half] });
    vertices.push(GizmoVertex { position: p1, neighbor: p0, extras: [ line_half, line_half] });
    vertices.push(GizmoVertex { position: p1, neighbor: p0, extras: [-line_half, line_half] });
    // Arrow triangle: 3 verts (base −, base +, tip). `extras.y`
    // tapers from `ARROW_HALF_PX` at the base to 0 at the tip — a
    // fragment near the slanted edge has `|signed_offset| ≈ max_half`
    // so the FS feather kicks in along that edge as well.
    vertices.push(GizmoVertex { position: p1, neighbor: p_tip, extras: [-ARROW_HALF_PX, ARROW_HALF_PX] });
    vertices.push(GizmoVertex { position: p1, neighbor: p_tip, extras: [ ARROW_HALF_PX, ARROW_HALF_PX] });
    vertices.push(GizmoVertex { position: p_tip, neighbor: p1,  extras: [ 0.0,           0.0] });

    // Indices: line is two triangles (CCW from the +Y side); arrow is
    // one triangle. The expansion shader is symmetric in side, so
    // wind order doesn't affect what gets drawn — but cull_mode is
    // None on the gizmo pipeline anyway.
    let indices: Vec<u32> = vec![
        0, 1, 2,  0, 2, 3,
        4, 5, 6,
    ];

    GizmoMesh { vertices, indices }
}

/// Pixel thickness for the rotate ring's stroke.
const RING_THICKNESS_PX: f32 = 12.0;

/// Build the rotate-axis handle: a thin circular line in the XY plane
/// centered on origin with radius `radius`. The renderer rotates this
/// base mesh into the YZ / XZ plane for the X / Y axis rings via the
/// instance model matrix (Z axis ring uses identity).
///
/// 64 segments — fine enough that the polygonal step doesn't read as
/// faceted at editor zoom levels.
pub fn build_rotate_ring(radius: f32) -> GizmoMesh {
    const SEGMENTS: usize = 64;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(SEGMENTS * 2);
    let mut indices = Vec::<u32>::with_capacity(SEGMENTS * 6);

    let ring_half = RING_THICKNESS_PX * 0.5;

    // Two verts per ring point (inner / outer), each carrying the
    // *local tangent* direction at that point as `neighbor`. This is
    // the load-bearing fix for the ring sawtooth artifact:
    //
    // The shader computes the perpendicular as `rot90(screen-projected
    // tangent)`. If two adjacent segments at the same ring point use
    // tangents pointing in OPPOSITE directions (one says "back to the
    // previous point", the other "forward to the next"), the rot90
    // perps come out opposite-signed too, so the same world point
    // gets extruded to two different screen positions across the
    // junction → sawteeth.
    //
    // Storing one canonical tangent per ring point (the local-frame
    // tangent at angle θ, which is `(-sin θ, cos θ)`) makes the
    // junction perpendiculars match. We emit `SEGMENTS + 1` ring
    // points so the last shares position with the first cleanly.
    for i in 0..=SEGMENTS {
        let theta = (i % SEGMENTS) as f32 / (SEGMENTS as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        let p = [c * radius, s * radius, 0.0];
        // Local tangent: perpendicular to the radial in the ring's
        // plane. Same direction at this vertex regardless of which
        // segment is consuming it.
        let t = [-s, c, 0.0];
        let nbr = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
        vertices.push(GizmoVertex {
            position: p,
            neighbor: nbr,
            extras: [-ring_half, ring_half],
        });
        vertices.push(GizmoVertex {
            position: p,
            neighbor: nbr,
            extras: [ring_half, ring_half],
        });
    }
    for i in 0..SEGMENTS {
        let inner_a = (i * 2) as u32;
        let outer_a = (i * 2 + 1) as u32;
        let inner_b = ((i + 1) * 2) as u32;
        let outer_b = ((i + 1) * 2 + 1) as u32;
        // Two triangles per segment quad.
        indices.extend_from_slice(&[inner_a, outer_a, inner_b, outer_a, outer_b, inner_b]);
    }

    GizmoMesh { vertices, indices }
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
