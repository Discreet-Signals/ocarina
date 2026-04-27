//! Gizmo geometry. Lit-pass meshes (cube, sphere, user OBJs) are
//! owned by `ocarina-render`'s `MeshRegistry`; this file only holds
//! the editor's gizmo handles, which use a different vertex format
//! (`GizmoVertex`) and a screen-space-expanded line shader.

use bytemuck::{Pod, Zeroable};

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
    /// `x` = signed pixel offset perpendicular to the line. `y`
    /// reserved for future use (e.g. dashed-line phase).
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
pub fn build_translate_axis(length: f32) -> GizmoMesh {
    let line_end = length * 0.85;
    let arrow_tip = length;

    let p0 = [0.0_f32, 0.0, 0.0];
    let p1 = [line_end, 0.0, 0.0];
    let p_tip = [arrow_tip, 0.0, 0.0];

    let line_half = LINE_THICKNESS_PX * 0.5;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(7);
    // Line quad: 4 verts (start ±, end ±). `extras.y` carries the
    // local half-thickness; the FS uses it to feather alpha for AA.
    vertices.push(GizmoVertex { position: p0, neighbor: p1, extras: [-line_half, line_half] });
    vertices.push(GizmoVertex { position: p0, neighbor: p1, extras: [ line_half, line_half] });
    vertices.push(GizmoVertex { position: p1, neighbor: p0, extras: [ line_half, line_half] });
    vertices.push(GizmoVertex { position: p1, neighbor: p0, extras: [-line_half, line_half] });
    // Arrow triangle: 3 verts (base −, base +, tip). `extras.y`
    // tapers from `ARROW_HALF_PX` at the base to 0 at the tip so the
    // FS feather kicks in along the slanted edge as well.
    vertices.push(GizmoVertex { position: p1, neighbor: p_tip, extras: [-ARROW_HALF_PX, ARROW_HALF_PX] });
    vertices.push(GizmoVertex { position: p1, neighbor: p_tip, extras: [ ARROW_HALF_PX, ARROW_HALF_PX] });
    vertices.push(GizmoVertex { position: p_tip, neighbor: p1,  extras: [ 0.0,           0.0] });

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
pub fn build_rotate_ring(radius: f32) -> GizmoMesh {
    const SEGMENTS: usize = 64;
    let mut vertices = Vec::<GizmoVertex>::with_capacity(SEGMENTS * 2);
    let mut indices = Vec::<u32>::with_capacity(SEGMENTS * 6);

    let ring_half = RING_THICKNESS_PX * 0.5;

    // Two verts per ring point (inner / outer), each carrying the
    // *local tangent* direction at that point as `neighbor`. The
    // shader computes the perpendicular as `rot90(screen-projected
    // tangent)`. Storing one canonical tangent per ring point (the
    // local-frame tangent at angle θ, `(-sin θ, cos θ)`) makes the
    // junction perpendiculars match across adjacent segments and
    // avoids sawtooth artifacts at vertex shared between two
    // segments.
    for i in 0..=SEGMENTS {
        let theta = (i % SEGMENTS) as f32 / (SEGMENTS as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        let p = [c * radius, s * radius, 0.0];
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
        indices.extend_from_slice(&[inner_a, outer_a, inner_b, outer_a, outer_b, inner_b]);
    }

    GizmoMesh { vertices, indices }
}
