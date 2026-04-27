//! Built-in primitive mesh generators. Returned as plain
//! [`crate::mesh::MeshAsset`] data; the registry uploads them to
//! GPU buffers via the RHI just like any user-loaded asset.

use glam::Vec3;

use crate::mesh::{MeshAsset, Vertex};

// ─────────────────────────── Unit cube ───────────────────────────

/// Unit cube centered at the origin (corners at ±0.5). Flat-shaded
/// with one normal per face (24 vertices, 36 indices, 12 triangles).
pub fn build_unit_cube() -> MeshAsset {
    let h = 0.5_f32;
    let mut vertices: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    // Each face: 4 CCW corners (viewed from outside) + outward normal.
    let faces: [([Vec3; 4], Vec3); 6] = [
        ( // +X
            [
                Vec3::new(h, -h, -h), Vec3::new(h, h, -h),
                Vec3::new(h, h, h), Vec3::new(h, -h, h),
            ],
            Vec3::X,
        ),
        ( // -X
            [
                Vec3::new(-h, -h, h), Vec3::new(-h, h, h),
                Vec3::new(-h, h, -h), Vec3::new(-h, -h, -h),
            ],
            -Vec3::X,
        ),
        ( // +Y
            [
                Vec3::new(-h, h, h), Vec3::new(h, h, h),
                Vec3::new(h, h, -h), Vec3::new(-h, h, -h),
            ],
            Vec3::Y,
        ),
        ( // -Y
            [
                Vec3::new(-h, -h, -h), Vec3::new(h, -h, -h),
                Vec3::new(h, -h, h), Vec3::new(-h, -h, h),
            ],
            -Vec3::Y,
        ),
        ( // +Z
            [
                Vec3::new(-h, -h, h), Vec3::new(h, -h, h),
                Vec3::new(h, h, h), Vec3::new(-h, h, h),
            ],
            Vec3::Z,
        ),
        ( // -Z
            [
                Vec3::new(h, -h, -h), Vec3::new(-h, -h, -h),
                Vec3::new(-h, h, -h), Vec3::new(h, h, -h),
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

    MeshAsset {
        vertices,
        indices,
        aabb_min: Vec3::splat(-h),
        aabb_max: Vec3::splat(h),
    }
}

// ─────────────────────────── Unit sphere ───────────────────────────

/// Unit-diameter sphere (radius 0.5, fits the same ±0.5 cube extent).
/// Built by subdividing an icosahedron `subdivisions` times. Smooth-
/// shaded: vertex normal == normalised position.
///
/// `subdivisions = 2` gives 320 triangles, plenty for gameplay.
pub fn build_unit_sphere(subdivisions: u32) -> MeshAsset {
    let t = (1.0_f32 + 5.0_f32.sqrt()) * 0.5;

    let mut verts: Vec<Vec3> = vec![
        Vec3::new(-1.0,  t,  0.0),
        Vec3::new( 1.0,  t,  0.0),
        Vec3::new(-1.0, -t,  0.0),
        Vec3::new( 1.0, -t,  0.0),
        Vec3::new( 0.0, -1.0,  t),
        Vec3::new( 0.0,  1.0,  t),
        Vec3::new( 0.0, -1.0, -t),
        Vec3::new( 0.0,  1.0, -t),
        Vec3::new( t,  0.0, -1.0),
        Vec3::new( t,  0.0,  1.0),
        Vec3::new(-t,  0.0, -1.0),
        Vec3::new(-t,  0.0,  1.0),
    ];

    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],  [0, 5, 1],   [0, 1, 7],   [0, 7, 10],  [0, 10, 11],
        [1, 5, 9],   [5, 11, 4],  [11, 10, 2], [10, 7, 6],  [7, 1, 8],
        [3, 9, 4],   [3, 4, 2],   [3, 2, 6],   [3, 6, 8],   [3, 8, 9],
        [4, 9, 5],   [2, 4, 11],  [6, 2, 10],  [8, 6, 7],   [9, 8, 1],
    ];

    use std::collections::HashMap;
    for _ in 0..subdivisions {
        let mut next: Vec<[u32; 3]> = Vec::with_capacity(tris.len() * 4);
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = midpoints.get(&key) {
                return idx;
            }
            let mid = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
            let idx = verts.len() as u32;
            verts.push(mid);
            midpoints.insert(key, idx);
            idx
        };
        for tri in &tris {
            let [a, b, c] = *tri;
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }

    for v in &mut verts {
        *v = v.normalize();
    }

    let radius = 0.5_f32;
    let vertices: Vec<Vertex> = verts
        .iter()
        .map(|p| {
            let n = *p;
            let pos = n * radius;
            Vertex {
                position: [pos.x, pos.y, pos.z],
                normal: [n.x, n.y, n.z],
                uv: [0.0, 0.0],
            }
        })
        .collect();
    let mut indices: Vec<u32> = Vec::with_capacity(tris.len() * 3);
    for tri in &tris {
        indices.extend_from_slice(tri);
    }

    MeshAsset {
        vertices,
        indices,
        aabb_min: Vec3::splat(-radius),
        aabb_max: Vec3::splat(radius),
    }
}
