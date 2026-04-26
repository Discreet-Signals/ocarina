// Inverted-hull outline shader.
//
// Pass 1 (lit.wgsl) draws the mesh normally and writes depth.
// Pass 2 (this shader) draws the same mesh with cull_mode = Front
// (so only back faces show), each vertex pushed a small distance
// outward along its world-space normal, depth_compare = LessEqual,
// and depth_write_enabled = false. The expanded back-facing shell
// extends slightly beyond the mesh's silhouette in screen space;
// where it overlaps the original mesh the depth test rejects it,
// where it extends past the silhouette the empty far-plane depth
// lets it draw — the result is a clean silhouette outline with no
// stencil buffer or post-process pass required.
//
// Reuses the same Camera + Instance UBO layouts as lit.wgsl so the
// caller can drive both pipelines with one set of bind groups.

struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
};

struct Instance {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> instance: Instance;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
};

// Outline thickness in world units. Stays constant in world space; the
// silhouette will appear thinner as the camera pulls back. Trade-off
// vs. screen-space expansion (which needs viewport size in the shader)
// — for the editor's typical zoom range this looks fine.
const OUTLINE_OFFSET: f32 = 0.04;

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = (instance.model * vec4<f32>(in.position, 1.0)).xyz;
    let world_normal = normalize((instance.model * vec4<f32>(in.normal, 0.0)).xyz);
    let pushed = world_pos + world_normal * OUTLINE_OFFSET;
    out.clip_position = camera.view_proj * vec4<f32>(pushed, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Amber/orange — high contrast against neutral cool greys and the
    // editor's dark background. Solid (no shading) so the outline reads
    // as UI overlay, not part of the lit scene.
    return vec4<f32>(1.00, 0.65, 0.20, 1.0);
}
