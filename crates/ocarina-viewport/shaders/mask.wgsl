// Selection mask shader.
//
// Renders the selected mesh as solid 1.0 into a single-channel
// (R8Unorm) screen-sized texture. The outline post-process below
// samples this mask + 4 neighbors to find the silhouette boundary.
//
// Reuses the lit shader's vertex layout + Camera/Instance UBOs so the
// caller can drive this pipeline with the same bind groups it uses
// for the main lit draw — only the fragment output differs.

struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    /// Layout-only padding so the camera UBO matches the
    /// gizmo shader's struct (which uses `viewport_size`).
    viewport_size: vec4<f32>,
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

@vertex
fn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    return camera.view_proj * world_pos;
}

@fragment
fn fs_main() -> @location(0) f32 {
    return 1.0;
}
