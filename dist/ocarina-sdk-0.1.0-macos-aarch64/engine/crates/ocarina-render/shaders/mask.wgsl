// Selection mask shader.
//
// Renders the selected mesh as solid 1.0 into a single-channel
// (R8Unorm) screen-sized texture. The outline post-process below
// samples this mask + 4 neighbors to find the silhouette boundary.
//
// Uses the simple pipeline layout: frame bind group at group 0
// (camera + lights + sampler, declared for compatibility but only
// `view_proj` is read) and the simple instance bind group at
// group 1 (just the instance UBO, no material).

const MAX_LIGHTS: u32 = 16u;

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    viewport_size: vec4<f32>,
};

struct Light {
    kind_pad: vec4<u32>,
    color: vec4<f32>,
    position: vec4<f32>,
    direction: vec4<f32>,
    params: vec4<f32>,
};

struct Lights {
    count_pad: vec4<u32>,
    lights: array<Light, MAX_LIGHTS>,
};

struct Instance {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var samp: sampler;

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
