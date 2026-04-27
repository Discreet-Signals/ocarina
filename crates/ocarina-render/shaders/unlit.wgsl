// Unlit variant of the default scene shader. Editor-only: lets the
// user toggle "lighting off" to see the model under flat reference
// shading without scrapping the lit pipeline.
//
// Uses the *simple* pipeline layout (no material UBO, no material
// textures). The lights / sampler bindings on group 0 are declared
// for layout compatibility with the default scene shader -- the
// frame bind group is shared across pipelines -- and ignored.

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

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.uv = in.uv;
    return out;
}

/// Linear → sRGB encode. The renderer targets non-sRGB surfaces, so
/// gamma encoding has to happen here rather than as a hardware
/// sRGB-format conversion. See `default.wgsl` for the rationale.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < cutoff);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(instance.tint.rgb), 1.0);
}
