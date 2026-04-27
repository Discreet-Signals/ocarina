// Unlit variant of the lit shader. Same vertex layout, same UBO
// bindings, no lighting term. Editor-only: lets the user toggle
// "lighting off" to see flat reference shading without scrapping
// the lit pipeline.

struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = camera.base_color.rgb * instance.tint.rgb;
    return vec4<f32>(color, 1.0);
}
