// Flat-shaded lit mesh shader. Consumes interleaved position/normal/uv
// vertex data, a camera uniform at (0, 0), and a per-instance uniform at
// (1, 0) carrying the model matrix + tint. The instance UBO is bound
// with a dynamic offset so a single draw loop can stride through many
// per-entity records in one buffer.

struct Camera {
    view_proj: mat4x4<f32>,
    // xyz = direction *to* the light (pre-normalized): the vector you
    // would draw from a surface up to the sun. Lambert is `max(dot(n, l),
    // 0)`, so a positive-Y light_dir lights upward-facing surfaces.
    // w unused.
    light_dir: vec4<f32>,
    // rgb = base color, a unused
    base_color: vec4<f32>,
};

struct Instance {
    model: mat4x4<f32>,
    // rgba multiplier applied on top of camera.base_color
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
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    // Normal is rotated by the model matrix's upper-3x3. Approximated by
    // multiplying as a direction (w = 0), which is correct for uniform
    // scale and good enough for the placeholder cubes Phase 4 renders.
    let n = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_normal = n;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(camera.light_dir.xyz);

    let lambert = max(dot(n, l), 0.0);
    let up = max(n.y, 0.0);
    let down = max(-n.y, 0.0);
    let ambient = vec3<f32>(0.18, 0.22, 0.30) * up + vec3<f32>(0.10, 0.08, 0.06) * down + vec3<f32>(0.06);

    let base = camera.base_color.rgb * instance.tint.rgb;
    let color = base * (ambient + vec3<f32>(lambert * 0.85));

    let view_dir = normalize(vec3<f32>(0.0, 0.4, 1.0));
    let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0) * 0.18;
    let final_color = color + vec3<f32>(fresnel * 0.9, fresnel * 0.8, fresnel * 0.4);

    return vec4<f32>(final_color, 1.0);
}
