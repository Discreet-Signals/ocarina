// Phase-1 hello-triangle. Vertex positions and per-vertex colors are
// hard-coded; no input buffers needed.

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.60),
        vec2<f32>(-0.60, -0.50),
        vec2<f32>( 0.60, -0.50),
    );
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(0.95, 0.40, 0.25),
        vec3<f32>(0.20, 0.85, 0.35),
        vec3<f32>(0.25, 0.55, 0.95),
    );
    var out: VsOut;
    out.position = vec4<f32>(positions[i], 0.0, 1.0);
    out.color = colors[i];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
