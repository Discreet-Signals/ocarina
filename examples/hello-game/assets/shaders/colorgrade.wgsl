// Sample post-process shader: luma-weighted grayscale.
//
// Demonstrates the bind-group contract user shaders must follow.
// Same shape as the engine's default identity passthrough at
// `crates/ocarina-viewport/shaders/compositor.wgsl` — copy that file
// as a starting point when authoring new effects.
//
// Usage: in a camera entity's component file:
//   [camera]
//   fov_y = 55.0
//   post_process = "assets/shaders/grayscale.wgsl"
//
// Save edits to this file while the editor is running to see them
// applied on the next frame (mtime-based hot-reload).

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct PostUniforms {
    resolution: vec2<f32>,
    time: f32,
    delta: f32,
};
@group(1) @binding(0) var<uniform> uniforms: PostUniforms;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(p[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = textureDimensions(input_color);
    let uv = frag.xy / vec2<f32>(f32(dim.x), f32(dim.y));
    var c = textureSample(input_color, samp, uv);
    c = 1.25 * tanh(1.5 * c);
    return vec4<f32>(c.r, c.g, c.b, c.a);
}
