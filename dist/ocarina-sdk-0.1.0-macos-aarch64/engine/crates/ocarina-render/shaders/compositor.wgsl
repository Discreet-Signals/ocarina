// Default post-process compositor: identity passthrough.
//
// Runs as a fullscreen triangle over the swapchain texture, sampling
// `input_color` at the matching pixel and writing it back unchanged.
// This is what the renderer uses when no `post_process = [...]` is
// bound to the camera. User-authored post-process shaders implement
// the same vs_main / fs_main signatures.
//
// Engine-managed bindings (the renderer hooks these up automatically):
//   group 0 binding 0: input_color  texture_2d<f32>  scene on the first pass,
//                                                     previous pass's output otherwise
//   group 0 binding 1: prev_color   texture_2d<f32>  this shader's last-frame output
//                                                     (drives feedback effects)
//   group 0 binding 2: samp         sampler          filtering
//   group 1 binding 0: uniforms     PostUniforms     resolution / time / delta
//
// User-managed bindings (declared by the user shader, allocated by
// the renderer via WGSL reflection):
//   group 2 binding 0: var<uniform> any struct       custom params (defaults from
//                                                     `// @default(...)` magic comments)
//   group 2 binding 1+: storage texture              persistent scratch (read_write)
//
// `PostUniforms` carries:
//   - resolution: panel size in physical pixels
//   - time:       seconds since renderer init
//   - delta:      seconds since the previous frame

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var prev_color: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct PostUniforms {
    resolution: vec2<f32>,
    time: f32,
    delta: f32,
};
@group(1) @binding(0) var<uniform> uniforms: PostUniforms;

// Three-vertex fullscreen triangle. No vertex buffer; the vertex
// shader generates positions from `@builtin(vertex_index)`. Triangle
// extends past the viewport on two edges so the rasteriser clips
// cleanly to the panel rect.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // `input_color` is sized to the full swapchain attachment, so a
    // 1:1 sample at `frag_coord.xy` returns exactly what the lit /
    // outline / gizmo passes wrote at the panel rect.
    let dim = textureDimensions(input_color);
    let uv = frag_coord.xy / vec2<f32>(f32(dim.x), f32(dim.y));
    return textureSample(input_color, samp, uv);
}
