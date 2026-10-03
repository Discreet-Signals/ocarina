// CRT post-process: barrel distortion + RGB pixel mask + chromatic
// aberration + vignette.
//
// Second pass of a two-pass CRT chain. Reads the temporal-blurred
// signal from `input_color` (= `crt_buffer.wgsl`'s output this frame)
// and applies the spatial CRT effects: lens warp, sub-pixel mask,
// per-channel colour fringing, screen vignette.
//
// Wire it up in your camera component, in chain order:
//   [camera]
//   post_process = [
//       "assets/shaders/crt_buffer.wgsl",
//       "assets/shaders/crt_display.wgsl",
//   ]

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct PostUniforms {
    resolution: vec2<f32>,
    time: f32,
    delta: f32,
};
@group(1) @binding(0) var<uniform> uniforms: PostUniforms;

struct CrtDisplayParams {
    // @default(1.0) @range(0, 1) @label("Pixel Amount")
    pixel_amount: f32,
    // @default(0.35) @range(0, 5) @label("Aberration Amount")
    aberration_amount: f32,
    // @default(1.0) @range(0, 4) @label("Distortion")
    distortion: f32,
    // naga inserts 4 bytes of implicit padding here to land
    // `resolution_crt` on its 8-byte alignment.
    // @default([512.0, 320.0]) @label("CRT Resolution") // [1024.0, 640.0]
    resolution_crt: vec2<f32>,
};
@group(2) @binding(0) var<uniform> params: CrtDisplayParams;

fn vignette(uv: vec2<f32>, strength: f32) -> f32 {
    let center = uv - vec2<f32>(0.5);
    let dist = length(center) * 1.414;
    return 1.0 - smoothstep(0.7, 1.0, dist) * strength;
}

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
    var uv = frag.xy / vec2<f32>(f32(dim.x), f32(dim.y));

    // --- Barrel distortion: aspect-corrected radial warp.
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    var p = uv * 2.0 - vec2<f32>(1.0);
    p.x = p.x * aspect;
    let r2 = dot(p, p);
    let warp = 0.01 * params.distortion;
    p = p * (1.0 + warp * r2);
    p.x = p.x / aspect;
    uv = (p + vec2<f32>(1.0)) * 0.5;

    // --- Pixel sim: snap UV to the CRT pixel grid.
    let res_crt = params.resolution_crt;
    let pos_crt = uv * res_crt;
    uv = (floor(uv * res_crt) + vec2<f32>(0.5)) / res_crt;

    // --- Chromatic aberration: per-channel UV offsets.
    let uv_r = uv + vec2<f32>(0.001, 0.0) * params.aberration_amount;
    let uv_g = uv + vec2<f32>(-0.0005, 0.0) * params.aberration_amount;
    let uv_b = uv + vec2<f32>(0.0, 0.0001) * params.aberration_amount;

    var col = vec3<f32>(0.0);
    if (all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
        col.r = textureSample(input_color, samp, uv_r).r;
        col.g = textureSample(input_color, samp, uv_g).g;
        col.b = textureSample(input_color, samp, uv_b).b;
    }

    // --- RGB sub-pixel mask: 3-wide stripes that shift per row so
    // the "phosphors" form a triangular pattern under a magnifier.
    var mask = vec3<f32>(1.0 - params.pixel_amount);
    let row_y = pos_crt.y - 3.0 * floor(pos_crt.y / 3.0);
    let row = i32(row_y);
    let sub_x = (3.0 * pos_crt.x + f32(row));
    let sub_mod = sub_x - 3.0 * floor(sub_x / 3.0);
    let sub = i32(sub_mod);
    if (sub == 0) { mask.r = 1.0; }
    else if (sub == 1) { mask.g = 1.0; }
    else { mask.b = 1.0; }

    let vig = vignette(uv, 0.2);
    return vec4<f32>(col * mask * vig, 1.0);
}
