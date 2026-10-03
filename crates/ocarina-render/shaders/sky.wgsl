// 3-stop gradient skybox.
//
// Renders a single oversized triangle that fills the colour
// attachment with a world-space gradient sampled by the per-pixel
// view direction. No vertex buffer; three vertices generated from
// `@builtin(vertex_index)`. The triangle extends past the viewport
// on two edges so the rasteriser clips cleanly to the visible
// `[-1, 1]²` area, with no diagonal seam to generate redundant
// helper-quad shading.
//
// World-direction reconstruction: each pixel's NDC coordinate +
// far-plane depth (z = 1) maps through `inv_view_proj` to a world
// position. Subtracting the camera position and normalising gives
// the ray direction, which is what we use to blend top / horizon /
// bottom colours. This is the load-bearing piece for "looking
// straight down shows only bottom" -- the gradient follows the world
// up vector, not screen Y.

struct Sky {
    inv_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    top_color: vec4<f32>,
    horizon_color: vec4<f32>,
    bottom_color: vec4<f32>,
    /// x = horizon_falloff, y = intensity.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sky: Sky;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single oversized triangle that overhangs the viewport on two
    // edges. Clipping handles the trim; no shared diagonal means
    // fewer 2x2 helper-quad fragment invocations vs a two-triangle
    // quad.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.clip_position = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

/// Linear → sRGB encode (piecewise sRGB curve, matching the default
/// scene shader). The renderer targets non-sRGB surfaces so we do
/// the conversion in-shader.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < cutoff);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space ray direction at this pixel.
    // Far plane in NDC is z = 1 in wgpu (which uses 0..1 depth).
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world_h = sky.inv_view_proj * clip;
    let world = world_h.xyz / world_h.w;
    let dir = normalize(world - sky.eye_pos.xyz);

    // Y is the world up axis. dir.y > 0 looks above the horizon,
    // < 0 below. Square the falloff so a single scalar covers both
    // halves and "horizon_falloff = 2" gives a tight horizon.
    let falloff = max(sky.params.x, 0.001);
    let above = pow(clamp(dir.y, 0.0, 1.0), 1.0 / falloff);
    let below = pow(clamp(-dir.y, 0.0, 1.0), 1.0 / falloff);

    // Above-horizon: blend horizon → top by `above`. Below: blend
    // horizon → bottom by `below`. At the exact horizon line both
    // are 0 so we stay on `horizon_color`, which is the right
    // semantic.
    let upper = mix(sky.horizon_color.rgb, sky.top_color.rgb, above);
    let lower = mix(sky.horizon_color.rgb, sky.bottom_color.rgb, below);
    let color = select(lower, upper, dir.y >= 0.0);

    let intensity = sky.params.y;
    let final_color = color * intensity;

    return vec4<f32>(linear_to_srgb(final_color), 1.0);
}
