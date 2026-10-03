// Screen-space silhouette outline.
//
// Reads the selection mask written by `mask.wgsl` and emits an outline
// color where the mask transitions from 1 → 0 (or 0 → 1). A 4-tap
// neighbor sample is enough to find a 1-pixel-thick contour around
// any rendered shape — including hard-edged cubes — without adding
// any geometry or relying on per-vertex normals.
//
// Sampling at +/- `OUTLINE_THICKNESS_PX` texels gives a thicker line
// than 1 px, which reads better at high-DPI editor sizes.
//
// The fragment alpha-blends over the lit scene already in the color
// attachment (LoadOp::Load on the swapchain), so the outline appears
// painted on top of whatever the lit pass and Floem rendered first.

@group(0) @binding(0) var mask_tex: texture_2d<f32>;
@group(0) @binding(1) var mask_sampler: sampler;

// Two triangles forming a fullscreen quad in clip-space, generated
// from `@builtin(vertex_index)` so no vertex buffer is needed.
// The viewport+scissor on the render pass restricts rasterization to
// the panel rect inside the swapchain — clip-space `[-1, 1]²` maps to
// that panel rect, not the whole window.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    return vec4<f32>(positions[vid], 0.0, 1.0);
}

const OUTLINE_THICKNESS_PX: f32 = 2.0;
const OUTLINE_COLOR: vec3<f32> = vec3<f32>(1.00, 0.65, 0.20);

/// Linear → sRGB encode. The renderer targets non-sRGB surfaces, so
/// gamma encoding has to happen here rather than as a hardware
/// sRGB-format conversion.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < cutoff);
}

// Derive the sample UV from `@builtin(position)` — the rasterizer's
// screen-space pixel coordinate after viewport transform. Dividing by
// the mask texture's dimensions gives the UV of THIS PIXEL in the
// mask, which lines up with where the mask pass actually wrote.
//
// Using interpolated quad UVs would be wrong here: the quad is drawn
// inside the panel viewport, but its `[0, 1]` UV range stretches over
// the whole quad, not the whole texture. Fragments at the panel's
// top-left would read the *swapchain's* top-left of the mask, which
// is somewhere else entirely (or empty), so the outline would float
// off-position relative to the mesh.
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(mask_tex));
    let uv = frag_coord.xy / dim;
    let texel = vec2<f32>(OUTLINE_THICKNESS_PX, OUTLINE_THICKNESS_PX) / dim;

    let center = textureSample(mask_tex, mask_sampler, uv).r;
    let n = textureSample(mask_tex, mask_sampler, uv + vec2<f32>( 0.0, -texel.y)).r;
    let s = textureSample(mask_tex, mask_sampler, uv + vec2<f32>( 0.0,  texel.y)).r;
    let e = textureSample(mask_tex, mask_sampler, uv + vec2<f32>( texel.x, 0.0)).r;
    let w = textureSample(mask_tex, mask_sampler, uv + vec2<f32>(-texel.x, 0.0)).r;

    // Edge intensity = max(neighbors, center) - min(neighbors, center).
    // Non-zero wherever the mask transitions across the kernel — i.e.,
    // the silhouette of the selected mesh in screen space.
    let max_n = max(max(n, s), max(e, w));
    let min_n = min(min(n, s), min(e, w));
    let edge = max(max_n, center) - min(min_n, center);

    // Threshold + smooth ramp. Below 0.1 the edge is too weak to
    // bother (anti-alias halo from filtering); above 0.5 fully solid.
    let alpha = smoothstep(0.1, 0.5, edge);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(linear_to_srgb(OUTLINE_COLOR), alpha);
}
