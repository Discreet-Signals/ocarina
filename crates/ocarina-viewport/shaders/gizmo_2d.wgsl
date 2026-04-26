// Screen-space 2D gizmo shader.
//
// Each gizmo vertex carries:
//   - a world-space *anchor* position
//   - a world-space *neighbor* along the same line, used to compute
//     the line's screen-space direction at this vertex
//   - extras.x: signed pixel offset perpendicular to the line at
//     this vertex (positive on one side, negative on the other,
//     0 collapses to a point — used for arrow tips).
//   - extras.y: max half-thickness in pixels around this vertex's
//     local point. Used by the fragment shader to fade alpha at the
//     shape boundary for anti-aliasing — fragments interpolate this
//     plus the signed offset, and we feather alpha at
//     `|signed_offset| ≈ extras.y`.
//
// `viewport_size` is the panel rect's pixel extent (NOT the swapchain
// extent). NDC `[-1,1]` after `set_viewport(...)` maps to that rect,
// so converting pixels → NDC must use the panel's dimensions.

struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    base_color: vec4<f32>,
    /// xy = panel rect in physical pixels.
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
    @location(1) neighbor: vec3<f32>,
    /// x = signed pixel offset; y = max half-thickness in pixels
    /// (≥ |x|), passed through to the fragment for AA.
    @location(2) extras: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    /// Pixel offset interpolated to this fragment.
    @location(0) signed_offset_px: f32,
    /// Max half-thickness at this fragment (interpolated). For a
    /// line quad this is constant (= line_half); for an arrow it
    /// tapers to 0 at the tip.
    @location(1) max_half_px: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = (instance.model * vec4<f32>(in.position, 1.0));
    let world_nb = (instance.model * vec4<f32>(in.neighbor, 1.0));

    let p_clip = camera.view_proj * world_pos;
    let n_clip = camera.view_proj * world_nb;

    let p_ndc = p_clip.xy / max(p_clip.w, 1e-4);
    let n_ndc = n_clip.xy / max(n_clip.w, 1e-4);

    // Convert both endpoints from NDC to *pixel* space first, so the
    // perpendicular we compute is actually perpendicular on screen
    // (NDC is square `[-1, 1]²` while pixel space has the panel's
    // aspect — taking perp in NDC made diagonal lines thicker /
    // thinner than horizontal / vertical ones).
    let half_size = camera.viewport_size.xy * 0.5;
    let p_px = p_ndc * half_size;
    let n_px = n_ndc * half_size;
    let dir_px_raw = n_px - p_px;
    let dir_len = length(dir_px_raw);
    var perp_px = vec2<f32>(0.0, 0.0);
    if (dir_len > 1e-6) {
        let dir_px = dir_px_raw / dir_len;
        perp_px = vec2<f32>(-dir_px.y, dir_px.x);
    }

    // Offset in pixels, then convert back to NDC. 1 px = 2 / panel_px
    // NDC units per axis.
    let offset_px = perp_px * in.extras.x;
    let offset_ndc = offset_px * vec2<f32>(2.0 / camera.viewport_size.x, 2.0 / camera.viewport_size.y);

    var out: VsOut;
    out.clip_position = vec4<f32>(
        p_clip.xy + offset_ndc * p_clip.w,
        p_clip.z,
        p_clip.w,
    );
    out.signed_offset_px = in.extras.x;
    out.max_half_px = in.extras.y;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance from the line center, in pixels (interpolated).
    let dist = abs(in.signed_offset_px);
    // The shape's local half-thickness at this fragment, in pixels.
    // `max(.., 0.5)` keeps the math sane at the arrow tip where
    // both `dist` and `max_half_px` go to 0.
    let edge = max(in.max_half_px, 0.5);
    // 1 px feather inside the boundary — slightly soft edge that
    // reads as a smooth line rather than a stairstep.
    let aa_inner = max(edge - 1.0, 0.0);
    let alpha = 1.0 - smoothstep(aa_inner, edge, dist);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(instance.tint.rgb, instance.tint.a * alpha);
}
