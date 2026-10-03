// CRT post-process: temporal accumulator + analog sim + scan lines.
//
// First pass of a two-pass CRT chain. Reads the lit scene from
// `chain_input`, blends it with this shader's previous output
// (`self_previous`) so the picture gains a soft motion-blur tail,
// applies analog colour shifts and per-frame jitter, and adds a
// rolling scan line. The result feeds the second pass
// (`crt_display.wgsl`) which handles barrel distortion + chromatic
// aberration + RGB pixel mask + vignette.
//
// Wire it up in your camera component, in chain order:
//   [camera]
//   post_process = [
//       "assets/shaders/crt_buffer.wgsl",
//       "assets/shaders/crt_display.wgsl",
//   ]

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var prev_color: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct PostUniforms {
    resolution: vec2<f32>,
    time: f32,
    delta: f32,
};
@group(1) @binding(0) var<uniform> uniforms: PostUniforms;

struct CrtBufferParams {
    // @default(1.0) @range(0, 10) @label("Analog Amount")
    analog_amount: f32,
    // @default(0.2) @range(0, 1) @label("Jitter Amount")
    jitter_amount: f32,
    // @default(0.3) @range(0, 1) @label("Scan Line Amount")
    scan_line_amount: f32,
};
@group(2) @binding(0) var<uniform> params: CrtBufferParams;

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(c.y));
}

fn rand1(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
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

    // --- Jitter: high-frequency drift + occasional impulse, scaled
    // by `jitter_amount`.
    var jitter = vec2<f32>(0.0, 0.0);
    let hf_amount = 0.0005;
    let hf_time = uniforms.time * 100.0;
    jitter.x = jitter.x + (sin(hf_time + uv.y * 400.0) * 0.5 + 0.5) * hf_amount;
    jitter.y = jitter.y + (sin(hf_time * 0.8 + uv.x * 300.0) * 0.5 + 0.5)
        * hf_amount * 0.3;
    let impulse_amount = 0.0007;
    let impulse_freq = 0.005;
    jitter.x = jitter.x + (rand1(uniforms.time * impulse_freq) - 0.5) * impulse_amount;
    uv = uv + params.jitter_amount * jitter;

    // --- Feedback mix: 50/50 weighted by analog_amount.
    let feedback = 0.5 * params.analog_amount;
    let new_col = textureSample(input_color, samp, uv).rgb;
    let old_col = textureSample(prev_color, samp, uv).rgb;
    var col = new_col * (1.0 - feedback) + old_col * feedback;

    // --- Analog sim: saturation push + S-curve.
    var hsv = rgb2hsv(col);
    hsv.y = hsv.y * (1.0 + 0.05 * params.analog_amount);
    col = hsv2rgb(hsv);
    let s = tanh(1.1 * (col - vec3<f32>(0.5))) + vec3<f32>(0.5);
    col = params.analog_amount * s + (1.0 - params.analog_amount) * col;

    // --- Rolling scan line at panel resolution.
    let alpha = 0.07 * params.scan_line_amount;
    let res_y = uniforms.resolution.y;
    let mag_raw = (30.1 * res_y * uniforms.time + frag.y) - res_y * floor((30.1 * res_y * uniforms.time + frag.y) / res_y);
    let mag = (1.0 - alpha) + alpha * (1.0 - mag_raw / res_y);
    col = col * mag;

    return vec4<f32>(col, 1.0);
}
