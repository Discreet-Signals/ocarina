// Half-Lambert "ray-traced" scene shader.
//
// Honest disclaimer: this is a fragment shader running in the lit
// pipeline. It has no scene geometry to trace against (we'd need
// a depth-buffer binding or a scene BVH for real RT), so the
// "multiple reflections" here are an art-trick: each "bounce"
// samples a procedural sky color along a perturbed reflection
// vector, weighted by a diminishing Schlick fresnel. The result
// looks like multi-bounce GI off a glossy surface without any
// actual ray traversal.
//
// What's real:
//   - Half-Lambert direct lighting summed over every light in the
//     world (DirectionalLight + PointLight + SpotLight).
//   - Schlick fresnel on the reflection vector.
//   - Per-bounce blur: each successive bounce drifts the sample
//     direction toward the surface normal (a cheap stand-in for an
//     integrated cone of sky directions).
//
// To use it, point a camera at this file:
//   [camera]
//   scene_shader = "assets/shaders/half_lambert_rt.wgsl"

const MAX_LIGHTS: u32 = 16u;

const KIND_DIRECTIONAL: u32 = 0u;
const KIND_POINT: u32 = 1u;
const KIND_SPOT: u32 = 2u;

const PI: f32 = 3.14159265358979;

// Number of synthetic "reflection bounces" to accumulate. Each
// bounce mixes the procedural sky into the specular term using a
// fresnel weight that decays with bounce index.
const BOUNCES: i32 = 4;

// --- Engine bindings (must match the lit pipeline) ---

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    viewport_size: vec4<f32>,
};

struct Light {
    // (kind, _pad, _pad, _pad)
    kind_pad: vec4<u32>,
    color: vec4<f32>,
    position: vec4<f32>,
    direction: vec4<f32>,
    // (range, inner_cos, outer_cos, _pad)
    params: vec4<f32>,
};

struct Lights {
    count_pad: vec4<u32>,
    lights: array<Light, MAX_LIGHTS>,
};

struct Instance {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};

struct Material {
    albedo: vec4<f32>,
    emission: vec4<f32>,
    // (metallic, roughness, emission_strength, _pad)
    metal_rough: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var samp: sampler;

@group(1) @binding(0) var<uniform> instance: Instance;
@group(1) @binding(1) var<uniform> material: Material;

@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var normal_tex: texture_2d<f32>;
@group(2) @binding(2) var mr_tex: texture_2d<f32>;
@group(2) @binding(3) var emission_tex: texture_2d<f32>;

// --- Vertex ---

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos4 = instance.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_normal = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}

// --- Helpers ---

// Half-Lambert: dot(N,L) * 0.5 + 0.5, then squared for a softer
// wraparound. Used in Half-Life 2 (Valve) for diffuse that doesn't
// go fully black on the shadow side.
fn half_lambert(n: vec3<f32>, l: vec3<f32>) -> f32 {
    let h = dot(n, l) * 0.5 + 0.5;
    return h * h;
}

// Per-light attenuation: directional = 1.0; point/spot = inverse-
// square clamped by `range`; spot = additionally cone-falloff.
fn light_attenuation(light: Light, world_pos: vec3<f32>) -> vec3<f32> {
    let kind = light.kind_pad.x;
    if (kind == KIND_DIRECTIONAL) {
        return vec3<f32>(1.0);
    }
    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let range = max(light.params.x, 0.0001);
    let radial = clamp(1.0 - dist / range, 0.0, 1.0);
    var atten = radial * radial;
    if (kind == KIND_SPOT) {
        let l = to_light / dist;
        let cos_to_pixel = dot(-light.direction.xyz, l);
        let inner = light.params.y;
        let outer = light.params.z;
        let denom = max(inner - outer, 1.0e-4);
        let cone = clamp((cos_to_pixel - outer) / denom, 0.0, 1.0);
        atten = atten * cone;
    }
    return vec3<f32>(atten);
}

// Returns the unit-direction FROM the surface toward the light.
// Directional lights store `direction` as the direction the light
// travels (sun pointing down), so the to-light vector is `-direction`.
fn light_direction(light: Light, world_pos: vec3<f32>) -> vec3<f32> {
    let kind = light.kind_pad.x;
    if (kind == KIND_DIRECTIONAL) {
        return normalize(-light.direction.xyz);
    }
    return normalize(light.position.xyz - world_pos);
}

// Schlick approximation of Fresnel reflectance. F0 is the
// reflectance at normal incidence (~0.04 for dielectrics).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let k = clamp(1.0 - cos_theta, 0.0, 1.0);
    let k2 = k * k;
    return f0 + (vec3<f32>(1.0) - f0) * (k2 * k2 * k);
}

// Procedural sky-ish gradient queried along a direction. No real
// scene to reflect, so this stands in for an environment map: blue
// up, warm horizon, dim down. Modulated by sun color through the
// directional light if one is present.
fn procedural_sky(dir: vec3<f32>) -> vec3<f32> {
    let up_amount = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.18, 0.32, 0.55);
    let horizon = vec3<f32>(0.65, 0.55, 0.42);
    let nadir = vec3<f32>(0.04, 0.04, 0.05);
    let sky = mix(horizon, zenith, smoothstep(0.0, 1.0, up_amount));
    return mix(nadir, sky, smoothstep(0.0, 0.4, up_amount));
}

// --- Fragment ---

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(camera.eye_pos.xyz - in.world_pos);

    // Material values, with texture multiplies. Defaults already
    // baked into the textures (white) so a textureless material is
    // effectively the scalar values alone.
    let tex_albedo = textureSample(albedo_tex, samp, in.uv).rgb;
    let albedo = material.albedo.rgb * tex_albedo * instance.tint.rgb;
    let mr = textureSample(mr_tex, samp, in.uv);
    let metallic = clamp(material.metal_rough.x * mr.b, 0.0, 1.0);
    let roughness = clamp(material.metal_rough.y * mr.g, 0.04, 1.0);
    let emission_tex_rgb = textureSample(emission_tex, samp, in.uv).rgb;
    let emission = material.emission.rgb
        * emission_tex_rgb
        * material.metal_rough.z;

    // F0: 0.04 baseline for dielectrics, lerping toward the
    // albedo for metals (standard PBR trick).
    let f0 = mix(vec3<f32>(0.04), albedo, vec3<f32>(metallic));

    // --- Direct lighting: half-lambert summed over every light.
    var direct = vec3<f32>(0.0);
    let count = lights.count_pad.x;
    for (var i = 0u; i < count; i = i + 1u) {
        let light = lights.lights[i];
        let l = light_direction(light, in.world_pos);
        let atten = light_attenuation(light, in.world_pos);
        let wrap = half_lambert(n, l);
        direct = direct + albedo * light.color.rgb * atten * wrap;
    }

    // --- Multi-bounce specular: synthetic environment sampling.
    // Each bounce samples the procedural sky along a reflection
    // vector that drifts deterministically toward the surface
    // normal, standing in for an integrated cone of sky directions.
    // Fresnel decays with bounce index so the sum converges.
    let r0 = reflect(-v, n);
    let cos_v = max(dot(n, v), 0.0);
    var spec = vec3<f32>(0.0);
    var fr_weight = vec3<f32>(1.0);
    for (var b = 0; b < BOUNCES; b = b + 1) {
        // Per-bounce drift: bounce 0 is the mirror direction;
        // later bounces blend toward the normal in proportion to
        // roughness. Smooth and screen-space coherent, so no noise.
        let drift = roughness * f32(b) / f32(BOUNCES);
        let r = normalize(mix(r0, n, drift));
        let env = procedural_sky(r);

        // Schlick fresnel evaluated on the bounce vector against
        // the surface normal. Decays per bounce so the sum is
        // bounded.
        let cos_b = max(dot(n, r), 0.0);
        let fr = fresnel_schlick(cos_b, f0);
        spec = spec + fr_weight * fr * env;
        fr_weight = fr_weight * (1.0 - fr) * 0.6;
    }

    // --- Ambient floor: keep shadow-side pixels from going jet
    // black on a flat-lit scene with one directional light.
    let ambient = procedural_sky(n) * albedo * 0.08;

    var color = direct + spec + ambient + emission;

    // Reinhard tone map + manual sRGB encode (the swapchain texture
    // for the editor's Vger-shared surface is linear, so the engine
    // does sRGB in-shader to match the runtime's behaviour).
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, material.albedo.a);
}
