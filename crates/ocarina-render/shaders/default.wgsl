// Default scene shader: PBR (Cook-Torrance microfacet) over the
// lights buffer, sampling albedo / emission / metallic-roughness
// textures plus their per-instance scalar multipliers.
//
// Normal mapping is intentionally absent: the shared vertex format
// (`Vertex` in mesh.rs) carries position + normal + uv but no
// tangent, which is the prerequisite for tangent-space normal
// mapping. The normal-map slot at @group(2) @binding(1) is still
// declared so the bind group layout stays uniform across
// alternative scene shaders (a user-supplied shader with vertex
// tangents *can* read it). We just don't sample it here.

const MAX_LIGHTS: u32 = 16u;

const KIND_DIRECTIONAL: u32 = 0u;
const KIND_POINT: u32 = 1u;
const KIND_SPOT: u32 = 2u;

const PI: f32 = 3.14159265358979;

// ─── Uniforms ───

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    viewport_size: vec4<f32>,
};

struct Light {
    kind_pad: vec4<u32>,
    color: vec4<f32>,
    position: vec4<f32>,
    direction: vec4<f32>,
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

// ─── Vertex ───

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
    let n = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_normal = n;
    out.uv = in.uv;
    return out;
}

// ─── PBR helpers ───

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn distance_attenuation(d: f32, range: f32) -> f32 {
    if (range <= 0.0 || d >= range) {
        return 0.0;
    }
    let inv_sq = 1.0 / max(d * d, 0.0001);
    let t = clamp(1.0 - (d / range), 0.0, 1.0);
    return inv_sq * (t * t);
}

/// Linear → sRGB encode (piecewise curve, more accurate than
/// `pow(c, 1/2.2)` in the toe). The renderer targets non-sRGB
/// surfaces (Bgra8Unorm / Rgba8Unorm) so this gamma encoding has to
/// happen here rather than as a hardware sRGB-format conversion.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < cutoff);
}

// ─── Fragment ───

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(camera.eye_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 0.0);

    // ─── Material samples ───
    // `instance.tint` is read by the simple pipelines (mask / gizmo
    // / unlit) but not here -- the default scene path goes through
    // `material.albedo`, which the renderer mirrors `tint` into for
    // every entity. Editor selection highlight is applied to both
    // upstream so a selected entity boosts its colour through both
    // shaders' paths.
    let albedo_sample = textureSample(albedo_tex, samp, in.uv);
    let albedo = albedo_sample.rgb * material.albedo.rgb;
    let alpha = albedo_sample.a * material.albedo.a;

    // glTF MR convention: green = roughness, blue = metallic.
    let mr_sample = textureSample(mr_tex, samp, in.uv);
    let metallic = clamp(mr_sample.b * material.metal_rough.x, 0.0, 1.0);
    let roughness = clamp(mr_sample.g * material.metal_rough.y, 0.04, 1.0);

    let emission_sample = textureSample(emission_tex, samp, in.uv);
    let emission = emission_sample.rgb * material.emission.rgb * material.emission.a;

    // ─── F0 (reflectance at normal incidence) ───
    // Dielectrics: 0.04. Metals: albedo. Linear blend on metallic.
    let f0_dielectric = vec3<f32>(0.04);
    let f0 = mix(f0_dielectric, albedo, metallic);

    // ─── Direct lighting ───
    var color = vec3<f32>(0.0);

    let count = lights.count_pad.x;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let light = lights.lights[i];
        let kind = light.kind_pad.x;
        let light_color = light.color.rgb;
        let intensity = light.color.a;

        var l: vec3<f32>;
        var atten: f32 = 1.0;

        if (kind == KIND_DIRECTIONAL) {
            l = normalize(-light.direction.xyz);
        } else if (kind == KIND_POINT) {
            let to_light = light.position.xyz - in.world_pos;
            let d = length(to_light);
            if (d < 1e-4) {
                continue;
            }
            l = to_light / d;
            atten = distance_attenuation(d, light.params.x);
        } else if (kind == KIND_SPOT) {
            let to_light = light.position.xyz - in.world_pos;
            let d = length(to_light);
            if (d < 1e-4) {
                continue;
            }
            l = to_light / d;
            atten = distance_attenuation(d, light.params.x);
            let cos_theta = dot(normalize(light.direction.xyz), -l);
            let inner_cos = light.params.y;
            let outer_cos = light.params.z;
            atten = atten * smoothstep(outer_cos, inner_cos, cos_theta);
        } else {
            continue;
        }

        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l <= 0.0 || atten <= 0.0) {
            continue;
        }

        let h = normalize(l + v);
        let n_dot_h = max(dot(n, h), 0.0);
        let l_dot_h = max(dot(l, h), 0.0);

        let radiance = light_color * (intensity * atten);

        // Cook-Torrance specular term: D * G * F / (4 * NdotV * NdotL).
        let d_term = distribution_ggx(n_dot_h, roughness);
        let g_term = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f_term = fresnel_schlick(l_dot_h, f0);

        let specular_num = d_term * g_term * f_term;
        let specular_den = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular = specular_num / specular_den;

        // Energy conservation: diffuse fraction = (1 - F)(1 - metallic).
        let kd = (vec3<f32>(1.0) - f_term) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;

        color = color + (diffuse + specular) * radiance * n_dot_l;
    }

    // ─── Ambient + emission ───
    // Tiny ambient floor so unlit hemispheres aren't pure black until
    // a proper IBL / GI pass lands.
    let ambient = vec3<f32>(0.03) * albedo;
    color = color + ambient + emission;

    // ─── Reinhard tone map for HDR-friendly output. A future
    // post-processing chain (Camera.post_process) is the right home
    // for tone mapping, but until that ships we apply it here so HDR
    // emission values don't blow out as pure white. ───
    let mapped = color / (vec3<f32>(1.0) + color);

    return vec4<f32>(linear_to_srgb(mapped), alpha);
}
