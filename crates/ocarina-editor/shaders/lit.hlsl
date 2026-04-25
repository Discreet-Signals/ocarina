// Flat-shaded lit mesh shader — HLSL authoring.
//
// Compiled to SPIR-V by DXC at startup and handed to the wgpu backend via
// `ShaderBytecode::Spirv`. Behavior is equivalent to shaders/lit.wgsl; the
// HLSL version is the canonical authoring form. If DXC isn't available the
// editor falls back to the WGSL.

cbuffer Camera : register(b0, space0)
{
    float4x4 view_proj;
    // xyz = direction *to* the light (vector from surface up to the
    // sun). Lambert is max(dot(n, l), 0), so positive Y here lights
    // tops of objects rather than their undersides.
    float4   light_dir;   // w = 0
    float4   base_color;  // rgb = albedo, a = 1
};

cbuffer Instance : register(b0, space1)
{
    float4x4 model;       // entity world transform
    float4   tint;        // rgba multiplier on top of base_color
};

struct VsIn
{
    float3 position : POSITION0;
    float3 normal   : NORMAL0;
    float2 uv       : TEXCOORD0;
};

struct VsOut
{
    float4 position     : SV_Position;
    float3 world_normal : NORMAL0;
    float2 uv           : TEXCOORD0;
};

VsOut vs_main(VsIn vin)
{
    VsOut vout;
    float4 world_pos = mul(model, float4(vin.position, 1.0));
    vout.position    = mul(view_proj, world_pos);
    // Direction-mode multiply rotates normals by the model's upper-3x3.
    // Approximate (no inverse-transpose) — fine for the uniform-scale
    // placeholder cubes that Phase 4 renders.
    vout.world_normal = mul(model, float4(vin.normal, 0.0)).xyz;
    vout.uv           = vin.uv;
    return vout;
}

float4 fs_main(VsOut pin) : SV_Target0
{
    float3 n = normalize(pin.world_normal);
    float3 l = normalize(light_dir.xyz);

    float lambert = max(dot(n, l), 0.0);
    float up   = max( n.y, 0.0);
    float down = max(-n.y, 0.0);
    float3 ambient = float3(0.18, 0.22, 0.30) * up
                   + float3(0.10, 0.08, 0.06) * down
                   + float3(0.06, 0.06, 0.06);

    float3 base  = base_color.rgb * tint.rgb;
    float3 color = base * (ambient + float3(lambert * 0.85, lambert * 0.85, lambert * 0.85));

    float3 view_dir = normalize(float3(0.0, 0.4, 1.0));
    float  fresnel  = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0) * 0.18;
    float3 final_color = color + float3(fresnel * 0.9, fresnel * 0.8, fresnel * 0.4);

    return float4(final_color, 1.0);
}
