//! Shader module creation. Source is always compiled externally (DXC from HLSL
//! is canonical — see README). The RHI takes pre-compiled bytecode; each
//! backend consumes the flavor it speaks (SPIR-V for wgpu/VK, DXIL for D3D12,
//! MSL for Metal, WGSL as a convenience during bringup).

use bitflags::bitflags;

#[derive(Debug, Clone, Copy)]
pub struct ShaderDesc<'a> {
    pub label: Option<&'a str>,
    pub stage: ShaderStage,
    pub entry_point: &'a str,
    pub bytecode: ShaderBytecode<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    /// Gated behind `Features::MESH_SHADERS`.
    Mesh,
    /// Gated behind `Features::MESH_SHADERS` (task shader / amplification).
    Task,
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderBytecode<'a> {
    /// SPIR-V as a `u32` slice. Consumed by wgpu and Vulkan backends.
    Spirv(&'a [u32]),
    /// DXIL bytes. Consumed by the D3D12 backend.
    Dxil(&'a [u8]),
    /// Metal Shading Language source. Consumed by the Metal backend.
    Msl(&'a str),
    /// WGSL source. Useful during bringup on the wgpu backend.
    Wgsl(&'a str),
}

bitflags! {
    /// Set of shader stages, used for bind group visibility and push constant
    /// ranges. Vertex+Fragment is the common case; `ALL_GRAPHICS` covers it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderStageFlags: u32 {
        const VERTEX   = 1 << 0;
        const FRAGMENT = 1 << 1;
        const COMPUTE  = 1 << 2;
        const MESH     = 1 << 3;
        const TASK     = 1 << 4;

        const ALL_GRAPHICS = Self::VERTEX.bits() | Self::FRAGMENT.bits();
        const ALL = Self::VERTEX.bits()
                  | Self::FRAGMENT.bits()
                  | Self::COMPUTE.bits()
                  | Self::MESH.bits()
                  | Self::TASK.bits();
    }
}
