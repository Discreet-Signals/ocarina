//! RHI → wgpu type conversions.
//!
//! Straightforward enum/bitflag mapping. Kept exhaustive (no `_ => ...`
//! fallbacks) so that adding a new variant to the RHI is a compile error
//! here until it's translated.

use ocarina_rhi as rhi;

// ────────── Formats ──────────

pub fn texture_format(fmt: rhi::TextureFormat) -> wgpu::TextureFormat {
    use rhi::TextureFormat as R;
    use wgpu::TextureFormat as W;
    match fmt {
        R::R8Unorm => W::R8Unorm,
        R::R8Snorm => W::R8Snorm,
        R::R8Uint => W::R8Uint,
        R::R8Sint => W::R8Sint,
        R::Rg8Unorm => W::Rg8Unorm,
        R::Rg8Snorm => W::Rg8Snorm,
        R::Rg8Uint => W::Rg8Uint,
        R::Rg8Sint => W::Rg8Sint,
        R::Rgba8Unorm => W::Rgba8Unorm,
        R::Rgba8UnormSrgb => W::Rgba8UnormSrgb,
        R::Rgba8Snorm => W::Rgba8Snorm,
        R::Rgba8Uint => W::Rgba8Uint,
        R::Rgba8Sint => W::Rgba8Sint,
        R::Bgra8Unorm => W::Bgra8Unorm,
        R::Bgra8UnormSrgb => W::Bgra8UnormSrgb,

        R::R16Float => W::R16Float,
        R::Rg16Float => W::Rg16Float,
        R::Rgba16Float => W::Rgba16Float,
        R::R16Uint => W::R16Uint,
        R::R16Sint => W::R16Sint,
        R::Rg16Uint => W::Rg16Uint,
        R::Rg16Sint => W::Rg16Sint,
        R::Rgba16Uint => W::Rgba16Uint,
        R::Rgba16Sint => W::Rgba16Sint,

        R::R32Float => W::R32Float,
        R::Rg32Float => W::Rg32Float,
        R::Rgba32Float => W::Rgba32Float,
        R::R32Uint => W::R32Uint,
        R::Rg32Uint => W::Rg32Uint,
        R::Rgba32Uint => W::Rgba32Uint,
        R::R32Sint => W::R32Sint,
        R::Rg32Sint => W::Rg32Sint,
        R::Rgba32Sint => W::Rgba32Sint,

        R::Rgb10A2Unorm => W::Rgb10a2Unorm,
        R::Rg11B10Float => W::Rg11b10Ufloat,

        R::Depth16Unorm => W::Depth16Unorm,
        R::Depth24Plus => W::Depth24Plus,
        R::Depth24PlusStencil8 => W::Depth24PlusStencil8,
        R::Depth32Float => W::Depth32Float,
        R::Depth32FloatStencil8 => W::Depth32FloatStencil8,

        R::Bc1RgbaUnorm => W::Bc1RgbaUnorm,
        R::Bc1RgbaUnormSrgb => W::Bc1RgbaUnormSrgb,
        R::Bc3RgbaUnorm => W::Bc3RgbaUnorm,
        R::Bc3RgbaUnormSrgb => W::Bc3RgbaUnormSrgb,
        R::Bc5RgUnorm => W::Bc5RgUnorm,
        R::Bc5RgSnorm => W::Bc5RgSnorm,
        R::Bc7RgbaUnorm => W::Bc7RgbaUnorm,
        R::Bc7RgbaUnormSrgb => W::Bc7RgbaUnormSrgb,
    }
}

pub fn vertex_format(fmt: rhi::VertexFormat) -> wgpu::VertexFormat {
    use rhi::VertexFormat as R;
    use wgpu::VertexFormat as W;
    match fmt {
        R::Uint8x2 => W::Uint8x2,
        R::Uint8x4 => W::Uint8x4,
        R::Sint8x2 => W::Sint8x2,
        R::Sint8x4 => W::Sint8x4,
        R::Unorm8x2 => W::Unorm8x2,
        R::Unorm8x4 => W::Unorm8x4,
        R::Snorm8x2 => W::Snorm8x2,
        R::Snorm8x4 => W::Snorm8x4,
        R::Uint16x2 => W::Uint16x2,
        R::Uint16x4 => W::Uint16x4,
        R::Sint16x2 => W::Sint16x2,
        R::Sint16x4 => W::Sint16x4,
        R::Unorm16x2 => W::Unorm16x2,
        R::Unorm16x4 => W::Unorm16x4,
        R::Snorm16x2 => W::Snorm16x2,
        R::Snorm16x4 => W::Snorm16x4,
        R::Float16x2 => W::Float16x2,
        R::Float16x4 => W::Float16x4,
        R::Float32 => W::Float32,
        R::Float32x2 => W::Float32x2,
        R::Float32x3 => W::Float32x3,
        R::Float32x4 => W::Float32x4,
        R::Uint32 => W::Uint32,
        R::Uint32x2 => W::Uint32x2,
        R::Uint32x3 => W::Uint32x3,
        R::Uint32x4 => W::Uint32x4,
        R::Sint32 => W::Sint32,
        R::Sint32x2 => W::Sint32x2,
        R::Sint32x3 => W::Sint32x3,
        R::Sint32x4 => W::Sint32x4,
    }
}

pub fn index_format(fmt: rhi::IndexFormat) -> wgpu::IndexFormat {
    match fmt {
        rhi::IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        rhi::IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

// ────────── Usage bitflags ──────────

pub fn buffer_usages(
    usage: rhi::BufferUsage,
    memory: rhi::MemoryLocation,
) -> wgpu::BufferUsages {
    let mut u = wgpu::BufferUsages::empty();
    if usage.contains(rhi::BufferUsage::VERTEX) {
        u |= wgpu::BufferUsages::VERTEX;
    }
    if usage.contains(rhi::BufferUsage::INDEX) {
        u |= wgpu::BufferUsages::INDEX;
    }
    if usage.contains(rhi::BufferUsage::UNIFORM) {
        u |= wgpu::BufferUsages::UNIFORM;
    }
    if usage.contains(rhi::BufferUsage::STORAGE) {
        u |= wgpu::BufferUsages::STORAGE;
    }
    if usage.contains(rhi::BufferUsage::INDIRECT) {
        u |= wgpu::BufferUsages::INDIRECT;
    }
    if usage.contains(rhi::BufferUsage::COPY_SRC) {
        u |= wgpu::BufferUsages::COPY_SRC;
    }
    if usage.contains(rhi::BufferUsage::COPY_DST) {
        u |= wgpu::BufferUsages::COPY_DST;
    }
    if usage.contains(rhi::BufferUsage::MAP_READ) {
        u |= wgpu::BufferUsages::MAP_READ;
    }
    if usage.contains(rhi::BufferUsage::MAP_WRITE) {
        u |= wgpu::BufferUsages::MAP_WRITE;
    }
    // `MemoryLocation::Upload` and `Readback` imply the respective map flags.
    match memory {
        rhi::MemoryLocation::Device => {}
        rhi::MemoryLocation::Upload => {
            u |= wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
        }
        rhi::MemoryLocation::Readback => {
            u |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        }
    }
    u
}

pub fn texture_usages(usage: rhi::TextureUsage) -> wgpu::TextureUsages {
    let mut u = wgpu::TextureUsages::empty();
    if usage.contains(rhi::TextureUsage::COPY_SRC) {
        u |= wgpu::TextureUsages::COPY_SRC;
    }
    if usage.contains(rhi::TextureUsage::COPY_DST) {
        u |= wgpu::TextureUsages::COPY_DST;
    }
    if usage.contains(rhi::TextureUsage::TEXTURE_BINDING) {
        u |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(rhi::TextureUsage::STORAGE_BINDING) {
        u |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(rhi::TextureUsage::RENDER_ATTACHMENT) {
        u |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    u
}

pub fn shader_stages(stages: rhi::ShaderStageFlags) -> wgpu::ShaderStages {
    let mut s = wgpu::ShaderStages::empty();
    if stages.contains(rhi::ShaderStageFlags::VERTEX) {
        s |= wgpu::ShaderStages::VERTEX;
    }
    if stages.contains(rhi::ShaderStageFlags::FRAGMENT) {
        s |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.contains(rhi::ShaderStageFlags::COMPUTE) {
        s |= wgpu::ShaderStages::COMPUTE;
    }
    // MESH/TASK map on platforms with EXPERIMENTAL_MESH_SHADER; wgpu 29 does
    // not expose them yet. Silently drop — the pipeline create will reject.
    s
}

pub fn color_writes(cw: rhi::ColorWrites) -> wgpu::ColorWrites {
    let mut w = wgpu::ColorWrites::empty();
    if cw.contains(rhi::ColorWrites::RED) {
        w |= wgpu::ColorWrites::RED;
    }
    if cw.contains(rhi::ColorWrites::GREEN) {
        w |= wgpu::ColorWrites::GREEN;
    }
    if cw.contains(rhi::ColorWrites::BLUE) {
        w |= wgpu::ColorWrites::BLUE;
    }
    if cw.contains(rhi::ColorWrites::ALPHA) {
        w |= wgpu::ColorWrites::ALPHA;
    }
    w
}

// ────────── Enums ──────────

pub fn address_mode(a: rhi::AddressMode) -> wgpu::AddressMode {
    match a {
        rhi::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        rhi::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        rhi::AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        rhi::AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

pub fn filter_mode(f: rhi::FilterMode) -> wgpu::FilterMode {
    match f {
        rhi::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        rhi::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

pub fn mipmap_filter(f: rhi::FilterMode) -> wgpu::FilterMode {
    match f {
        rhi::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        rhi::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

pub fn compare_function(c: rhi::CompareFunction) -> wgpu::CompareFunction {
    match c {
        rhi::CompareFunction::Never => wgpu::CompareFunction::Never,
        rhi::CompareFunction::Less => wgpu::CompareFunction::Less,
        rhi::CompareFunction::Equal => wgpu::CompareFunction::Equal,
        rhi::CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        rhi::CompareFunction::Greater => wgpu::CompareFunction::Greater,
        rhi::CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        rhi::CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        rhi::CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

pub fn border_color(b: rhi::BorderColor) -> wgpu::SamplerBorderColor {
    match b {
        rhi::BorderColor::TransparentBlack => wgpu::SamplerBorderColor::TransparentBlack,
        rhi::BorderColor::OpaqueBlack => wgpu::SamplerBorderColor::OpaqueBlack,
        rhi::BorderColor::OpaqueWhite => wgpu::SamplerBorderColor::OpaqueWhite,
    }
}

pub fn primitive_topology(t: rhi::PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match t {
        rhi::PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        rhi::PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        rhi::PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        rhi::PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        rhi::PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

pub fn front_face(f: rhi::FrontFace) -> wgpu::FrontFace {
    match f {
        rhi::FrontFace::Ccw => wgpu::FrontFace::Ccw,
        rhi::FrontFace::Cw => wgpu::FrontFace::Cw,
    }
}

pub fn cull_face(f: rhi::Face) -> wgpu::Face {
    match f {
        rhi::Face::Front => wgpu::Face::Front,
        rhi::Face::Back => wgpu::Face::Back,
    }
}

pub fn polygon_mode(p: rhi::PolygonMode) -> wgpu::PolygonMode {
    match p {
        rhi::PolygonMode::Fill => wgpu::PolygonMode::Fill,
        rhi::PolygonMode::Line => wgpu::PolygonMode::Line,
        rhi::PolygonMode::Point => wgpu::PolygonMode::Point,
    }
}

pub fn step_mode(m: rhi::VertexStepMode) -> wgpu::VertexStepMode {
    match m {
        rhi::VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        rhi::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

pub fn blend_factor(b: rhi::BlendFactor) -> wgpu::BlendFactor {
    use rhi::BlendFactor as R;
    use wgpu::BlendFactor as W;
    match b {
        R::Zero => W::Zero,
        R::One => W::One,
        R::Src => W::Src,
        R::OneMinusSrc => W::OneMinusSrc,
        R::SrcAlpha => W::SrcAlpha,
        R::OneMinusSrcAlpha => W::OneMinusSrcAlpha,
        R::Dst => W::Dst,
        R::OneMinusDst => W::OneMinusDst,
        R::DstAlpha => W::DstAlpha,
        R::OneMinusDstAlpha => W::OneMinusDstAlpha,
        R::SrcAlphaSaturated => W::SrcAlphaSaturated,
        R::Constant => W::Constant,
        R::OneMinusConstant => W::OneMinusConstant,
    }
}

pub fn blend_operation(o: rhi::BlendOperation) -> wgpu::BlendOperation {
    match o {
        rhi::BlendOperation::Add => wgpu::BlendOperation::Add,
        rhi::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        rhi::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        rhi::BlendOperation::Min => wgpu::BlendOperation::Min,
        rhi::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

pub fn stencil_op(o: rhi::StencilOp) -> wgpu::StencilOperation {
    match o {
        rhi::StencilOp::Keep => wgpu::StencilOperation::Keep,
        rhi::StencilOp::Zero => wgpu::StencilOperation::Zero,
        rhi::StencilOp::Replace => wgpu::StencilOperation::Replace,
        rhi::StencilOp::Invert => wgpu::StencilOperation::Invert,
        rhi::StencilOp::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        rhi::StencilOp::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        rhi::StencilOp::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        rhi::StencilOp::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
    }
}

#[allow(dead_code)] // wired up once WgpuRhi grows swapchain support
pub fn present_mode(p: rhi::PresentMode) -> wgpu::PresentMode {
    match p {
        rhi::PresentMode::Immediate => wgpu::PresentMode::Immediate,
        rhi::PresentMode::Fifo => wgpu::PresentMode::Fifo,
        rhi::PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        rhi::PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
    }
}

pub fn view_dimension(d: rhi::TextureViewDimension) -> wgpu::TextureViewDimension {
    match d {
        rhi::TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
        rhi::TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        rhi::TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        rhi::TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        rhi::TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        rhi::TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
    }
}

pub fn texture_sample_type(s: rhi::TextureSampleType) -> wgpu::TextureSampleType {
    match s {
        rhi::TextureSampleType::Float { filterable } => wgpu::TextureSampleType::Float { filterable },
        rhi::TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
        rhi::TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
        rhi::TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
    }
}

pub fn storage_access(a: rhi::StorageTextureAccess) -> wgpu::StorageTextureAccess {
    match a {
        rhi::StorageTextureAccess::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
        rhi::StorageTextureAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
        rhi::StorageTextureAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
    }
}

pub fn sampler_binding_type(s: rhi::SamplerBindingType) -> wgpu::SamplerBindingType {
    match s {
        rhi::SamplerBindingType::Filtering => wgpu::SamplerBindingType::Filtering,
        rhi::SamplerBindingType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
        rhi::SamplerBindingType::Comparison => wgpu::SamplerBindingType::Comparison,
    }
}

pub fn binding_type(b: rhi::BindingType) -> wgpu::BindingType {
    match b {
        rhi::BindingType::UniformBuffer {
            has_dynamic_offset,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: min_binding_size.and_then(std::num::NonZeroU64::new),
        },
        rhi::BindingType::StorageBuffer {
            read_only,
            has_dynamic_offset,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset,
            min_binding_size: min_binding_size.and_then(std::num::NonZeroU64::new),
        },
        rhi::BindingType::Sampler(s) => wgpu::BindingType::Sampler(sampler_binding_type(s)),
        rhi::BindingType::Texture {
            sample_type,
            view_dimension: view,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: texture_sample_type(sample_type),
            view_dimension: view_dimension(view),
            multisampled,
        },
        rhi::BindingType::StorageTexture {
            access,
            format,
            view_dimension: view,
        } => wgpu::BindingType::StorageTexture {
            access: storage_access(access),
            format: texture_format(format),
            view_dimension: view_dimension(view),
        },
    }
}

pub fn color_load_op(op: rhi::LoadOp<[f32; 4]>) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        rhi::LoadOp::Load => wgpu::LoadOp::Load,
        rhi::LoadOp::Clear([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color {
            r: r.into(),
            g: g.into(),
            b: b.into(),
            a: a.into(),
        }),
    }
}

pub fn depth_load_op(op: rhi::LoadOp<f32>) -> wgpu::LoadOp<f32> {
    match op {
        rhi::LoadOp::Load => wgpu::LoadOp::Load,
        rhi::LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
    }
}

pub fn stencil_load_op(op: rhi::LoadOp<u32>) -> wgpu::LoadOp<u32> {
    match op {
        rhi::LoadOp::Load => wgpu::LoadOp::Load,
        rhi::LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
    }
}

pub fn store_op(op: rhi::StoreOp) -> wgpu::StoreOp {
    match op {
        rhi::StoreOp::Store => wgpu::StoreOp::Store,
        rhi::StoreOp::Discard => wgpu::StoreOp::Discard,
    }
}

pub fn texture_dimension(dim: rhi::TextureDimension) -> (wgpu::TextureDimension, wgpu::Extent3d) {
    match dim {
        rhi::TextureDimension::D1 { width } => (
            wgpu::TextureDimension::D1,
            wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
        ),
        rhi::TextureDimension::D2 {
            width,
            height,
            array_layers,
        } => (
            wgpu::TextureDimension::D2,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
        ),
        rhi::TextureDimension::D3 {
            width,
            height,
            depth,
        } => (
            wgpu::TextureDimension::D3,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
        ),
    }
}
