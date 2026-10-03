//! Graphics and compute pipeline descriptors.

use crate::error::RhiError;
use crate::format::{TextureFormat, VertexFormat};
use crate::handle::{BindGroupLayoutHandle, PipelineHandle, ShaderHandle};
use crate::shader::ShaderStageFlags;

// ────────────────────────── Async pipeline build ──────────────────────────

/// Opaque identifier for an in-flight async pipeline build. Returned
/// by [`Rhi::create_pipeline_async`] and matched by the caller against
/// the entries from [`Rhi::poll_async_pipelines`] to know when (and
/// whether) a particular submission completed.
///
/// The id is unique per RHI for the lifetime of the RHI (no reuse —
/// the backend holds a monotonic counter). Comparing ids by equality
/// is the intended use.
///
/// [`Rhi::create_pipeline_async`]: crate::Rhi::create_pipeline_async
/// [`Rhi::poll_async_pipelines`]: crate::Rhi::poll_async_pipelines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncPipelineId(pub u64);

/// Result of one async pipeline compile, drained by
/// [`Rhi::poll_async_pipelines`]. The successful payload is a
/// [`PipelineHandle`] adopted into the RHI's regular pipeline pool —
/// callers use it identically to a handle from the synchronous
/// [`Rhi::create_pipeline`] path, including `destroy_pipeline`.
///
/// [`Rhi::create_pipeline`]: crate::Rhi::create_pipeline
/// [`Rhi::poll_async_pipelines`]: crate::Rhi::poll_async_pipelines
#[derive(Debug)]
pub struct AsyncPipelineCompletion {
    pub id: AsyncPipelineId,
    pub result: Result<PipelineHandle, RhiError>,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineDesc<'a> {
    pub label: Option<&'a str>,
    pub kind: PipelineKind<'a>,
    /// Bind group layouts the pipeline uses, indexed by bind group slot.
    pub bind_group_layouts: &'a [BindGroupLayoutHandle],
    pub push_constant_ranges: &'a [PushConstantRange],
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineKind<'a> {
    Graphics(GraphicsPipelineDesc<'a>),
    Compute(ComputePipelineDesc),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushConstantRange {
    pub stages: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

// ────────────────────────── Graphics ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex: VertexStage<'a>,
    pub fragment: Option<FragmentStage<'a>>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
}

#[derive(Debug, Clone, Copy)]
pub struct VertexStage<'a> {
    pub shader: ShaderHandle,
    pub buffers: &'a [VertexBufferLayout<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct FragmentStage<'a> {
    pub shader: ShaderHandle,
    pub targets: &'a [Option<ColorTargetState>],
}

#[derive(Debug, Clone, Copy)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<crate::format::IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: Option<Face>,
    pub polygon_mode: PolygonMode,
    pub unclipped_depth: bool,
    pub conservative: bool,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontFace {
    Ccw,
    Cw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: crate::resource::CompareFunction,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub bias: DepthBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StencilFaceState {
    pub compare: Option<crate::resource::CompareFunction>,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DepthBias {
    pub constant: i32,
    pub slope_scale: f32,
    pub clamp: f32,
}

impl Eq for DepthBias {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent::ALPHA_BLENDING,
        alpha: BlendComponent::OVER,
    };
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };
    pub const ALPHA_BLENDING: Self = Self {
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ColorWrites: u32 {
        const RED   = 1 << 0;
        const GREEN = 1 << 1;
        const BLUE  = 1 << 2;
        const ALPHA = 1 << 3;
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        const ALL   = Self::COLOR.bits() | Self::ALPHA.bits();
    }
}

// ────────────────────────── Compute ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct ComputePipelineDesc {
    pub shader: ShaderHandle,
}
