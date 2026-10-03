//! Command recording.
//!
//! A [`CommandList`] is a backend-agnostic list of rendering operations. User
//! code records commands by calling methods on the list (or on a scoped
//! `RenderPass` / `ComputePass`), then submits the list via [`crate::Rhi::submit`].
//! Backends translate [`Command`] entries into their native command buffers
//! at submit time.
//!
//! Recording is pure data — no backend calls happen until submit. This lets
//! us auto-insert resource state transitions inside each backend (the RHI's
//! promise to engine code is that it never has to write barriers).

use std::ops::Range;

use crate::format::IndexFormat;
use crate::handle::{BindGroupHandle, BufferHandle, PipelineHandle, TextureHandle};
use crate::shader::ShaderStageFlags;

// ────────────────────────── Public surface ──────────────────────────

#[derive(Default)]
pub struct CommandList {
    commands: Vec<Command>,
}

impl CommandList {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) -> RenderPass<'_> {
        self.commands.push(Command::BeginRenderPass(RenderPassInfo::from_desc(desc)));
        RenderPass { cl: self }
    }

    pub fn begin_compute_pass(&mut self, label: Option<&str>) -> ComputePass<'_> {
        self.commands.push(Command::BeginComputePass {
            label: label.map(str::to_owned),
        });
        ComputePass { cl: self }
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) {
        self.commands.push(Command::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        src_bytes_per_row: u32,
        src_rows_per_image: u32,
        dst: TextureHandle,
        dst_region: TextureCopyRegion,
    ) {
        self.commands.push(Command::CopyBufferToTexture {
            src,
            src_offset,
            src_bytes_per_row,
            src_rows_per_image,
            dst,
            dst_region,
        });
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        src_region: TextureCopyRegion,
        dst: BufferHandle,
        dst_offset: u64,
        dst_bytes_per_row: u32,
        dst_rows_per_image: u32,
    ) {
        self.commands.push(Command::CopyTextureToBuffer {
            src,
            src_region,
            dst,
            dst_offset,
            dst_bytes_per_row,
            dst_rows_per_image,
        });
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        src_region: TextureCopyRegion,
        dst: TextureHandle,
        dst_region: TextureCopyRegion,
    ) {
        self.commands.push(Command::CopyTextureToTexture {
            src,
            src_region,
            dst,
            dst_region,
        });
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.commands.push(Command::PushDebugGroup(label.to_owned()));
    }

    pub fn pop_debug_group(&mut self) {
        self.commands.push(Command::PopDebugGroup);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.commands.push(Command::InsertDebugMarker(label.to_owned()));
    }

    #[must_use]
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    #[must_use]
    pub fn into_commands(self) -> Vec<Command> {
        self.commands
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ────────────────────────── Render pass ──────────────────────────

pub struct RenderPass<'a> {
    cl: &'a mut CommandList,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.cl.commands.push(Command::EndRenderPass);
    }
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        self.cl.commands.push(Command::SetPipeline(pipeline));
    }

    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: BufferHandle,
        offset: u64,
        size: Option<u64>,
    ) {
        self.cl.commands.push(Command::SetVertexBuffer {
            slot,
            buffer,
            offset,
            size,
        });
    }

    pub fn set_index_buffer(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        size: Option<u64>,
        format: IndexFormat,
    ) {
        self.cl.commands.push(Command::SetIndexBuffer {
            buffer,
            offset,
            size,
            format,
        });
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        group: BindGroupHandle,
        dynamic_offsets: &[u32],
    ) {
        self.cl.commands.push(Command::SetBindGroup {
            index,
            group,
            dynamic_offsets: dynamic_offsets.to_vec(),
        });
    }

    pub fn set_push_constants(&mut self, stages: ShaderStageFlags, offset: u32, data: &[u8]) {
        self.cl.commands.push(Command::SetPushConstants {
            stages,
            offset,
            data: data.to_vec(),
        });
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.cl.commands.push(Command::SetViewport(viewport));
    }

    pub fn set_scissor(&mut self, rect: Rect) {
        self.cl.commands.push(Command::SetScissor(rect));
    }

    pub fn set_blend_constant(&mut self, color: [f32; 4]) {
        self.cl.commands.push(Command::SetBlendConstant(color));
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.cl.commands.push(Command::Draw {
            vertices,
            instances,
        });
    }

    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        self.cl.commands.push(Command::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
    }

    pub fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.cl.commands.push(Command::DrawIndirect { buffer, offset });
    }

    pub fn draw_indexed_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.cl.commands.push(Command::DrawIndexedIndirect { buffer, offset });
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.cl.commands.push(Command::PushDebugGroup(label.to_owned()));
    }

    pub fn pop_debug_group(&mut self) {
        self.cl.commands.push(Command::PopDebugGroup);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.cl.commands.push(Command::InsertDebugMarker(label.to_owned()));
    }
}

// ────────────────────────── Compute pass ──────────────────────────

pub struct ComputePass<'a> {
    cl: &'a mut CommandList,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.cl.commands.push(Command::EndComputePass);
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: PipelineHandle) {
        self.cl.commands.push(Command::SetPipeline(pipeline));
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        group: BindGroupHandle,
        dynamic_offsets: &[u32],
    ) {
        self.cl.commands.push(Command::SetBindGroup {
            index,
            group,
            dynamic_offsets: dynamic_offsets.to_vec(),
        });
    }

    pub fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
        self.cl.commands.push(Command::SetPushConstants {
            stages: ShaderStageFlags::COMPUTE,
            offset,
            data: data.to_vec(),
        });
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.cl.commands.push(Command::Dispatch { x, y, z });
    }

    pub fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.cl.commands.push(Command::DispatchIndirect { buffer, offset });
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.cl.commands.push(Command::PushDebugGroup(label.to_owned()));
    }

    pub fn pop_debug_group(&mut self) {
        self.cl.commands.push(Command::PopDebugGroup);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.cl.commands.push(Command::InsertDebugMarker(label.to_owned()));
    }
}

// ────────────────────────── Descs ──────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct RenderPassDesc<'a> {
    pub label: Option<&'a str>,
    pub color_attachments: &'a [Option<ColorAttachment>],
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

#[derive(Debug, Clone, Copy)]
pub struct ColorAttachment {
    pub view: TextureHandle,
    pub resolve_target: Option<TextureHandle>,
    pub load: LoadOp<[f32; 4]>,
    pub store: StoreOp,
}

#[derive(Debug, Clone, Copy)]
pub struct DepthStencilAttachment {
    pub view: TextureHandle,
    pub depth_load: LoadOp<f32>,
    pub depth_store: StoreOp,
    pub stencil_load: LoadOp<u32>,
    pub stencil_store: StoreOp,
    pub read_only_depth: bool,
    pub read_only_stencil: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<T> {
    Clear(T),
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureCopyRegion {
    pub mip_level: u32,
    pub array_layer: u32,
    pub origin: [u32; 3],
    pub extent: [u32; 3],
}

// ────────────────────────── Owned command form ──────────────────────────

/// Owned, cloneable rendering of a [`RenderPassDesc`] for storage in [`Command`].
#[derive(Debug, Clone)]
pub struct RenderPassInfo {
    pub label: Option<String>,
    pub color_attachments: Vec<Option<ColorAttachment>>,
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

impl RenderPassInfo {
    #[must_use]
    pub fn from_desc(desc: &RenderPassDesc<'_>) -> Self {
        Self {
            label: desc.label.map(str::to_owned),
            color_attachments: desc.color_attachments.to_vec(),
            depth_stencil_attachment: desc.depth_stencil_attachment,
        }
    }
}

/// All recorded operations. Backends iterate and translate to native APIs.
#[derive(Debug, Clone)]
pub enum Command {
    BeginRenderPass(RenderPassInfo),
    EndRenderPass,
    BeginComputePass {
        label: Option<String>,
    },
    EndComputePass,

    SetPipeline(PipelineHandle),
    SetVertexBuffer {
        slot: u32,
        buffer: BufferHandle,
        offset: u64,
        size: Option<u64>,
    },
    SetIndexBuffer {
        buffer: BufferHandle,
        offset: u64,
        size: Option<u64>,
        format: IndexFormat,
    },
    SetBindGroup {
        index: u32,
        group: BindGroupHandle,
        dynamic_offsets: Vec<u32>,
    },
    SetPushConstants {
        stages: ShaderStageFlags,
        offset: u32,
        data: Vec<u8>,
    },
    SetViewport(Viewport),
    SetScissor(Rect),
    SetBlendConstant([f32; 4]),

    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
    DrawIndirect {
        buffer: BufferHandle,
        offset: u64,
    },
    DrawIndexedIndirect {
        buffer: BufferHandle,
        offset: u64,
    },

    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    DispatchIndirect {
        buffer: BufferHandle,
        offset: u64,
    },

    CopyBufferToBuffer {
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    },
    CopyBufferToTexture {
        src: BufferHandle,
        src_offset: u64,
        src_bytes_per_row: u32,
        src_rows_per_image: u32,
        dst: TextureHandle,
        dst_region: TextureCopyRegion,
    },
    CopyTextureToBuffer {
        src: TextureHandle,
        src_region: TextureCopyRegion,
        dst: BufferHandle,
        dst_offset: u64,
        dst_bytes_per_row: u32,
        dst_rows_per_image: u32,
    },
    CopyTextureToTexture {
        src: TextureHandle,
        src_region: TextureCopyRegion,
        dst: TextureHandle,
        dst_region: TextureCopyRegion,
    },

    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
}
