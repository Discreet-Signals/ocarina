//! Translate an [`ocarina_rhi::CommandList`] into wgpu command encoder calls.
//!
//! Non-pass commands (buffer copies, debug groups) execute directly on the
//! encoder. Pass commands (`BeginRenderPass`, `BeginComputePass`) trigger
//! a scoped sub-interpreter that consumes all subsequent commands until the
//! matching `End*Pass`.

use std::num::NonZeroU32;

use ocarina_rhi::{Command, RhiError, RhiResult};

use crate::convert as cv;
use crate::rhi::{PipelineEntry, Pools};

pub fn interpret(
    encoder: &mut wgpu::CommandEncoder,
    pools: &Pools,
    commands: &[Command],
) -> RhiResult<()> {
    let mut i = 0;
    while i < commands.len() {
        match &commands[i] {
            Command::BeginRenderPass(info) => {
                i = run_render_pass(encoder, pools, i + 1, commands, info)?;
            }
            Command::BeginComputePass { label } => {
                i = run_compute_pass(encoder, pools, i + 1, commands, label.as_deref())?;
            }
            Command::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let s = pools
                    .buffers
                    .get(src.raw().slot, src.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (copy src)"))?;
                let d = pools
                    .buffers
                    .get(dst.raw().slot, dst.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (copy dst)"))?;
                encoder.copy_buffer_to_buffer(s, *src_offset, d, *dst_offset, *size);
            }
            Command::CopyBufferToTexture {
                src,
                src_offset,
                src_bytes_per_row,
                src_rows_per_image,
                dst,
                dst_region,
            } => {
                let s = pools
                    .buffers
                    .get(src.raw().slot, src.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (copy src)"))?;
                let d = pools
                    .textures
                    .get(dst.raw().slot, dst.raw().generation)
                    .ok_or(RhiError::InvalidHandle("TextureHandle (copy dst)"))?;
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: s,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: *src_offset,
                            bytes_per_row: Some(*src_bytes_per_row),
                            rows_per_image: Some(*src_rows_per_image),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &d.texture,
                        mip_level: dst_region.mip_level,
                        origin: wgpu::Origin3d {
                            x: dst_region.origin[0],
                            y: dst_region.origin[1],
                            z: dst_region.origin[2],
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: dst_region.extent[0],
                        height: dst_region.extent[1],
                        depth_or_array_layers: dst_region.extent[2],
                    },
                );
            }
            Command::CopyTextureToBuffer {
                src,
                src_region,
                dst,
                dst_offset,
                dst_bytes_per_row,
                dst_rows_per_image,
            } => {
                let s = pools
                    .textures
                    .get(src.raw().slot, src.raw().generation)
                    .ok_or(RhiError::InvalidHandle("TextureHandle (copy src)"))?;
                let d = pools
                    .buffers
                    .get(dst.raw().slot, dst.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (copy dst)"))?;
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &s.texture,
                        mip_level: src_region.mip_level,
                        origin: wgpu::Origin3d {
                            x: src_region.origin[0],
                            y: src_region.origin[1],
                            z: src_region.origin[2],
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: d,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: *dst_offset,
                            bytes_per_row: Some(*dst_bytes_per_row),
                            rows_per_image: Some(*dst_rows_per_image),
                        },
                    },
                    wgpu::Extent3d {
                        width: src_region.extent[0],
                        height: src_region.extent[1],
                        depth_or_array_layers: src_region.extent[2],
                    },
                );
            }
            Command::CopyTextureToTexture {
                src,
                src_region,
                dst,
                dst_region,
            } => {
                let s = pools
                    .textures
                    .get(src.raw().slot, src.raw().generation)
                    .ok_or(RhiError::InvalidHandle("TextureHandle (copy src)"))?;
                let d = pools
                    .textures
                    .get(dst.raw().slot, dst.raw().generation)
                    .ok_or(RhiError::InvalidHandle("TextureHandle (copy dst)"))?;
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &s.texture,
                        mip_level: src_region.mip_level,
                        origin: wgpu::Origin3d {
                            x: src_region.origin[0],
                            y: src_region.origin[1],
                            z: src_region.origin[2],
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &d.texture,
                        mip_level: dst_region.mip_level,
                        origin: wgpu::Origin3d {
                            x: dst_region.origin[0],
                            y: dst_region.origin[1],
                            z: dst_region.origin[2],
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: src_region.extent[0],
                        height: src_region.extent[1],
                        depth_or_array_layers: src_region.extent[2],
                    },
                );
            }
            Command::PushDebugGroup(label) => encoder.push_debug_group(label),
            Command::PopDebugGroup => encoder.pop_debug_group(),
            Command::InsertDebugMarker(label) => encoder.insert_debug_marker(label),

            // Commands that only make sense inside a pass:
            Command::EndRenderPass
            | Command::EndComputePass
            | Command::SetPipeline(_)
            | Command::SetVertexBuffer { .. }
            | Command::SetIndexBuffer { .. }
            | Command::SetBindGroup { .. }
            | Command::SetPushConstants { .. }
            | Command::SetViewport(_)
            | Command::SetScissor(_)
            | Command::SetBlendConstant(_)
            | Command::Draw { .. }
            | Command::DrawIndexed { .. }
            | Command::DrawIndirect { .. }
            | Command::DrawIndexedIndirect { .. }
            | Command::Dispatch { .. }
            | Command::DispatchIndirect { .. } => {
                return Err(RhiError::Backend(format!(
                    "command outside pass: {:?}",
                    &commands[i]
                )));
            }
        }
        i += 1;
    }
    Ok(())
}

fn run_render_pass(
    encoder: &mut wgpu::CommandEncoder,
    pools: &Pools,
    start: usize,
    commands: &[Command],
    info: &ocarina_rhi::RenderPassInfo,
) -> RhiResult<usize> {
    // Resolve color attachments.
    let color_atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = info
        .color_attachments
        .iter()
        .map(|a| -> RhiResult<_> {
            Ok(match a {
                None => None,
                Some(att) => {
                    let view = pools
                        .textures
                        .get(att.view.raw().slot, att.view.raw().generation)
                        .ok_or(RhiError::InvalidHandle("TextureHandle (color attachment)"))?;
                    let resolve_target = match att.resolve_target {
                        None => None,
                        Some(r) => Some(
                            &pools
                                .textures
                                .get(r.raw().slot, r.raw().generation)
                                .ok_or(RhiError::InvalidHandle(
                                    "TextureHandle (resolve target)",
                                ))?
                                .view,
                        ),
                    };
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view.view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: cv::color_load_op(att.load),
                            store: cv::store_op(att.store),
                        },
                    })
                }
            })
        })
        .collect::<Result<_, _>>()?;

    let depth_att = info
        .depth_stencil_attachment
        .as_ref()
        .map(|d| -> RhiResult<_> {
            let view = pools
                .textures
                .get(d.view.raw().slot, d.view.raw().generation)
                .ok_or(RhiError::InvalidHandle("TextureHandle (depth attachment)"))?;
            Ok(wgpu::RenderPassDepthStencilAttachment {
                view: &view.view,
                depth_ops: Some(wgpu::Operations {
                    load: cv::depth_load_op(d.depth_load),
                    store: cv::store_op(d.depth_store),
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: cv::stencil_load_op(d.stencil_load),
                    store: cv::store_op(d.stencil_store),
                }),
            })
        })
        .transpose()?;

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: info.label.as_deref(),
        color_attachments: &color_atts,
        depth_stencil_attachment: depth_att,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    let mut i = start;
    while i < commands.len() {
        match &commands[i] {
            Command::EndRenderPass => {
                return Ok(i);
            }
            Command::SetPipeline(p) => {
                let entry = pools
                    .pipelines
                    .get(p.raw().slot, p.raw().generation)
                    .ok_or(RhiError::InvalidHandle("PipelineHandle"))?;
                match entry {
                    PipelineEntry::Graphics(pipe) => pass.set_pipeline(pipe),
                    PipelineEntry::Compute(_) => {
                        return Err(RhiError::Backend(
                            "SetPipeline with compute pipeline in render pass".into(),
                        ));
                    }
                }
            }
            Command::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                let buf = pools
                    .buffers
                    .get(buffer.raw().slot, buffer.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (vertex)"))?;
                let slice = match size {
                    Some(s) => buf.slice(*offset..offset + s),
                    None => buf.slice(*offset..),
                };
                pass.set_vertex_buffer(*slot, slice);
            }
            Command::SetIndexBuffer {
                buffer,
                offset,
                size,
                format,
            } => {
                let buf = pools
                    .buffers
                    .get(buffer.raw().slot, buffer.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (index)"))?;
                let slice = match size {
                    Some(s) => buf.slice(*offset..offset + s),
                    None => buf.slice(*offset..),
                };
                pass.set_index_buffer(slice, cv::index_format(*format));
            }
            Command::SetBindGroup {
                index,
                group,
                dynamic_offsets,
            } => {
                let bg = pools
                    .bgs
                    .get(group.raw().slot, group.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BindGroupHandle"))?;
                pass.set_bind_group(*index, bg, dynamic_offsets);
            }
            Command::SetPushConstants {
                stages,
                offset,
                data,
            } => {
                pass.set_push_constants(cv::shader_stages(*stages), *offset, data);
            }
            Command::SetViewport(v) => {
                pass.set_viewport(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
            }
            Command::SetScissor(r) => {
                pass.set_scissor_rect(r.x, r.y, r.width, r.height);
            }
            Command::SetBlendConstant([r, g, b, a]) => {
                pass.set_blend_constant(wgpu::Color {
                    r: (*r).into(),
                    g: (*g).into(),
                    b: (*b).into(),
                    a: (*a).into(),
                });
            }
            Command::Draw {
                vertices,
                instances,
            } => pass.draw(vertices.clone(), instances.clone()),
            Command::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => pass.draw_indexed(indices.clone(), *base_vertex, instances.clone()),
            Command::DrawIndirect { buffer, offset } => {
                let buf = pools
                    .buffers
                    .get(buffer.raw().slot, buffer.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (indirect)"))?;
                pass.draw_indirect(buf, *offset);
            }
            Command::DrawIndexedIndirect { buffer, offset } => {
                let buf = pools
                    .buffers
                    .get(buffer.raw().slot, buffer.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (indirect)"))?;
                pass.draw_indexed_indirect(buf, *offset);
            }
            Command::PushDebugGroup(label) => pass.push_debug_group(label),
            Command::PopDebugGroup => pass.pop_debug_group(),
            Command::InsertDebugMarker(label) => pass.insert_debug_marker(label),

            Command::BeginRenderPass(_)
            | Command::BeginComputePass { .. }
            | Command::EndComputePass
            | Command::Dispatch { .. }
            | Command::DispatchIndirect { .. }
            | Command::CopyBufferToBuffer { .. }
            | Command::CopyBufferToTexture { .. }
            | Command::CopyTextureToBuffer { .. }
            | Command::CopyTextureToTexture { .. } => {
                return Err(RhiError::Backend(format!(
                    "command not valid inside render pass: {:?}",
                    &commands[i]
                )));
            }
        }
        i += 1;
    }
    // Reached end without seeing EndRenderPass — the recorder didn't drop the
    // `RenderPass` scope. Treat as end; wgpu ends it when dropped anyway.
    let _ = NonZeroU32::new(1); // keep the import used
    Ok(i)
}

fn run_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    pools: &Pools,
    start: usize,
    commands: &[Command],
    label: Option<&str>,
) -> RhiResult<usize> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label,
        timestamp_writes: None,
    });

    let mut i = start;
    while i < commands.len() {
        match &commands[i] {
            Command::EndComputePass => return Ok(i),
            Command::SetPipeline(p) => {
                let entry = pools
                    .pipelines
                    .get(p.raw().slot, p.raw().generation)
                    .ok_or(RhiError::InvalidHandle("PipelineHandle"))?;
                match entry {
                    PipelineEntry::Compute(pipe) => pass.set_pipeline(pipe),
                    PipelineEntry::Graphics(_) => {
                        return Err(RhiError::Backend(
                            "SetPipeline with graphics pipeline in compute pass".into(),
                        ));
                    }
                }
            }
            Command::SetBindGroup {
                index,
                group,
                dynamic_offsets,
            } => {
                let bg = pools
                    .bgs
                    .get(group.raw().slot, group.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BindGroupHandle"))?;
                pass.set_bind_group(*index, bg, dynamic_offsets);
            }
            Command::SetPushConstants { offset, data, .. } => {
                pass.set_push_constants(*offset, data);
            }
            Command::Dispatch { x, y, z } => pass.dispatch_workgroups(*x, *y, *z),
            Command::DispatchIndirect { buffer, offset } => {
                let buf = pools
                    .buffers
                    .get(buffer.raw().slot, buffer.raw().generation)
                    .ok_or(RhiError::InvalidHandle("BufferHandle (indirect)"))?;
                pass.dispatch_workgroups_indirect(buf, *offset);
            }
            Command::PushDebugGroup(label) => pass.push_debug_group(label),
            Command::PopDebugGroup => pass.pop_debug_group(),
            Command::InsertDebugMarker(label) => pass.insert_debug_marker(label),

            _other => {
                return Err(RhiError::Backend(format!(
                    "command not valid inside compute pass: {:?}",
                    &commands[i]
                )));
            }
        }
        i += 1;
    }
    Ok(i)
}
