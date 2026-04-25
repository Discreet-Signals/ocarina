//! Viewport: renders the ECS world through the RHI into an offscreen
//! texture and lets the user pick + manipulate entities directly.
//!
//! Phase 4 added:
//!  - Entities with `Transform` are drawn as tinted unit cubes.
//!  - Camera pans on middle-mouse / Shift+drag in addition to orbit
//!    (left drag) and zoom (scroll).
//!  - Click an entity → it becomes selected and highlights.
//!  - Three colored gizmo bars at the selection translate along an
//!    axis when dragged. On drag-end the new transform is spliced
//!    back into the entity's `.toml` file (formatting preserved).
//!  - A `.hlsl` file selected in the file browser is recompiled and
//!    swapped in as the active material so authors can preview shader
//!    edits live.

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use egui::epaint::mutex::RwLock;
use glam::{Mat4, Quat, Vec3, Vec4};
use std::path::PathBuf;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use ocarina_core::components::Transform as EcsTransform;
use ocarina_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupHandle, BindGroupLayoutDesc, BindGroupLayoutEntry,
    BindGroupLayoutHandle, BindingResource, BindingType, BufferDesc, BufferHandle, BufferUsage,
    ColorAttachment, ColorTargetState, ColorWrites, CommandList, CompareFunction,
    DepthStencilAttachment, DepthStencilState, FragmentStage, GraphicsPipelineDesc, IndexFormat,
    LoadOp, MemoryLocation, MultisampleState, PipelineDesc, PipelineHandle, PipelineKind,
    PrimitiveState, RenderPassDesc, Rhi, RhiResult, ShaderBytecode, ShaderDesc, ShaderHandle,
    ShaderStage, ShaderStageFlags, StoreOp, TextureDesc, TextureDimension, TextureFormat,
    TextureHandle, TextureUsage, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStage,
    VertexStepMode,
};
use ocarina_rhi_wgpu::WgpuRhi;
use ocarina_scene::Scene;

use crate::camera::OrbitCamera;
use crate::gizmo::{closest_point_on_axis, ray_aabb, Axis};
use crate::mesh::{self, Mesh, Vertex};
use crate::selection::Selection;

const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

const LIT_WGSL: &str = include_str!("../shaders/lit.wgsl");
const LIT_HLSL: &str = include_str!("../shaders/lit.hlsl");

/// Std140-style camera uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
    light_dir: Vec4,
    base_color: Vec4,
}
const CAMERA_UNIFORM_SIZE: u64 = std::mem::size_of::<CameraUniform>() as u64;

/// Per-instance uniform: model matrix + tint. 80 bytes; padded to
/// 256 (the wgpu min_uniform_buffer_offset_alignment baseline) so a
/// single buffer can stride through many entities with dynamic offsets.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceUniform {
    model: Mat4,
    tint: Vec4,
}
const INSTANCE_UNIFORM_SIZE: u64 = std::mem::size_of::<InstanceUniform>() as u64;
const INSTANCE_STRIDE: u64 = 256;

/// Cap on how many entities + gizmo handles we'll render in a frame.
/// Plenty for the scenes we expect; keeps the per-instance buffer
/// statically sized so we don't reallocate per frame.
const MAX_INSTANCES: u32 = 1024;

pub struct Viewport {
    // ── Cube + Triforce mesh ────────────────────────────────
    cube_vb: BufferHandle,
    cube_ib: BufferHandle,
    cube_index_count: u32,

    // ── Bindings ───────────────────────────────────────────
    camera_buffer: BufferHandle,
    camera_bgl: BindGroupLayoutHandle,
    camera_bg: BindGroupHandle,

    instance_buffer: BufferHandle,
    instance_bgl: BindGroupLayoutHandle,
    instance_bg: BindGroupHandle,

    // ── Pipelines ──────────────────────────────────────────
    pipeline: PipelineHandle,
    /// Last shader path successfully swapped in via shader preview.
    /// `None` while running on the built-in lit shader.
    preview_shader: Option<PathBuf>,
    /// File modification time of the previewed shader, for poll-based
    /// hot-reload.
    preview_shader_mtime: Option<std::time::SystemTime>,
    /// Most recent shader-compile error, surfaced as an overlay.
    preview_error: Option<String>,

    // ── Per-size state ────────────────────────────────────
    target: Option<ViewportTarget>,

    // ── Interaction ───────────────────────────────────────
    camera: OrbitCamera,
    drag: Option<DragState>,
    /// Computed once per frame so `pick` and `gizmo` agree on the
    /// projection used to test the cursor.
    last_aspect: f32,
    last_rect: Option<egui::Rect>,
}

struct ViewportTarget {
    color: TextureHandle,
    depth: TextureHandle,
    width: u32,
    height: u32,
    egui_id: egui::TextureId,
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    axis: Axis,
    /// Initial value of `s` (axis-line parameter) at drag start —
    /// subtracted from current `s` to get the per-frame translation
    /// in world space.
    initial_s: f32,
    /// Local-space `Transform.position` at drag start. We persist this
    /// (plus a per-frame local-delta) so the file write-back stays in
    /// the parent-relative space the loader expects.
    start_local_position: Vec3,
    /// World-space position of the entity at drag start. Used as the
    /// axis line's origin for closest-point math.
    line_origin: Vec3,
    /// Inverse of the parent's world matrix at drag start. Multiplying
    /// a world-space delta by this yields the equivalent local-space
    /// delta (rotation/scale-aware).
    parent_world_inv: Mat4,
}

impl Viewport {
    pub fn new(rhi: &mut WgpuRhi) -> RhiResult<Self> {
        // Bind group 0: camera uniform.
        let camera_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("camera.bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(CAMERA_UNIFORM_SIZE),
                },
                count: None,
            }],
        })?;
        let camera_buffer = rhi.create_buffer(&BufferDesc {
            label: Some("camera.ubo"),
            size: CAMERA_UNIFORM_SIZE,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let camera_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("camera.bg"),
            layout: camera_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: camera_buffer,
                    offset: 0,
                    size: Some(CAMERA_UNIFORM_SIZE),
                },
            }],
        })?;

        // Bind group 1: per-instance uniform with dynamic offset.
        let instance_bgl = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("instance.bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::ALL_GRAPHICS,
                binding_type: BindingType::UniformBuffer {
                    has_dynamic_offset: true,
                    min_binding_size: Some(INSTANCE_UNIFORM_SIZE),
                },
                count: None,
            }],
        })?;
        let instance_buffer = rhi.create_buffer(&BufferDesc {
            label: Some("instance.ubo"),
            size: INSTANCE_STRIDE * MAX_INSTANCES as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryLocation::Device,
        })?;
        let instance_bg = rhi.create_bind_group(&BindGroupDesc {
            label: Some("instance.bg"),
            layout: instance_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: instance_buffer,
                    offset: 0,
                    size: Some(INSTANCE_UNIFORM_SIZE),
                },
            }],
        })?;

        // Cube mesh.
        let cube = mesh::build_unit_cube();
        let (cube_vb, cube_ib, cube_index_count) = upload_mesh(rhi, "cube", &cube)?;

        // Built-in lit shader (HLSL → DXC → SPIR-V, with WGSL fallback).
        let (vs, fs) = load_lit_shaders(rhi)?;
        let pipeline = build_pipeline(rhi, vs, fs, camera_bgl, instance_bgl)?;
        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);

        let camera = OrbitCamera::new(Vec3::ZERO, 6.0);

        Ok(Self {
            cube_vb,
            cube_ib,
            cube_index_count,
            camera_buffer,
            camera_bgl,
            camera_bg,
            instance_buffer,
            instance_bgl,
            instance_bg,
            pipeline,
            preview_shader: None,
            preview_shader_mtime: None,
            preview_error: None,
            target: None,
            camera,
            drag: None,
            last_aspect: 1.0,
            last_rect: None,
        })
    }

    fn ensure_target(
        &mut self,
        rhi: &mut WgpuRhi,
        renderer: &Arc<RwLock<egui_wgpu::Renderer>>,
        width: u32,
        height: u32,
    ) -> RhiResult<()> {
        if let Some(t) = &self.target {
            if t.width == width && t.height == height {
                return Ok(());
            }
            let old = self.target.take().expect("checked above");
            rhi.destroy_texture(old.color);
            rhi.destroy_texture(old.depth);
            renderer.write().free_texture(&old.egui_id);
        }
        let color = rhi.create_texture(&TextureDesc {
            label: Some("viewport.color"),
            dimension: TextureDimension::D2 {
                width,
                height,
                array_layers: 1,
            },
            format: COLOR_FORMAT,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            mip_levels: 1,
            sample_count: 1,
        })?;
        let depth = rhi.create_texture(&TextureDesc {
            label: Some("viewport.depth"),
            dimension: TextureDimension::D2 {
                width,
                height,
                array_layers: 1,
            },
            format: DEPTH_FORMAT,
            usage: TextureUsage::RENDER_ATTACHMENT,
            mip_levels: 1,
            sample_count: 1,
        })?;
        let view = rhi
            .texture_view(color)
            .expect("just-created texture has a view");
        let egui_id =
            renderer
                .write()
                .register_native_texture(rhi.device(), view, wgpu::FilterMode::Linear);
        self.target = Some(ViewportTarget {
            color,
            depth,
            width,
            height,
            egui_id,
        });
        Ok(())
    }

    /// Try to swap in a new shader compiled from `path`. On failure the
    /// existing pipeline is kept and the error is stored on
    /// `self.preview_error` for the overlay to display.
    pub fn try_load_preview_shader(
        &mut self,
        rhi: &mut WgpuRhi,
        path: &std::path::Path,
    ) -> Result<(), String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let (vs, fs) = compile_lit_pair(rhi, &source, path)?;
        let pipeline = build_pipeline(rhi, vs, fs, self.camera_bgl, self.instance_bgl)
            .map_err(|e| format!("pipeline: {e}"))?;
        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);

        // On success, replace.
        let old = std::mem::replace(&mut self.pipeline, pipeline);
        rhi.destroy_pipeline(old);
        self.preview_shader = Some(path.to_path_buf());
        self.preview_shader_mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok();
        self.preview_error = None;
        log::info!("viewport: shader preview → {}", path.display());
        Ok(())
    }

    /// Revert to the built-in lit shader if a preview is active.
    pub fn clear_preview_shader(&mut self, rhi: &mut WgpuRhi) {
        if self.preview_shader.is_none() {
            return;
        }
        let Ok((vs, fs)) = load_lit_shaders(rhi) else {
            return;
        };
        match build_pipeline(rhi, vs, fs, self.camera_bgl, self.instance_bgl) {
            Ok(pipeline) => {
                let old = std::mem::replace(&mut self.pipeline, pipeline);
                rhi.destroy_pipeline(old);
                self.preview_shader = None;
                self.preview_shader_mtime = None;
                self.preview_error = None;
            }
            Err(e) => log::warn!("viewport: failed to revert built-in shader: {e}"),
        }
        rhi.destroy_shader(vs);
        rhi.destroy_shader(fs);
    }

    /// Re-test the previewed shader's mtime; recompile if the file
    /// changed on disk. Cheap: a single fs::metadata call when there's
    /// nothing to do.
    pub fn poll_preview_shader(&mut self, rhi: &mut WgpuRhi) {
        let Some(path) = self.preview_shader.clone() else {
            return;
        };
        let new_mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if new_mtime == self.preview_shader_mtime {
            return;
        }
        if let Err(e) = self.try_load_preview_shader(rhi, &path) {
            self.preview_error = Some(e);
            self.preview_shader_mtime = new_mtime;
        }
    }

    /// Main per-frame render + interaction. `scene` may be `None` if
    /// no project is loaded — the viewport renders a blank slate.
    pub fn render_and_show(
        &mut self,
        rhi: &mut WgpuRhi,
        renderer: &Arc<RwLock<egui_wgpu::Renderer>>,
        ui: &mut egui::Ui,
        mut scene: Option<&mut Scene>,
        selection: &mut Selection,
    ) {
        let avail = ui.available_size_before_wrap();
        let width = avail.x.clamp(1.0, 4096.0) as u32;
        let height = avail.y.clamp(1.0, 4096.0) as u32;

        if let Err(e) = self.ensure_target(rhi, renderer, width, height) {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("Viewport target: {e}"));
            return;
        }
        let Some(target_ids) = self.target.as_ref().map(|t| (t.egui_id, t.width, t.height)) else {
            return;
        };

        // Allocate the interactive rect.
        let id = ui.id().with("ocarina-viewport");
        let (rect, _alloc) = ui.allocate_exact_size(
            egui::vec2(target_ids.1 as f32, target_ids.2 as f32),
            egui::Sense::hover(),
        );
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());
        self.last_rect = Some(rect);
        let aspect = (target_ids.1 as f32) / (target_ids.2 as f32).max(1.0);
        self.last_aspect = aspect;

        if response.clicked() || response.drag_started() {
            response.request_focus();
        }

        // Collect entities + their world-space matrices once per frame
        // so picking / gizmo / draw all agree. World matrices fold the
        // local Transform with every ancestor folder's Transform —
        // the canonical hierarchy walk lives in `Scene::world_transforms`.
        let mut entities: Vec<(Entity, Mat4)> = Vec::new();
        let mut basic_meshes: Vec<(Entity, Vec4)> = Vec::new();
        if let Some(scene) = scene.as_deref_mut() {
            let world_xforms = scene.world_transforms();
            for (e, m) in world_xforms {
                entities.push((e, m));
            }
            // Stable order for deterministic picking when AABBs overlap.
            entities.sort_by_key(|(e, _)| *e);

            let world = &mut scene.world;
            let mut query = world.query::<(Entity, &ocarina_core::components::BasicMesh)>();
            for (e, m) in query.iter(world) {
                basic_meshes.push((e, m.tint));
            }
        }

        // Mouse → interaction: gizmo drag has priority, then picking,
        // then camera controls.
        self.process_input(&response, ui, &entities, scene.as_deref_mut(), selection);

        // Update camera UBO.
        let cam_uniform = CameraUniform {
            view_proj: self.camera.view_projection(aspect),
            // Direction TO the sun (Lambert convention: positive when
            // surface normals point toward the light). Sun is overhead
            // and a touch behind/right — so an upward-facing surface
            // gets the most diffuse.
            light_dir: Vec4::new(0.35, 1.0, 0.85, 0.0).normalize(),
            // base_color is "neutral white" — per-entity tint in the
            // instance buffer multiplies on top.
            base_color: Vec4::ONE,
        };
        rhi.write_buffer(self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        // Build the instance batch: every entity, then gizmo handles
        // for the selection.
        let mut batch: Vec<DrawInstance> = Vec::with_capacity(entities.len() + 3);
        for (entity, world) in &entities {
            let tint = basic_meshes
                .iter()
                .find(|(e, _)| e == entity)
                .map(|(_, tint)| *tint)
                .unwrap_or_else(|| Vec4::new(0.78, 0.82, 0.90, 1.0));
            let highlight = selection.entity == Some(*entity);
            let final_tint = if highlight { tint * 1.4 + Vec4::splat(0.15) } else { tint };
            batch.push(DrawInstance {
                model: *world,
                tint: final_tint,
            });
        }

        // Gizmo bars at the selection — placed at the entity's WORLD
        // position so they appear where the cube actually is, even when
        // the entity sits inside a translated/rotated parent folder.
        let gizmo_origin = selection
            .entity
            .and_then(|e| entities.iter().find(|(eid, _)| *eid == e).map(|(_, m)| *m))
            .map(|m| m.col(3).truncate());
        if let Some(origin) = gizmo_origin {
            batch.extend_from_slice(&build_gizmo_instances(origin));
        }

        // Upload the instance buffer (one write covers the whole batch).
        let mut instance_bytes: Vec<u8> = vec![0; (batch.len() * INSTANCE_STRIDE as usize).max(1)];
        for (i, inst) in batch.iter().enumerate() {
            let offset = i * INSTANCE_STRIDE as usize;
            let uniform = InstanceUniform {
                model: inst.model,
                tint: inst.tint,
            };
            instance_bytes[offset..offset + INSTANCE_UNIFORM_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if !batch.is_empty() {
            rhi.write_buffer(self.instance_buffer, 0, &instance_bytes);
        }

        let Some(target_full) = &self.target else {
            return;
        };

        // Record render pass.
        let mut list = CommandList::new();
        {
            let color_attachments = [Some(ColorAttachment {
                view: target_full.color,
                resolve_target: None,
                load: LoadOp::Clear([0.07, 0.09, 0.12, 1.0]),
                store: StoreOp::Store,
            })];
            let depth_attachment = DepthStencilAttachment {
                view: target_full.depth,
                depth_load: LoadOp::Clear(1.0),
                depth_store: StoreOp::Store,
                stencil_load: LoadOp::Clear(0),
                stencil_store: StoreOp::Discard,
                read_only_depth: false,
                read_only_stencil: false,
            };
            let mut pass = list.begin_render_pass(&RenderPassDesc {
                label: Some("viewport.scene"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(depth_attachment),
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, self.camera_bg, &[]);
            pass.set_vertex_buffer(0, self.cube_vb, 0, None);
            pass.set_index_buffer(self.cube_ib, 0, None, IndexFormat::Uint32);
            for i in 0..batch.len().min(MAX_INSTANCES as usize) {
                let dyn_offset = (i as u64 * INSTANCE_STRIDE) as u32;
                pass.set_bind_group(1, self.instance_bg, &[dyn_offset]);
                pass.draw_indexed(0..self.cube_index_count, 0, 0..1);
            }
        }
        if let Err(e) = rhi.submit(list) {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("Submit: {e}"));
            return;
        }

        // Paint texture.
        ui.painter().image(
            target_ids.0,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Overlay: stats + selection + shader-preview status.
        let mut info = format!(
            "{} entities   dist {:.1}",
            entities.len(),
            self.camera.distance
        );
        if let Some(e) = selection.entity {
            info.push_str(&format!("   sel {:?}", e));
        }
        if let Some(p) = &self.preview_shader {
            info.push_str(&format!(
                "   shader: {}",
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
            ));
        }
        ui.painter().text(
            rect.min + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            info,
            egui::FontId::proportional(11.0),
            egui::Color32::from_white_alpha(180),
        );
        if let Some(err) = &self.preview_error {
            ui.painter().text(
                rect.min + egui::vec2(8.0, 22.0),
                egui::Align2::LEFT_TOP,
                format!("shader: {err}"),
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(240, 140, 140),
            );
        }
    }

    fn process_input(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        entities: &[(Entity, Mat4)],
        scene: Option<&mut Scene>,
        selection: &mut Selection,
    ) {
        let aspect = self.last_aspect;
        let rect = match self.last_rect {
            Some(r) => r,
            None => return,
        };
        let pointer_pos = response.interact_pointer_pos().or_else(|| response.hover_pos());

        // ── Drag handling ──
        // Active drag continues until the user releases the primary button.
        if let Some(drag) = self.drag {
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pos) = pointer_pos {
                    if let Some(new_local_pos) = drag_translate(
                        &self.camera,
                        aspect,
                        rect,
                        pos,
                        drag,
                    ) {
                        if let Some(scene) = scene {
                            if let Some(e) = selection.entity {
                                if let Some(mut t) = scene.world.get_mut::<EcsTransform>(e) {
                                    t.position = new_local_pos;
                                }
                            }
                        }
                    }
                }
                return;
            }
            // Drag ended — splice the file with the entity's local
            // (parent-relative) transform. The watcher will re-apply
            // through the loader on the next pump tick.
            self.drag = None;
            if let Some(path) = selection.transform_file.clone() {
                if let Some(e) = selection.entity {
                    if let Some(scene) = scene {
                        if let Some(t) = scene.world.get::<EcsTransform>(e).copied() {
                            if let Err(err) = crate::transform_io::write_transform(
                                &path,
                                t.position,
                                t.rotation,
                                t.scale,
                            ) {
                                log::warn!("transform write-back failed: {err}");
                            } else {
                                log::info!(
                                    "wrote transform → {} (local pos {:.2?})",
                                    path.display(),
                                    t.position
                                );
                            }
                        }
                    }
                }
            }
            return;
        }

        // ── Drag start: did we hit a gizmo handle? ──
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let (Some(pos), Some(sel_entity)) = (pointer_pos, selection.entity) {
                if let Some((_, world_m)) =
                    entities.iter().find(|(e, _)| *e == sel_entity).copied()
                {
                    let world_pos = world_m.col(3).truncate();
                    let (ray_o, ray_d) = ray_for_pointer(&self.camera, aspect, rect, pos);
                    if let Some(axis) = pick_gizmo_axis(world_pos, ray_o, ray_d) {
                        let s = closest_point_on_axis(world_pos, axis.dir(), ray_o, ray_d)
                            .unwrap_or(0.0);
                        // Capture local position and parent's world
                        // matrix at drag start. Per-frame deltas are
                        // computed in world space, then transformed
                        // into local space via parent_world_inv before
                        // being persisted.
                        let (start_local, parent_world_inv) = match scene {
                            Some(scene) => {
                                let local = scene
                                    .world
                                    .get::<EcsTransform>(sel_entity)
                                    .map(|t| t.position)
                                    .unwrap_or(Vec3::ZERO);
                                let parent_world = scene.parent_world(sel_entity);
                                (local, parent_world.inverse())
                            }
                            None => (Vec3::ZERO, Mat4::IDENTITY),
                        };
                        self.drag = Some(DragState {
                            axis,
                            initial_s: s,
                            start_local_position: start_local,
                            line_origin: world_pos,
                            parent_world_inv,
                        });
                        return;
                    }
                }
            }
        }

        // ── Click-pick (only if no drag started) ──
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer_pos {
                let (ray_o, ray_d) = ray_for_pointer(&self.camera, aspect, rect, pos);
                let mut best: Option<(f32, Entity)> = None;
                for (e, world) in entities {
                    let (min, max) = entity_aabb(*world);
                    if let Some(t_hit) = ray_aabb(ray_o, ray_d, min, max) {
                        if best.map_or(true, |(b, _)| t_hit < b) {
                            best = Some((t_hit, *e));
                        }
                    }
                }
                if let Some((_, e)) = best {
                    selection.entity = Some(e);
                    if let Some(scene) = scene.as_deref() {
                        selection.transform_file = file_for_entity(scene, e);
                    }
                } else {
                    // Click on empty space clears.
                    selection.clear();
                }
                return;
            }
        }

        // ── Camera controls (only when not dragging gizmo) ──
        if response.dragged_by(egui::PointerButton::Primary) {
            let d = response.drag_delta();
            let shift = ui.input(|i| i.modifiers.shift);
            if shift {
                self.camera.pan(d.x, d.y);
            } else {
                self.camera.orbit(d.x, d.y);
            }
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let d = response.drag_delta();
            self.camera.pan(d.x, d.y);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.001 {
                self.camera.zoom(scroll * 0.02);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct DrawInstance {
    model: Mat4,
    tint: Vec4,
}

/// World-space AABB for a unit cube under the given world matrix.
/// Conservative: rotated cubes inflate to the smallest axis-aligned
/// box containing the rotated geometry.
fn entity_aabb(world: Mat4) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &dx in &[-0.5_f32, 0.5] {
        for &dy in &[-0.5_f32, 0.5] {
            for &dz in &[-0.5_f32, 0.5] {
                let p = (world * Vec4::new(dx, dy, dz, 1.0)).truncate();
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    (lo, hi)
}

fn ray_for_pointer(
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    pos: egui::Pos2,
) -> (Vec3, Vec3) {
    let u = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
    let v = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
    camera.ray_through_uv(u, v, aspect)
}

/// Translate-gizmo handles: three thin bars along +X, +Y, +Z from the
/// entity's position. Length 1 unit, thickness ~0.08.
fn build_gizmo_instances(origin: Vec3) -> [DrawInstance; 3] {
    const LENGTH: f32 = 1.2;
    const THICKNESS: f32 = 0.08;
    [
        bar_instance(origin, Axis::X, LENGTH, THICKNESS),
        bar_instance(origin, Axis::Y, LENGTH, THICKNESS),
        bar_instance(origin, Axis::Z, LENGTH, THICKNESS),
    ]
}

fn bar_instance(origin: Vec3, axis: Axis, length: f32, thickness: f32) -> DrawInstance {
    let half = length * 0.5;
    let (offset, scale) = match axis {
        Axis::X => (Vec3::new(half, 0.0, 0.0), Vec3::new(length, thickness, thickness)),
        Axis::Y => (Vec3::new(0.0, half, 0.0), Vec3::new(thickness, length, thickness)),
        Axis::Z => (Vec3::new(0.0, 0.0, half), Vec3::new(thickness, thickness, length)),
    };
    DrawInstance {
        model: Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, origin + offset),
        tint: Vec4::from_array(axis.tint()),
    }
}

/// Hit-test the cursor ray against each gizmo bar. Returns the axis of
/// the closest hit (or `None`).
fn pick_gizmo_axis(origin: Vec3, ray_o: Vec3, ray_d: Vec3) -> Option<Axis> {
    let candidates = [Axis::X, Axis::Y, Axis::Z];
    let mut best: Option<(f32, Axis)> = None;
    for axis in candidates {
        let (min, max) = bar_aabb(origin, axis);
        if let Some(t) = ray_aabb(ray_o, ray_d, min, max) {
            if best.map_or(true, |(b, _)| t < b) {
                best = Some((t, axis));
            }
        }
    }
    best.map(|(_, a)| a)
}

fn bar_aabb(origin: Vec3, axis: Axis) -> (Vec3, Vec3) {
    let length = 1.2;
    let thickness = 0.12; // padded a touch beyond the visual for forgiving picks
    match axis {
        Axis::X => (
            origin + Vec3::new(0.0, -thickness, -thickness),
            origin + Vec3::new(length, thickness, thickness),
        ),
        Axis::Y => (
            origin + Vec3::new(-thickness, 0.0, -thickness),
            origin + Vec3::new(thickness, length, thickness),
        ),
        Axis::Z => (
            origin + Vec3::new(-thickness, -thickness, 0.0),
            origin + Vec3::new(thickness, thickness, length),
        ),
    }
}

/// Compute the new local-space position to write to the entity's
/// `Transform.position`, given a drag-state and current cursor
/// position. The cursor lives in world space; `parent_world_inv`
/// rotates the world delta into the local frame so what the user sees
/// (drag along world-X) lines up with what gets persisted (a delta in
/// the entity's parent-relative space).
///
/// Returns `None` if the cursor's ray is parallel to the axis line.
fn drag_translate(
    camera: &OrbitCamera,
    aspect: f32,
    rect: egui::Rect,
    cursor: egui::Pos2,
    drag: DragState,
) -> Option<Vec3> {
    let (ray_o, ray_d) = ray_for_pointer(camera, aspect, rect, cursor);
    let s = closest_point_on_axis(drag.line_origin, drag.axis.dir(), ray_o, ray_d)?;
    let world_delta = drag.axis.dir() * (s - drag.initial_s);
    let local_delta = drag.parent_world_inv.transform_vector3(world_delta);
    Some(drag.start_local_position + local_delta)
}

/// Reverse-lookup: which `.toml` file declares this entity's transform?
/// Walks `scene.file_entity` and inspects each candidate.
fn file_for_entity(scene: &Scene, entity: Entity) -> Option<PathBuf> {
    // First, files explicitly mapped to this entity by the loader.
    let candidates: Vec<PathBuf> = scene
        .file_entity
        .iter()
        .filter(|(_, e)| **e == entity)
        .map(|(p, _)| p.clone())
        .collect();
    for path in &candidates {
        if let Ok(text) = std::fs::read_to_string(path) {
            if text.parse::<toml_edit::DocumentMut>()
                .map(|d| d.contains_key("transform"))
                .unwrap_or(false)
            {
                return Some(path.clone());
            }
        }
    }
    // Fallback: walk the entity's directory.
    let dir_entity = scene
        .dir_entity
        .iter()
        .find(|(_, e)| **e == entity)
        .map(|(p, _)| p.clone())?;
    crate::transform_io::find_transform_file(&dir_entity)
}

// ─────────────────────── Mesh / pipeline helpers ────────────────────

fn upload_mesh(
    rhi: &mut WgpuRhi,
    label: &str,
    mesh: &Mesh,
) -> RhiResult<(BufferHandle, BufferHandle, u32)> {
    let vb_size = (mesh.vertices.len() * Vertex::SIZE) as u64;
    let vb = rhi.create_buffer(&BufferDesc {
        label: Some(&format!("{label}.vb")),
        size: vb_size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(vb, 0, bytemuck::cast_slice(&mesh.vertices));

    let ib_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;
    let ib = rhi.create_buffer(&BufferDesc {
        label: Some(&format!("{label}.ib")),
        size: ib_size,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
        memory: MemoryLocation::Device,
    })?;
    rhi.write_buffer(ib, 0, bytemuck::cast_slice(&mesh.indices));

    Ok((vb, ib, mesh.indices.len() as u32))
}

fn build_pipeline(
    rhi: &mut WgpuRhi,
    vs: ShaderHandle,
    fs: ShaderHandle,
    camera_bgl: BindGroupLayoutHandle,
    instance_bgl: BindGroupLayoutHandle,
) -> RhiResult<PipelineHandle> {
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 24,
            shader_location: 2,
        },
    ];
    let vertex_buffers = [VertexBufferLayout {
        array_stride: Vertex::SIZE as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &attributes,
    }];
    let color_targets = [Some(ColorTargetState {
        format: COLOR_FORMAT,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];
    rhi.create_pipeline(&PipelineDesc {
        label: Some("lit"),
        kind: PipelineKind::Graphics(GraphicsPipelineDesc {
            vertex: VertexStage {
                shader: vs,
                buffers: &vertex_buffers,
            },
            fragment: Some(FragmentStage {
                shader: fs,
                targets: &color_targets,
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                bias: Default::default(),
            }),
            multisample: MultisampleState::default(),
        }),
        bind_group_layouts: &[camera_bgl, instance_bgl],
        push_constant_ranges: &[],
    })
}

fn load_lit_shaders(rhi: &mut WgpuRhi) -> RhiResult<(ShaderHandle, ShaderHandle)> {
    use ocarina_shader::{compile_hlsl_to_spirv, Stage};

    if ocarina_shader::dxc_available() {
        let vs_spv = compile_hlsl_to_spirv("lit.hlsl", LIT_HLSL, "vs_main", Stage::Vertex);
        let fs_spv = compile_hlsl_to_spirv("lit.hlsl", LIT_HLSL, "fs_main", Stage::Fragment);
        match (vs_spv, fs_spv) {
            (Ok(vs_words), Ok(fs_words)) => {
                let vs = rhi.create_shader(&ShaderDesc {
                    label: Some("lit.hlsl:vs_main"),
                    stage: ShaderStage::Vertex,
                    entry_point: "vs_main",
                    bytecode: ShaderBytecode::Spirv(&vs_words),
                })?;
                let fs = rhi.create_shader(&ShaderDesc {
                    label: Some("lit.hlsl:fs_main"),
                    stage: ShaderStage::Fragment,
                    entry_point: "fs_main",
                    bytecode: ShaderBytecode::Spirv(&fs_words),
                })?;
                return Ok((vs, fs));
            }
            (Err(e), _) | (_, Err(e)) => {
                log::warn!("viewport: HLSL compile failed, falling back to WGSL: {e}");
            }
        }
    } else {
        log::info!("viewport: DXC not available, authoring in WGSL");
    }
    let vs = rhi.create_shader(&ShaderDesc {
        label: Some("lit.wgsl:vs_main"),
        stage: ShaderStage::Vertex,
        entry_point: "vs_main",
        bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
    })?;
    let fs = rhi.create_shader(&ShaderDesc {
        label: Some("lit.wgsl:fs_main"),
        stage: ShaderStage::Fragment,
        entry_point: "fs_main",
        bytecode: ShaderBytecode::Wgsl(LIT_WGSL),
    })?;
    Ok((vs, fs))
}

/// Compile a user-supplied lit-style HLSL file (must define `vs_main`
/// and `fs_main`). Returns the same `(vs, fs)` pair the built-in
/// pipeline expects.
fn compile_lit_pair(
    rhi: &mut WgpuRhi,
    source: &str,
    path: &std::path::Path,
) -> Result<(ShaderHandle, ShaderHandle), String> {
    use ocarina_shader::{compile_hlsl_to_spirv, Stage};
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("preview.hlsl");
    let vs_words = compile_hlsl_to_spirv(name, source, "vs_main", Stage::Vertex)
        .map_err(|e| format!("vs_main: {e}"))?;
    let fs_words = compile_hlsl_to_spirv(name, source, "fs_main", Stage::Fragment)
        .map_err(|e| format!("fs_main: {e}"))?;
    let vs = rhi
        .create_shader(&ShaderDesc {
            label: Some("preview.hlsl:vs_main"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Spirv(&vs_words),
        })
        .map_err(|e| format!("create vs: {e}"))?;
    let fs = rhi
        .create_shader(&ShaderDesc {
            label: Some("preview.hlsl:fs_main"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Spirv(&fs_words),
        })
        .map_err(|e| format!("create fs: {e}"))?;
    Ok((vs, fs))
}
