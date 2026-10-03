//! Viewport renderer wrapper.
//!
//! Delegates all scene rendering to [`ocarina_render::SceneRenderer`]
//! (which is shared with the runtime) and adds the editor's overlay
//! passes (selection mask + outline + 2D gizmos) on top of the same
//! `CommandList`. The overlay state lives in
//! `ocarina-render::editor_overlays` behind the `editor` feature
//! gate -- the runtime binary contains none of that code.
//!
//! Per-frame work:
//!
//!   1. Wrap Floem's borrowed `wgpu::TextureView` as a regular
//!      `TextureHandle` for the duration of the frame.
//!   2. Resolve the selected entity's mesh + instance slot for the
//!      mask pass.
//!   3. Build a `CommandList`:
//!      - `scene.record_scene` -- sky + lit into scene_color (or
//!        the borrowed target when no post-process chain is bound).
//!      - `scene.record_editor_overlays` -- mask + outline + gizmo,
//!        cfg-gated to the `editor` feature.
//!      - `scene.record_compositor` -- chain post-process + final
//!        blit, no-op when no chain.
//!   4. `rhi.submit(list)`; release the borrowed view.

use bevy_ecs::entity::Entity;
use floem_renderer::WgpuPassCtx;
use glam::{Mat4, Vec3, Vec4};

use ocarina_core::components::{Material, Sky};
use ocarina_core::AssetPath;

use ocarina_render::editor_overlays::{
    Axis as RenderAxis, GizmoOverlay as RenderGizmoOverlay, SelectedDraw,
};
use ocarina_render::{
    EntityInstance as SceneEntityInstance, LightInstance, SceneArgs, SceneRenderer,
    ViewportCamera as SceneCamera,
};
use ocarina_rhi::{CommandList, Rhi, RhiResult, TextureFormat};
use ocarina_rhi_wgpu::WgpuRhi;

#[derive(Clone, Copy, Debug)]
pub struct ViewportCamera {
    pub view_proj: Mat4,
    pub eye_pos: Vec3,
}

impl ViewportCamera {
    #[must_use]
    pub fn standard(view_proj: Mat4) -> Self {
        Self {
            view_proj,
            eye_pos: Vec3::ZERO,
        }
    }
}

/// Editor render mode: Edit carries the currently-selected entity
/// (drives mask + outline). Play disables the editor passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Edit { selected_entity: Option<Entity> },
    Play,
}

/// One renderable entity. Mirrors [`ocarina_render::EntityInstance`]
/// plus an editor-specific `tint` slot the viewport applies as a
/// selection highlight before forwarding.
#[derive(Clone)]
pub struct EntityInstance {
    pub entity: Entity,
    pub world_xform: Mat4,
    pub tint: Vec4,
    pub mesh: Option<AssetPath>,
    pub material: Material,
}

/// Translate + rotate gizmo handles to draw at a selection origin.
#[derive(Clone, Copy, Debug)]
pub struct GizmoOverlay {
    pub origin: Vec3,
    pub scale_world: f32,
    pub active_axis: Option<crate::gizmo::Axis>,
}

pub struct ViewportRenderer {
    rhi: WgpuRhi,
    scene: SceneRenderer,
}

impl ViewportRenderer {
    pub fn from_existing(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> RhiResult<Self> {
        let target_format = wgpu_format_to_rhi(format)?;
        let mut rhi = WgpuRhi::from_existing(device, queue, adapter);
        let scene = SceneRenderer::new(&mut rhi, target_format)?;
        Ok(Self { rhi, scene })
    }

    pub fn set_project_root(&mut self, root: std::sync::Arc<std::path::PathBuf>) {
        self.scene.set_project_root(root);
    }

    #[must_use]
    pub fn matches_target_format(&self, format: wgpu::TextureFormat) -> bool {
        wgpu_format_to_rhi(format).map_or(false, |f| f == self.scene.target_format())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        entities: &[EntityInstance],
        lights: &[LightInstance],
        sky: Option<Sky>,
        scene_shader: &AssetPath,
        ctx: &WgpuPassCtx<'_>,
        camera: ViewportCamera,
        mode: RenderMode,
        gizmo: Option<GizmoOverlay>,
        _config: ocarina_render::editor::RenderConfig,
        post_process: &[AssetPath],
    ) {
        let pixel_w = ctx.rect.width().round() as u32;
        let pixel_h = ctx.rect.height().round() as u32;
        if pixel_w == 0 || pixel_h == 0 {
            return;
        }
        let (target_w, target_h) = ctx.target_size;
        if target_w == 0 || target_h == 0 {
            return;
        }

        let target = self.rhi.register_external_view(
            ctx.view.clone(),
            ctx.target_size,
            ctx.format,
        );

        // Adapt entity list. Resolve mesh handles up front so we can
        // build the SelectedDraw list with the same slot indices the
        // shared renderer will assign during record_scene.
        let selected_entity = match mode {
            RenderMode::Edit { selected_entity } => selected_entity,
            RenderMode::Play => None,
        };
        let mut scene_entities: Vec<SceneEntityInstance> = Vec::with_capacity(entities.len());
        let mut selected_draws: Vec<SelectedDraw> = Vec::new();
        for (slot, e) in entities.iter().enumerate() {
            let mesh_handle = e
                .mesh
                .as_ref()
                .and_then(|p| self.scene.meshes_mut().ensure(&mut self.rhi, p));
            if Some(e.entity) == selected_entity {
                if let Some(h) = mesh_handle {
                    selected_draws.push(SelectedDraw {
                        instance_slot: slot as u32,
                        vertex_buffer: h.vertex_buffer,
                        index_buffer: h.index_buffer,
                        index_count: h.index_count,
                    });
                }
            }
            let mut material = e.material.clone();
            material.albedo = e.tint;
            scene_entities.push(SceneEntityInstance {
                entity: e.entity,
                world_xform: e.world_xform,
                mesh: e.mesh.clone(),
                material,
            });
        }

        let scene_camera = SceneCamera {
            view_proj: camera.view_proj,
            eye_pos: camera.eye_pos,
        };
        let args = SceneArgs {
            target,
            target_size: ctx.target_size,
            rect: (
                ctx.rect.x0 as f32,
                ctx.rect.y0 as f32,
                ctx.rect.width() as f32,
                ctx.rect.height() as f32,
            ),
            camera: scene_camera,
            entities: &scene_entities,
            lights,
            sky,
            scene_shader,
            post_process,
        };

        // Convert viewport's GizmoOverlay (uses crate::gizmo::Axis) to
        // ocarina-render's GizmoOverlay (uses its own Axis). They're
        // structurally identical; the per-axis interaction helpers in
        // viewport just don't make the trip.
        let render_gizmo = gizmo.map(|g| RenderGizmoOverlay {
            origin: g.origin,
            scale_world: g.scale_world,
            active_axis: g.active_axis.map(|a| match a {
                crate::gizmo::Axis::X => RenderAxis::X,
                crate::gizmo::Axis::Y => RenderAxis::Y,
                crate::gizmo::Axis::Z => RenderAxis::Z,
            }),
        });

        let mut list = CommandList::new();
        self.scene.record_scene(&mut self.rhi, &args, &mut list);
        let paint_target = self.scene.paint_target(&args);
        self.scene.record_editor_overlays(
            &mut self.rhi,
            paint_target,
            ctx.target_size,
            (
                ctx.rect.x0 as f32,
                ctx.rect.y0 as f32,
                ctx.rect.width() as f32,
                ctx.rect.height() as f32,
            ),
            &selected_draws,
            render_gizmo.as_ref(),
            &mut list,
        );
        self.scene.record_compositor(&mut self.rhi, &args, &mut list);

        if let Err(e) = self.rhi.submit(list) {
            log::warn!("viewport: submit failed: {e}");
        }

        self.rhi.unregister_external_view(target);
    }
}

fn wgpu_format_to_rhi(fmt: wgpu::TextureFormat) -> RhiResult<TextureFormat> {
    use wgpu::TextureFormat as W;
    Ok(match fmt {
        W::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        W::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        W::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        W::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        other => {
            return Err(ocarina_rhi::RhiError::Backend(format!(
                "unsupported swapchain format for viewport renderer: {other:?}"
            )));
        }
    })
}
