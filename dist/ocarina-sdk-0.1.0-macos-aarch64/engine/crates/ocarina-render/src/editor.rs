//! Editor-only render extensions. Compiled in only when the
//! `editor` feature is on, so the runtime binary contains none of
//! this code, no struct fields for these toggles, no per-frame
//! branches checking them.
//!
//! The wireframe alt pipeline + unlit shader compile here and the
//! editor's render path picks between them based on
//! [`RenderConfig`]. Custom-shader plumbing lands once `Material`
//! grows a `shader: AssetPath` field.

use ocarina_rhi::{
    ColorTargetState, CompareFunction, PipelineHandle, PolygonMode, PrimitiveState, Rhi,
    RhiResult, TextureFormat,
};

use crate::default::{
    build_scene_pipeline_against, build_simple_pipeline_against, DefaultPipeline,
    DEFAULT_SCENE_WGSL,
};

/// Per-frame editor render toggles. The runtime never sees this
/// struct (the whole module is `cfg`-gated out).
///
/// Defaults are "everything looks like the shipped game" (wireframe
/// off, lighting on, user-supplied shaders honored). Toggling
/// individual flags swaps in the editor's debug variant.
#[derive(Debug, Clone, Copy)]
pub struct RenderConfig {
    /// Draw entity meshes as polygon-line wireframes instead of
    /// solid triangles. Useful for inspecting topology without
    /// shading getting in the way.
    pub wireframe: bool,
    /// Apply the default scene shader's lighting model. When false,
    /// the renderer uses an unlit shader that emits `tint *
    /// base_color` directly. Useful for "does this entity look right
    /// under flat reference shading?"
    pub lighting: bool,
    /// Compile + use user-supplied shaders referenced by
    /// `Material.shader`. When false, every entity draws with the
    /// built-in default scene shader regardless of the per-entity
    /// setting. Useful for isolating "is my custom shader breaking
    /// the scene?" from other rendering issues.
    pub custom_shaders: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            wireframe: false,
            lighting: true,
            custom_shaders: true,
        }
    }
}

// ─────────────────────── Editor alt pipelines ───────────────────────

const UNLIT_WGSL: &str = include_str!("../shaders/unlit.wgsl");

/// Editor-only pipeline variants.
///
/// - `wireframe` reuses the default scene shader (full PBR layout
///   with material UBO + textures) but draws polygons in line mode.
///   Drawing it goes through the same per-entity bind-group setup
///   the default pipeline does.
/// - `unlit` uses the simple pipeline layout (no material UBO, no
///   textures); the simpler binding path that mask + gizmo also
///   take. Lighting toggle "off" uses this to render flat colours.
pub struct EditorPipelines {
    pub wireframe: PipelineHandle,
    pub unlit: PipelineHandle,
}

impl EditorPipelines {
    /// Build wireframe + unlit alongside an existing default
    /// pipeline. Requires the rhi to have been initialised with
    /// `Features::POLYGON_MODE_LINE` for wireframe rendering.
    pub fn build(
        rhi: &mut dyn Rhi,
        default_pipeline: &DefaultPipeline,
        target_format: TextureFormat,
        depth_format: TextureFormat,
    ) -> RhiResult<Self> {
        // Wireframe: full scene-shader layout, polygon_mode = Line.
        let wireframe = build_scene_pipeline_against(
            rhi,
            default_pipeline.frame_bgl,
            default_pipeline.instance_bgl,
            default_pipeline.material_textures_bgl,
            "render.editor.wireframe",
            DEFAULT_SCENE_WGSL,
            target_format,
            depth_format,
            PrimitiveState {
                cull_mode: None,
                polygon_mode: PolygonMode::Line,
                ..PrimitiveState::default()
            },
        )?;

        // Unlit: simple layout (no material / texture bindings).
        let unlit = build_simple_pipeline_against(
            rhi,
            default_pipeline.frame_bgl,
            default_pipeline.simple_instance_bgl,
            "render.editor.unlit",
            UNLIT_WGSL,
            target_format,
            depth_format,
            PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
            None,
            CompareFunction::Less,
            true,
            None::<&[Option<ColorTargetState>]>,
        )?;

        Ok(Self { wireframe, unlit })
    }
}

impl RenderConfig {
    /// Pick which pipeline the renderer should bind for entity
    /// draws this frame.
    ///
    /// Returns the pipeline handle plus a flag indicating whether
    /// it's a "scene" pipeline (uses material UBO + textures, group
    /// 1/2 binding pattern) or a "simple" pipeline (just instance
    /// UBO at group 1). The renderer needs the flag to set the
    /// right number of bind groups per draw.
    #[must_use]
    pub fn select_pipeline(
        self,
        default_pipeline: &DefaultPipeline,
        editor: &EditorPipelines,
    ) -> (PipelineHandle, PipelineLayout) {
        if !self.lighting {
            (editor.unlit, PipelineLayout::Simple)
        } else if self.wireframe {
            (editor.wireframe, PipelineLayout::Scene)
        } else {
            (default_pipeline.pipeline, PipelineLayout::Scene)
        }
    }
}

/// Which bind-group layout family a pipeline expects. The renderer
/// uses this to decide whether to bind material + textures (Scene)
/// or just the instance UBO (Simple).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineLayout {
    /// Group 0 (frame), group 1 (instance + material), group 2
    /// (material textures). Default + wireframe.
    Scene,
    /// Group 0 (frame), group 1 (simple instance only). Unlit, mask,
    /// gizmo.
    Simple,
}

