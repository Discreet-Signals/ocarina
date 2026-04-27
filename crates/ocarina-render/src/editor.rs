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
    PipelineHandle, PolygonMode, PrimitiveState, Rhi, RhiResult, ShaderBytecode, ShaderDesc,
    ShaderStage, TextureFormat,
};

use crate::lit::{build_lit_graphics_pipeline, LitPipeline};

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
    /// Apply the lit shader's lighting model. When false, the
    /// renderer uses an unlit shader that emits `tint * base_color`
    /// directly. Useful for "does this entity look right under flat
    /// reference shading?"
    pub lighting: bool,
    /// Compile + use user-supplied shaders referenced by
    /// `Material.shader`. When false, every entity draws with the
    /// built-in lit shader regardless of the per-entity setting.
    /// Useful for isolating "is my custom shader breaking the
    /// scene?" from other rendering issues.
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

// ─────────────────────── Wireframe alt pipeline ───────────────────────

const UNLIT_WGSL: &str = include_str!("../shaders/unlit.wgsl");

/// Editor-only pipeline variants. The wireframe pipeline reuses the
/// lit shader but draws lines; the unlit pipeline draws solid
/// triangles with no lighting term.
///
/// Both variants share the [`LitPipeline`]'s bind group layouts so
/// the same camera + instance UBOs feed every pipeline. Switching
/// between them is `pass.set_pipeline(...)` only.
pub struct EditorPipelines {
    pub wireframe: PipelineHandle,
    pub unlit: PipelineHandle,
}

impl EditorPipelines {
    /// Build wireframe + unlit alongside an existing lit pipeline.
    /// Requires the rhi to have been initialised with
    /// `Features::POLYGON_MODE_LINE` for wireframe rendering;
    /// currently `ocarina-rhi-wgpu` requests this when the editor
    /// feature is enabled by its consumer.
    pub fn build(
        rhi: &mut dyn Rhi,
        lit: &LitPipeline,
        target_format: TextureFormat,
        depth_format: TextureFormat,
    ) -> RhiResult<Self> {
        // Wireframe: same lit shader, polygon_mode = Line.
        let lit_wgsl = include_str!("../shaders/lit.wgsl");
        let vs_w = rhi.create_shader(&ShaderDesc {
            label: Some("render.editor.wire:vs"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(lit_wgsl),
        })?;
        let fs_w = rhi.create_shader(&ShaderDesc {
            label: Some("render.editor.wire:fs"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(lit_wgsl),
        })?;
        let wireframe = build_lit_graphics_pipeline(
            rhi,
            vs_w,
            fs_w,
            lit.camera_bgl,
            lit.instance_bgl,
            target_format,
            depth_format,
            PrimitiveState {
                cull_mode: None,
                polygon_mode: PolygonMode::Line,
                ..PrimitiveState::default()
            },
        )?;
        rhi.destroy_shader(vs_w);
        rhi.destroy_shader(fs_w);

        // Unlit: separate shader, same vertex layout, same UBO bindings.
        let vs_u = rhi.create_shader(&ShaderDesc {
            label: Some("render.editor.unlit:vs"),
            stage: ShaderStage::Vertex,
            entry_point: "vs_main",
            bytecode: ShaderBytecode::Wgsl(UNLIT_WGSL),
        })?;
        let fs_u = rhi.create_shader(&ShaderDesc {
            label: Some("render.editor.unlit:fs"),
            stage: ShaderStage::Fragment,
            entry_point: "fs_main",
            bytecode: ShaderBytecode::Wgsl(UNLIT_WGSL),
        })?;
        let unlit = build_lit_graphics_pipeline(
            rhi,
            vs_u,
            fs_u,
            lit.camera_bgl,
            lit.instance_bgl,
            target_format,
            depth_format,
            PrimitiveState {
                cull_mode: None,
                ..PrimitiveState::default()
            },
        )?;
        rhi.destroy_shader(vs_u);
        rhi.destroy_shader(fs_u);

        Ok(Self { wireframe, unlit })
    }
}

impl RenderConfig {
    /// Pick which pipeline the renderer should bind for entity
    /// draws this frame.
    #[must_use]
    pub fn select_pipeline(
        self,
        lit: &LitPipeline,
        editor: &EditorPipelines,
    ) -> PipelineHandle {
        if self.wireframe {
            editor.wireframe
        } else if !self.lighting {
            editor.unlit
        } else {
            lit.pipeline
        }
    }
}
