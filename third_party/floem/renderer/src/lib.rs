pub mod swash;
pub mod text;

use crate::text::LayoutRun;
use peniko::{
    kurbo::{Affine, Point, Rect, Shape, Stroke},
    BlendMode, BrushRef,
};
pub use resvg::tiny_skia;
pub use resvg::usvg;
use text::TextLayout;

pub mod gpu_resources;

/// Context handed to a user-provided wgpu render closure scheduled via
/// [`Renderer::submit_wgpu_pass`]. The closure runs during the host's
/// `finish()` phase, after the UI scene has been encoded into the
/// backbuffer but before `present()`, so a render pass that uses
/// `LoadOp::Load` against `view` paints on top of the UI without
/// disturbing the surrounding panel chrome.
///
/// `rect` is the panel rectangle in **framebuffer pixels** (already
/// scaled by the window's hidpi factor). Use it to configure the
/// viewport / scissor on the render pass.
///
/// `target_size` is the full swapchain texture's pixel extent
/// (`width`, `height`). wgpu requires every attachment in a single
/// render pass to share the same extent — so any depth/stencil or
/// auxiliary attachment the closure builds must be sized to
/// `target_size`, not `rect`. The `rect` still drives viewport and
/// scissor so drawing is clipped to the panel area.
pub struct WgpuPassCtx<'a> {
    pub view: &'a wgpu::TextureView,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub adapter: &'a wgpu::Adapter,
    pub rect: Rect,
    pub target_size: (u32, u32),
    pub format: wgpu::TextureFormat,
}

/// Boxed one-shot closure that renders directly into the host's
/// swapchain target. See [`Renderer::submit_wgpu_pass`].
pub type WgpuPass = Box<dyn for<'a> FnOnce(WgpuPassCtx<'a>) + 'static>;

pub struct Svg<'a> {
    pub tree: &'a usvg::Tree,
    pub hash: &'a [u8],
}

pub struct Img<'a> {
    pub img: peniko::Image,
    pub hash: &'a [u8],
}

pub trait Renderer {
    fn begin(&mut self, capture: bool);

    fn set_transform(&mut self, transform: Affine);

    fn set_z_index(&mut self, z_index: i32);

    /// Clip to a [`Shape`].
    fn clip(&mut self, shape: &impl Shape);

    fn clear_clip(&mut self);

    /// Stroke a [`Shape`].
    fn stroke<'b, 's>(
        &mut self,
        shape: &impl Shape,
        brush: impl Into<peniko::BrushRef<'b>>,
        stroke: &'s Stroke,
    );

    /// Fill a [`Shape`], using the [non-zero fill rule].
    ///
    /// [non-zero fill rule]: https://en.wikipedia.org/wiki/Nonzero-rule
    fn fill<'b>(&mut self, path: &impl Shape, brush: impl Into<BrushRef<'b>>, blur_radius: f64);

    /// Push a layer (This is not supported with Vger)
    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    );

    /// Pop a layer (This is not supported with Vger)
    fn pop_layer(&mut self);

    /// Draw a [`TextLayout`].
    ///
    /// The `pos` parameter specifies the upper-left corner of the layout object
    /// (even for right-to-left text).
    fn draw_text(&mut self, layout: &TextLayout, pos: impl Into<Point>) {
        self.draw_text_with_layout(layout.layout_runs(), pos);
    }

    fn draw_text_with_layout<'b>(
        &mut self,
        layout: impl Iterator<Item = LayoutRun<'b>>,
        pos: impl Into<Point>,
    );

    fn draw_svg<'b>(&mut self, svg: Svg<'b>, rect: Rect, brush: Option<impl Into<BrushRef<'b>>>);

    fn draw_img(&mut self, img: Img<'_>, rect: Rect);

    fn finish(&mut self) -> Option<peniko::Image>;

    /// Borrow the underlying `wgpu::Device` if the backend is GPU-based.
    /// Returns `None` for CPU backends (tiny-skia). Lets the host share
    /// the device with other wgpu-using subsystems (e.g. an embedded
    /// 3D viewport panel).
    fn wgpu_device(&self) -> Option<&wgpu::Device> {
        None
    }

    /// Borrow the underlying `wgpu::Queue` if the backend is GPU-based.
    fn wgpu_queue(&self) -> Option<&wgpu::Queue> {
        None
    }

    /// Borrow the underlying `wgpu::Adapter` if the backend is GPU-based.
    fn wgpu_adapter(&self) -> Option<&wgpu::Adapter> {
        None
    }

    /// Schedule a user-provided render pass that runs directly against
    /// the current frame's swapchain texture, after the UI scene is
    /// encoded but before `present`. Lets a host (e.g. an embedded 3D
    /// viewport panel) draw into the same backbuffer the UI is using —
    /// no offscreen texture, no readback, no compositor copy.
    ///
    /// The closure is queued during the paint pass and drained in
    /// [`finish`](Self::finish). It receives a [`WgpuPassCtx`] with the
    /// target view, the backend's shared device + queue + adapter, the
    /// panel rectangle in framebuffer pixels, and the target's texture
    /// format. The closure is responsible for building its own command
    /// encoder, beginning a render pass with `LoadOp::Load` so the
    /// panel chrome painted by the UI underneath is preserved, setting
    /// the viewport / scissor to the panel rect, drawing, and
    /// submitting the encoder via `queue.submit(...)`.
    ///
    /// Default implementation is a no-op: backends without a shared
    /// wgpu surface (CPU-rendered tiny-skia, the uninitialized
    /// renderer) silently drop the pass.
    fn submit_wgpu_pass(&mut self, _rect: Rect, _pass: WgpuPass) {}

    fn debug_info(&self) -> String {
        "Unknown".into()
    }
}
