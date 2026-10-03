//! Swapchain descriptor. Tied to a platform surface by the backend; the RHI
//! deliberately keeps the surface type opaque — backends get it from winit via
//! `raw-window-handle`, but the RHI trait doesn't know about that detail.

use crate::format::TextureFormat;

/// Description of a swapchain. `surface` is intentionally a raw pointer kept
/// as an opaque token: backends require `&dyn HasRawWindowHandle` but
/// abstracting that here would pull in heavy traits. The wgpu backend will
/// expose a dedicated `create_swapchain` path that takes the window directly;
/// this desc is the minimal cross-backend shape.
#[derive(Debug, Clone, Copy)]
pub struct SwapchainDesc<'a> {
    pub label: Option<&'a str>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub present_mode: PresentMode,
    pub image_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    /// Tearing, lowest latency; not available everywhere.
    Immediate,
    /// Double-buffered, vsync-locked. Always available.
    Fifo,
    /// Triple-buffered, tears on late frames. Common on macOS/Linux with VRR.
    Mailbox,
    /// Fifo variant that may show a late frame immediately. On platforms
    /// without support, falls back to `Fifo`.
    FifoRelaxed,
}
