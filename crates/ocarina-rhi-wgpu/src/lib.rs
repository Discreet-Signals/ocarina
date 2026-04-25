//! wgpu-backed RHI backend.
//!
//! Implements [`ocarina_rhi::Rhi`] on top of [`wgpu`], the cross-platform
//! WebGPU implementation that targets Vulkan / Metal / DX12 / GL natively.
//! This is backend #1; see the workspace README for the philosophy behind
//! the abstraction and the plan for additional backends.
//!
//! ## Usage
//!
//! ```no_run
//! use ocarina_rhi::Rhi;
//! use ocarina_rhi_wgpu::WgpuRhi;
//!
//! fn with_existing(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::Adapter) {
//!     let mut rhi = WgpuRhi::from_existing(device, queue, adapter);
//!     assert_eq!(rhi.backend_name(), "wgpu");
//! }
//! ```

mod capability;
mod command;
mod convert;
mod destroy;
mod pool;
mod rhi;

pub use rhi::{WgpuRhi, BACKEND_NAME};
