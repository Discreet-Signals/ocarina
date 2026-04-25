//! RHI-level errors. Kept small: most recoverable errors live at the engine
//! layer, not the RHI layer.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RhiError {
    #[error("backend initialization failed: {0}")]
    InitFailed(String),

    #[error("adapter request failed: {0}")]
    AdapterUnavailable(String),

    #[error("device request failed: {0}")]
    DeviceUnavailable(String),

    #[error("invalid handle: {0}")]
    InvalidHandle(&'static str),

    #[error("feature not supported by this backend: {0}")]
    FeatureUnsupported(&'static str),

    #[error("resource creation failed: {0}")]
    CreationFailed(String),

    #[error("swapchain acquisition failed: {0}")]
    SwapchainAcquireFailed(String),

    #[error("surface lost; swapchain must be recreated")]
    SurfaceLost,

    #[error("out of device memory")]
    OutOfDeviceMemory,

    #[error("out of host memory")]
    OutOfHostMemory,

    #[error("backend-specific error: {0}")]
    Backend(String),
}

pub type RhiResult<T> = Result<T, RhiError>;
