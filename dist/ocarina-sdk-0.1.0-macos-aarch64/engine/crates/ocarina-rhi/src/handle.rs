//! Typed, copy-cheap resource handles.
//!
//! Every resource kind has its own handle type so the type system prevents
//! mixing (e.g. passing a `BufferHandle` where a `TextureHandle` is expected).
//! Internally each handle is a `RawHandle` pair of `(slot, generation)` —
//! slot indexes into a backend pool, generation invalidates old handles when
//! a slot is reused. Handles are `Copy`, cheap to pass around, and opaque to
//! user code: backends mint them, user code treats them as tokens.

use std::fmt;

/// Shared representation behind every typed handle: pool slot + generation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct RawHandle {
    pub slot: u32,
    pub generation: u32,
}

impl RawHandle {
    pub const INVALID: Self = Self {
        slot: u32::MAX,
        generation: u32::MAX,
    };

    #[inline]
    #[must_use]
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.slot != u32::MAX
    }
}

impl fmt::Debug for RawHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}#{}", self.slot, self.generation)
        } else {
            write!(f, "INVALID")
        }
    }
}

impl Default for RawHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

macro_rules! define_handle {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(pub RawHandle);

        impl $name {
            pub const INVALID: Self = Self(RawHandle::INVALID);

            #[inline]
            #[must_use]
            pub const fn new(slot: u32, generation: u32) -> Self {
                Self(RawHandle::new(slot, generation))
            }

            #[inline]
            #[must_use]
            pub const fn is_valid(self) -> bool {
                self.0.is_valid()
            }

            #[inline]
            #[must_use]
            pub const fn raw(self) -> RawHandle {
                self.0
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}({:?})", stringify!($name), self.0)
            }
        }
    };
}

define_handle!(
    /// GPU-side buffer (vertex, index, uniform, storage, indirect).
    BufferHandle
);
define_handle!(
    /// GPU-side texture.
    TextureHandle
);
define_handle!(
    /// Texture sampler.
    SamplerHandle
);
define_handle!(
    /// Compiled shader module.
    ShaderHandle
);
define_handle!(
    /// Bind group layout describing the binding slots a pipeline expects.
    BindGroupLayoutHandle
);
define_handle!(
    /// Concrete bind group binding resources to a layout.
    BindGroupHandle
);
define_handle!(
    /// Graphics or compute pipeline.
    PipelineHandle
);
define_handle!(
    /// Swapchain tied to a platform surface.
    SwapchainHandle
);
