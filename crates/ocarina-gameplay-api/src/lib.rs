//! Stable FFI contract for Ocarina gameplay plugins.
//!
//! A *gameplay plugin* is a `cdylib` that the editor (and, eventually,
//! the shipped runtime) loads via `libloading`. The plugin exposes a
//! single `extern "C" fn ocarina_plugin_init() -> PluginInfo` symbol
//! that the editor calls to identify it and check ABI compatibility.
//!
//! Phase 2 scaffolds the loading mechanism only. Later phases grow
//! the API with system registration, hot-reload re-entry points, and
//! typed resource handles.

use core::ffi::{c_char, CStr};

/// Bumped whenever `PluginInfo`, the init signature, or any subsequent
/// registration entry point changes in a breaking way.
pub const PLUGIN_ABI_VERSION: u32 = 1;

/// The symbol name the editor resolves via `libloading`. Kept here as a
/// byte string so loader code can use it directly.
pub const PLUGIN_INIT_SYMBOL: &[u8] = b"ocarina_plugin_init\0";

/// Repr-C to keep layout stable across the FFI boundary. Strings are
/// passed as C pointers (plugin owns them, must be `'static`).
#[repr(C)]
pub struct PluginInfo {
    pub abi_version: u32,
    pub name: *const c_char,
    pub version: *const c_char,
}

impl PluginInfo {
    /// Borrow `name` as a Rust string slice. Returns `"?"` if the
    /// plugin's string is malformed.
    #[must_use]
    pub fn name_str(&self) -> &str {
        c_str(self.name)
    }
    #[must_use]
    pub fn version_str(&self) -> &str {
        c_str(self.version)
    }
}

fn c_str<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        return "?";
    }
    // SAFETY: caller promises `p` points at a NUL-terminated string
    // that outlives the borrow. Plugins produce these from `c"..."`
    // literals which are `'static`.
    unsafe {
        CStr::from_ptr(p).to_str().unwrap_or("?")
    }
}

/// Signature the editor resolves from a loaded plugin.
pub type PluginInitFn = unsafe extern "C" fn() -> PluginInfo;

// SAFETY: `PluginInfo` is `Send + Sync` as long as the plugin keeps its
// string data alive for its own lifetime — which is the same
// requirement as any other 'static data. The editor holds the loaded
// library for the lifetime of the plugin, so this is sound.
unsafe impl Send for PluginInfo {}
unsafe impl Sync for PluginInfo {}
