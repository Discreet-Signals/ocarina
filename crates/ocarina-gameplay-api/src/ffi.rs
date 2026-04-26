//! Legacy C-ABI plugin contract.
//!
//! Used by the editor's cdylib hot-reload path (load a `.dylib` /
//! `.so`, resolve `ocarina_plugin_init`, read `PluginInfo`). Static-
//! link consumers — the runtime, tests, the bake pipeline — don't go
//! through this; they rely on the inventory-based `ScriptRegistration`
//! collection in the parent module.
//!
//! Kept stable so editor tooling that grew up against the old contract
//! keeps working while we shift to the trait-based API.

use core::ffi::{c_char, CStr};

/// Bumped when [`PluginInfo`] or the init signature changes in a way
/// that breaks already-compiled cdylibs.
pub const PLUGIN_ABI_VERSION: u32 = 1;

/// The symbol name the editor resolves via `libloading`.
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

pub type PluginInitFn = unsafe extern "C" fn() -> PluginInfo;

// SAFETY: same rationale as before — plugin keeps string data alive.
unsafe impl Send for PluginInfo {}
unsafe impl Sync for PluginInfo {}
