//! Smallest possible gameplay plugin. Exposes just enough to prove the
//! editor can load a `cdylib`, resolve the init symbol, and read back
//! the plugin's identity.

use core::ffi::{c_char, CStr};

use ocarina_gameplay_api::{PluginInfo, PLUGIN_ABI_VERSION};

const NAME: &CStr = c"hello-gameplay";
const VERSION: &CStr = c"0.1.0";

/// ABI entry point the editor looks up via `libloading`.
///
/// SAFETY: the returned `*const c_char` pointers live for the lifetime
/// of the loaded library (they're pointers into the plugin's `.rodata`).
/// The editor keeps the library alive while it holds onto `PluginInfo`.
#[unsafe(no_mangle)]
pub extern "C" fn ocarina_plugin_init() -> PluginInfo {
    PluginInfo {
        abi_version: PLUGIN_ABI_VERSION,
        name: NAME.as_ptr() as *const c_char,
        version: VERSION.as_ptr() as *const c_char,
    }
}
