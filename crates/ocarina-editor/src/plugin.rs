//! Gameplay-plugin (cdylib) loader.
//!
//! At startup the editor tries to find a compiled gameplay dylib on
//! disk and load it via `libloading`. Successful loads surface in the
//! menu bar so the user can confirm hot-reload is wired up. Phase 3+
//! will grow this into real gameplay-system registration and re-load
//! on rebuild.

use std::path::{Path, PathBuf};

use ocarina_gameplay_api::{PluginInfo, PluginInitFn, PLUGIN_ABI_VERSION, PLUGIN_INIT_SYMBOL};

pub struct LoadedPlugin {
    /// Held to keep the library mapped. The `info` fields point into
    /// the library's `.rodata` — dropping this invalidates them.
    _lib: libloading::Library,
    pub info: PluginInfo,
    pub source_path: PathBuf,
}

impl LoadedPlugin {
    pub fn name(&self) -> &str {
        self.info.name_str()
    }
    pub fn version(&self) -> &str {
        self.info.version_str()
    }
}

#[derive(Debug)]
pub enum LoadError {
    NotFound,
    LibLoadingFailed(String),
    MissingSymbol(String),
    AbiMismatch { plugin: u32, editor: u32 },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "gameplay dylib not found at any candidate path"),
            Self::LibLoadingFailed(msg) => write!(f, "dlopen failed: {msg}"),
            Self::MissingSymbol(msg) => {
                write!(f, "ocarina_plugin_init not found in dylib: {msg}")
            }
            Self::AbiMismatch { plugin, editor } => write!(
                f,
                "plugin ABI v{plugin} incompatible with editor ABI v{editor}; rebuild the plugin"
            ),
        }
    }
}

impl std::error::Error for LoadError {}

/// Search a set of conventional locations and load the first plugin found.
pub fn try_load_default() -> Result<LoadedPlugin, LoadError> {
    for candidate in candidate_paths() {
        match try_load(&candidate) {
            Ok(p) => return Ok(p),
            Err(LoadError::NotFound) => continue,
            Err(e) => {
                log::warn!("plugin load error at {}: {e}", candidate.display());
            }
        }
    }
    Err(LoadError::NotFound)
}

fn candidate_paths() -> Vec<PathBuf> {
    // Compiled-artifact naming is platform-specific:
    //   macOS   → libhello_gameplay.dylib
    //   Linux   → libhello_gameplay.so
    //   Windows → hello_gameplay.dll
    #[cfg(target_os = "macos")]
    let lib_name = "libhello_gameplay.dylib";
    #[cfg(target_os = "linux")]
    let lib_name = "libhello_gameplay.so";
    #[cfg(target_os = "windows")]
    let lib_name = "hello_gameplay.dll";
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let lib_name = "libhello_gameplay.dylib";

    // Most common workspace-relative target paths. Try debug before
    // release since editor launches via `cargo run` typically.
    vec![
        PathBuf::from(format!("target/debug/{lib_name}")),
        PathBuf::from(format!("target/release/{lib_name}")),
        PathBuf::from(format!("../target/debug/{lib_name}")),
    ]
}

pub fn try_load(path: &Path) -> Result<LoadedPlugin, LoadError> {
    if !path.exists() {
        return Err(LoadError::NotFound);
    }
    // SAFETY: `libloading::Library::new` is unsafe because loading a
    // library runs any init routines it has, which can be arbitrary
    // code. We trust paths we control (target/debug of our own build).
    let lib = unsafe { libloading::Library::new(path) }
        .map_err(|e| LoadError::LibLoadingFailed(e.to_string()))?;
    let init: libloading::Symbol<PluginInitFn> = unsafe { lib.get(PLUGIN_INIT_SYMBOL) }
        .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
    let info = unsafe { init() };
    if info.abi_version != PLUGIN_ABI_VERSION {
        return Err(LoadError::AbiMismatch {
            plugin: info.abi_version,
            editor: PLUGIN_ABI_VERSION,
        });
    }
    // Shed the symbol borrow before we move `lib` into the return.
    drop(init);
    log::info!(
        "gameplay plugin loaded: {} v{} (abi v{}, from {})",
        info.name_str(),
        info.version_str(),
        info.abi_version,
        path.display()
    );
    Ok(LoadedPlugin {
        _lib: lib,
        info,
        source_path: path.to_path_buf(),
    })
}
