//! Ocarina SDK location helpers.
//!
//! The editor delegates Build / Run to the `ocarina` CLI inside an
//! installed SDK directory. This module owns:
//!
//!   - the search order for that directory,
//!   - the tiny `~/.ocarina/config.toml` file where the user-picked
//!     path is persisted between sessions,
//!   - the small "is this directory actually an SDK" validator the
//!     folder-picker calls before saving.
//!
//! Search order (first hit wins):
//!
//!   1. `~/.ocarina/config.toml` -- user-selected via `Configure
//!      Ocarina SDK...` in the run/play palette.
//!   2. `$OCARINA_SDK` env var -- CI / build-server override.
//!   3. `~/.ocarina/sdk/` -- the conventional install location for
//!      a manually-extracted SDK tarball.
//!   4. `Ocarina.app/Contents/Resources/sdk/` -- a future installer
//!      mode can drop the SDK inside the bundle and the editor
//!      picks it up without any user setup. Currently unused.
//!
//! Deliberate non-features: there's no "look at the editor's
//! sibling directory" or "search ~/.cargo/bin" fallback. Both led
//! to the editor matching itself or a stale `cargo install` and
//! silently spawning the wrong binary on Build/Run. If the search
//! above misses, the editor surfaces an explicit "SDK: None (click
//! to select)" entry in the palette and disables Build/Run until
//! the user picks a directory.

use std::path::{Path, PathBuf};

/// Resolved SDK install: the directory the lookup picked plus the
/// CLI binary inside it. The caller uses `cli` as the program for
/// Build/Run config, and `root` to display a human label.
#[derive(Debug, Clone)]
pub struct ResolvedSdk {
    pub root: PathBuf,
    pub cli: PathBuf,
}

/// Walk the candidate locations and return the first valid SDK.
pub fn locate_sdk() -> Option<ResolvedSdk> {
    for root in candidate_roots() {
        let cli = root.join("bin").join(cli_bin_name());
        if cli.is_file() {
            tracing::info!(
                "[ocarina-sdk] resolved {} (cli at {})",
                root.display(),
                cli.display()
            );
            return Some(ResolvedSdk { root, cli });
        }
    }
    tracing::info!("[ocarina-sdk] no SDK found in any candidate location");
    None
}

fn cli_bin_name() -> &'static str {
    if cfg!(windows) { "ocarina.exe" } else { "ocarina" }
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();

    if let Some(saved) = read_configured_sdk() {
        roots.push(saved);
    }

    if let Ok(env_sdk) = std::env::var("OCARINA_SDK") {
        roots.push(PathBuf::from(env_sdk));
    }

    #[allow(deprecated)]
    if let Some(home) = std::env::home_dir() {
        roots.push(home.join(".ocarina").join("sdk"));
    }

    // Ocarina.app/Contents/MacOS/Ocarina -> ../Resources/sdk
    if let Ok(exe) = std::env::current_exe() {
        if let Some(macos) = exe.parent() {
            if let Some(contents) = macos.parent() {
                roots.push(contents.join("Resources").join("sdk"));
            }
        }
    }

    roots
}

/// True if `dir` looks like an Ocarina SDK (has `bin/ocarina`).
/// Used by the folder picker before saving a user selection.
pub fn looks_like_sdk(dir: &Path) -> bool {
    dir.join("bin").join(cli_bin_name()).is_file()
}

/// Human-readable label for the SDK at `root`. Reads `VERSION` if
/// present (formats as `ocarina-<v>`); otherwise falls back to the
/// last directory name (lets people who installed a modded SDK to
/// e.g. `~/.ocarina/sdk-charlie/` see `sdk-charlie` in the
/// palette).
pub fn sdk_label(root: &Path) -> String {
    if let Ok(version) = std::fs::read_to_string(root.join("VERSION")) {
        let v = version.trim();
        if !v.is_empty() {
            return format!("ocarina-{v}");
        }
    }
    root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

// --- Config file IO --------------------------------------------------

fn config_path() -> Option<PathBuf> {
    #[allow(deprecated)]
    let home = std::env::home_dir()?;
    Some(home.join(".ocarina").join("config.toml"))
}

/// Read `sdk_path = "..."` out of `~/.ocarina/config.toml`. Cheap
/// line parser -- the file is expected to be tiny.
pub fn read_configured_sdk() -> Option<PathBuf> {
    let text = std::fs::read_to_string(config_path()?).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let Some((key, rest)) = trimmed.split_once('=') else {
            continue;
        };
        if key.trim() != "sdk_path" {
            continue;
        }
        let v = rest.trim().trim_matches('"');
        if v.is_empty() {
            return None;
        }
        return Some(PathBuf::from(v));
    }
    None
}

/// Persist `sdk_path` to `~/.ocarina/config.toml`. Creates the
/// parent directory if needed; overwrites any existing config.
pub fn write_configured_sdk(sdk_path: &Path) -> std::io::Result<()> {
    let path = config_path().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "no home directory")
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = format!(
        "# Written by Ocarina editor's `Configure Ocarina SDK...` command.\n\
         sdk_path = \"{}\"\n",
        sdk_path.display()
    );
    std::fs::write(path, body)
}
