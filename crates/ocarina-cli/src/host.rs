//! Host-crate generator. Reads `Ocarina.toml` and writes an
//! ephemeral cargo crate at `<project>/.ocarina/host/` that the
//! engine compiles + runs to host the project.
//!
//! The generated crate has three files:
//!   - `Cargo.toml`   -- project name + version + engine baseline
//!                       deps + the user's `[dependencies]` table.
//!   - `src/main.rs`  -- routes to `ocarina_runtime::windowed::run`,
//!                       force-links engine subsystems, includes the
//!                       generated `scripts.rs`.
//!   - `build.rs`     -- walks `<project>/scripts/*.rs` and emits
//!                       the script-include glue.
//!
//! Templates are baked in via `include_str!` and substituted with
//! simple `{placeholder}` replacement -- no external templating dep.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ocarina_manifest::Manifest;

const CARGO_TOML_TEMPLATE: &str = include_str!("../templates/Cargo.toml");
const MAIN_RS_TEMPLATE: &str = include_str!("../templates/main.rs");
const BUILD_RS_TEMPLATE: &str = include_str!("../templates/build.rs");

/// Resolve the engine workspace root at runtime. The host crate's
/// generated `Cargo.toml` has path-deps pointing into `crates/`
/// under this directory, so the same `ocarina` binary works from
/// three layouts:
///
///   1. **Override**: `$OCARINA_ENGINE_ROOT` set explicitly (CI,
///      multi-engine setups, in-tree testing).
///   2. **SDK install**: `<sdk>/bin/ocarina` next to `<sdk>/engine/`.
///      The shipped layout for both the headless tarball and the
///      editor's bundled SDK.
///   3. **Dev cargo build**: `target/{debug,release}/ocarina`,
///      where the workspace root is two parents up. Lets the same
///      binary run from a fresh `cargo build` without needing to
///      install the SDK.
///
/// Each candidate is rejected if it doesn't have a `crates/ocarina-core`
/// directory (the marker we treat as "this is an Ocarina engine root").
fn engine_root() -> Result<PathBuf> {
    // (1) Explicit override.
    if let Ok(raw) = std::env::var("OCARINA_ENGINE_ROOT") {
        let candidate = PathBuf::from(&raw);
        if candidate.join("crates").join("ocarina-core").is_dir() {
            return candidate.canonicalize().with_context(|| {
                format!("canonicalize OCARINA_ENGINE_ROOT=`{raw}`")
            });
        }
        bail!(
            "OCARINA_ENGINE_ROOT=`{raw}` is set but doesn't look like an engine root \
             (no `crates/ocarina-core` directory found)"
        );
    }

    // (2) + (3) Locate via the binary's own path. `current_exe`
    // resolves symlinks on Linux/macOS, so a `/usr/local/bin/ocarina`
    // symlink into `~/.ocarina/sdk/bin/ocarina` ends up at the SDK
    // layout, not at `/usr/local/bin/`.
    let exe = std::env::current_exe()
        .context("locate the ocarina cli binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("ocarina cli binary has no parent directory"))?;

    // SDK layout: <sdk>/bin/ocarina, engine at <sdk>/engine.
    let sdk_engine = bin_dir.join("..").join("engine");
    if sdk_engine.join("crates").join("ocarina-core").is_dir() {
        return sdk_engine.canonicalize().with_context(|| {
            format!("canonicalize SDK engine root `{}`", sdk_engine.display())
        });
    }

    // Dev cargo build: target/{debug,release}/ocarina, workspace
    // root two parents up.
    let dev_root = bin_dir.join("..").join("..");
    if dev_root.join("crates").join("ocarina-core").is_dir()
        && dev_root.join("Cargo.toml").is_file()
    {
        return dev_root.canonicalize().with_context(|| {
            format!("canonicalize dev workspace `{}`", dev_root.display())
        });
    }

    bail!(
        "cannot locate the Ocarina engine relative to `{}`. Searched:\n  \
         - $OCARINA_ENGINE_ROOT (unset)\n  \
         - SDK layout: `{}`\n  \
         - dev workspace: `{}`",
        exe.display(),
        sdk_engine.display(),
        dev_root.display(),
    )
}

/// `<project>/.ocarina/host/` -- the generated cargo crate's root.
pub fn host_dir(project_root: &Path) -> PathBuf {
    project_root.join(".ocarina").join("host")
}

/// Hash of the (project, manifest, engine_root) inputs that drive
/// `Cargo.toml` regeneration. Stored in `<host>/.stamp`; if it
/// matches we skip regen entirely on subsequent builds.
fn input_hash(manifest: &Manifest, engine_root: &Path, project_root: &Path) -> u64 {
    let mut h = DefaultHasher::new();
    manifest.project.name.hash(&mut h);
    manifest.project.version.hash(&mut h);
    manifest.project.description.hash(&mut h);
    manifest.project.license.hash(&mut h);
    for (name, dep) in &manifest.dependencies {
        name.hash(&mut h);
        dep.to_cargo_toml().hash(&mut h);
    }
    engine_root.hash(&mut h);
    project_root.hash(&mut h);
    h.finish()
}

/// Render `Ocarina.toml`'s `[dependencies]` table as cargo-syntax
/// lines. Engine subsystems (`ocarina-*` names) get rewritten to
/// path-deps inside the engine workspace; everything else passes
/// through verbatim using cargo's table syntax.
fn render_user_deps(manifest: &Manifest, engine_root: &Path) -> String {
    let mut out = String::new();
    for (name, dep) in &manifest.dependencies {
        if let Some(stripped) = name.strip_prefix("ocarina-") {
            // Engine crate. Always use a path-dep; ignore whatever
            // the user wrote for version / git -- the dev shouldn't
            // need to know how the engine is laid out.
            let crate_path = engine_root.join("crates").join(format!("ocarina-{stripped}"));
            if crate_path.exists() {
                out.push_str(&format!(
                    "{name} = {{ path = \"{}\" }}\n",
                    crate_path.display()
                ));
            } else {
                // Unknown engine subsystem; pass through so cargo's
                // error surfaces the typo.
                out.push_str(&format!("{name} = {}\n", dep.to_cargo_toml()));
            }
        } else {
            out.push_str(&format!("{name} = {}\n", dep.to_cargo_toml()));
        }
    }
    out
}

/// Generate (or refresh) the host crate at `<project>/.ocarina/host/`.
/// Returns the host directory path, ready for `cargo build/run`.
///
/// On a stamp-hit (no inputs changed) this is a no-op and returns
/// immediately; cargo's own incremental compilation handles the
/// "nothing to build" case after that.
pub fn ensure_host_crate(project_root: &Path) -> Result<PathBuf> {
    let manifest = Manifest::load(project_root)?;
    let engine_root = engine_root()?;
    let host = host_dir(project_root);

    let hash = input_hash(&manifest, &engine_root, project_root);
    let stamp = host.join(".stamp");
    let stamp_matches = std::fs::read_to_string(&stamp)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(false, |existing| existing == hash);
    if stamp_matches {
        return Ok(host);
    }

    std::fs::create_dir_all(host.join("src"))
        .with_context(|| format!("creating {}", host.display()))?;

    let cargo_toml = CARGO_TOML_TEMPLATE
        .replace("@@NAME@@", &manifest.project.name)
        .replace("@@VERSION@@", &manifest.project.version)
        .replace("@@LICENSE@@", &escape_quotes(&manifest.project.license))
        .replace(
            "@@DESCRIPTION@@",
            &escape_quotes(&manifest.project.description),
        )
        .replace("@@ENGINE_ROOT@@", &engine_root.display().to_string())
        .replace("@@USER_DEPS@@", &render_user_deps(&manifest, &engine_root));

    std::fs::write(host.join("Cargo.toml"), cargo_toml)
        .with_context(|| format!("write {}/Cargo.toml", host.display()))?;
    std::fs::write(host.join("src").join("main.rs"), MAIN_RS_TEMPLATE)
        .with_context(|| format!("write {}/src/main.rs", host.display()))?;
    std::fs::write(host.join("build.rs"), BUILD_RS_TEMPLATE)
        .with_context(|| format!("write {}/build.rs", host.display()))?;
    std::fs::write(&stamp, hash.to_string())
        .with_context(|| format!("write {}", stamp.display()))?;

    // Also drop a `.gitignore` at `.ocarina/` so the dev's repo
    // doesn't pick up the cache directory by accident.
    let gitignore = project_root.join(".ocarina").join(".gitignore");
    if !gitignore.exists() {
        let _ = std::fs::write(&gitignore, "*\n");
    }

    Ok(host)
}

/// Resolve a project-root argument. Accepts an explicit path,
/// otherwise defaults to the current working directory. Errors if
/// the resulting directory has no `Ocarina.toml`.
pub fn resolve_project_root(arg: Option<PathBuf>) -> Result<PathBuf> {
    let raw = match arg {
        Some(p) => p,
        None => std::env::current_dir().context("get cwd")?,
    };
    let canonical = raw
        .canonicalize()
        .with_context(|| format!("canonicalize {}", raw.display()))?;
    if !canonical.join(ocarina_manifest::MANIFEST_FILENAME).exists() {
        return Err(anyhow!(
            "no `{}` found at {}; an Ocarina project must have a manifest at its root",
            ocarina_manifest::MANIFEST_FILENAME,
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn escape_quotes(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
