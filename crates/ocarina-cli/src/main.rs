//! `ocarina` CLI entry point.
//!
//! Subcommands:
//! - `ocarina --version` / `-V`
//! - `ocarina schema [--output <path>] [--compact]`
//! - `ocarina lsp` — start an LSP server bound to stdin/stdout.
//! - `ocarina new component <Name>` — scaffold a new component file.
//! - `ocarina bake <project> [-o <output>]` — produce a runtime archive.
//! - `ocarina run <project> [--frames N] [--rate Hz]` — run the game
//!   headlessly (no rendering, just script ticks). Useful for smoke
//!   tests and CI.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};

// Pull in `ocarina_core` so its `#[derive(Component)]` submissions reach
// the inventory linker. Without this `use`, the schema dump and the LSP
// (when running stand-alone) would see zero registered components.
use ocarina_core as _;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--version" | "-V") | None => {
            println!("ocarina {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("schema") => schema_command(args.collect()),
        Some("lsp") => lsp_command(args.collect()),
        Some("new") => new_command(args.collect()),
        Some("bake") => bake_command(args.collect()),
        Some("build") => build_command(args.collect()),
        Some("run") => run_command(args.collect()),
        Some(cmd) => {
            eprintln!("ocarina: unknown subcommand `{cmd}`");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  ocarina --version
  ocarina schema [--output <path>] [--compact]
  ocarina lsp
  ocarina new component <Name>
  ocarina bake  <project> [-o <output>]
  ocarina build <project> [--release | --debug] [--package <name>]
  ocarina run   <project> [--frames N] [--rate Hz]"
    );
}

// ─────────────────────────── schema ────────────────────────────

fn schema_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut output: Option<PathBuf> = None;
    let mut pretty = true;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--output" | "-o" => {
                output = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow!("--output requires a path argument")
                })?));
            }
            "--compact" => pretty = false,
            other => bail!("unknown argument to `schema`: {other}"),
        }
    }

    let schemas = ocarina_schema::collect_schemas();
    let json = if pretty {
        serde_json::to_string_pretty(&schemas)?
    } else {
        serde_json::to_string(&schemas)?
    };

    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &json)?;
            eprintln!(
                "wrote {} component schemas → {}",
                schemas.len(),
                path.display()
            );
        }
        None => {
            println!("{json}");
        }
    }
    Ok(())
}

// ─────────────────────────── lsp ────────────────────────────

fn lsp_command(args: Vec<String>) -> anyhow::Result<()> {
    if !args.is_empty() {
        bail!("`ocarina lsp` takes no arguments (the LSP communicates via stdin/stdout)");
    }
    bail!(
        "`ocarina lsp` is temporarily disabled during the Lapce migration. \
         The previous LSP server depended on tower-lsp 0.20 → lsp-types 0.97, \
         which conflicts with Lapce's lsp-types 0.95 pin. Re-enabled in \
         Phase 2.5 after rebuilding on a Lapce-compatible LSP substrate."
    )
}

// ─────────────────────────── bake ────────────────────────────

fn bake_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut project: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = Some(PathBuf::from(it.next().ok_or_else(|| {
                    anyhow!("--output requires a path argument")
                })?));
            }
            other if !other.starts_with('-') && project.is_none() => {
                project = Some(PathBuf::from(other));
            }
            other => bail!("unknown argument to `bake`: {other}"),
        }
    }
    let project = project.ok_or_else(|| {
        anyhow!("usage: ocarina bake <project> [-o <output>]")
    })?;
    let output = output.unwrap_or_else(|| project.join("baked"));

    let report = ocarina_bake::bake(&project, &output)
        .with_context(|| format!("baking {}", project.display()))?;

    eprintln!(
        "baked: {} scene file(s), {} asset(s) referenced, {} copied → {}",
        report.scene_files,
        report.referenced_assets,
        report.copied_assets,
        output.display(),
    );
    if !report.missing_assets.is_empty() {
        eprintln!(
            "warning: {} referenced asset(s) missing from project (skipped):",
            report.missing_assets.len()
        );
        for m in &report.missing_assets {
            eprintln!("  - {m}");
        }
    }
    Ok(())
}

// ─────────────────────────── build ────────────────────────────

fn build_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut project: Option<PathBuf> = None;
    let mut release = true;
    let mut package: Option<String> = None;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--release" => release = true,
            "--debug" => release = false,
            "--package" | "-p" => {
                package = Some(it.next().ok_or_else(|| anyhow!("--package requires a name"))?);
            }
            other if !other.starts_with('-') && project.is_none() => {
                project = Some(PathBuf::from(other));
            }
            other => bail!("unknown argument to `build`: {other}"),
        }
    }
    let project = project.ok_or_else(|| {
        anyhow!("usage: ocarina build <project> [--release | --debug] [--package <name>]")
    })?;
    let project = project
        .canonicalize()
        .with_context(|| format!("canonicalize {}", project.display()))?;

    // The project root must be a Cargo crate (or workspace) — that's
    // the static-link template the binary lives in. We don't generate
    // one here; `ocarina new project` (Phase 6+ work) will scaffold
    // one. For now we delegate straight to cargo.
    let cargo_toml = project.join("Cargo.toml");
    if !cargo_toml.exists() {
        bail!(
            "no Cargo.toml at {} — `ocarina build` expects the project root to be a Cargo crate \
             (binary that links ocarina-runtime + your scripts). Run `ocarina new project <name>` \
             once that scaffolding lands, or hand-roll one for now.",
            project.display()
        );
    }

    let mut cmd = std::process::Command::new("cargo");
    cmd.current_dir(&project).arg("build");
    if release {
        cmd.arg("--release");
    }
    if let Some(pkg) = &package {
        cmd.arg("-p").arg(pkg);
    }
    eprintln!("running: cargo build {} {} (in {})",
        if release { "--release" } else { "" },
        package.as_ref().map(|p| format!("-p {p}")).unwrap_or_default(),
        project.display(),
    );
    let status = cmd.status().context("invoke cargo")?;
    if !status.success() {
        bail!("cargo build failed (exit {:?})", status.code());
    }

    // Surface the output binary path. `cargo metadata` knows the
    // real target dir even when the project is a workspace member
    // (where binaries land in the workspace root's `target/`, not
    // the project subdir's).
    let target_dir = match cargo_target_directory(&project) {
        Some(p) => p,
        None => project.join("target"),
    };
    let profile_dir = target_dir.join(if release { "release" } else { "debug" });
    let bin_name = package
        .clone()
        .or_else(|| cargo_binary_name(&project))
        .unwrap_or_else(|| {
            project
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string()
        });
    let mut candidate = profile_dir.join(&bin_name);
    if cfg!(windows) {
        candidate.set_extension("exe");
    }
    if candidate.exists() {
        eprintln!("built → {}", candidate.display());
    } else {
        eprintln!(
            "built; binary should be in {} (couldn't auto-locate `{}`)",
            profile_dir.display(),
            bin_name,
        );
    }
    Ok(())
}

/// Run `cargo metadata` to find the workspace's real target directory.
fn cargo_target_directory(project: &std::path::Path) -> Option<PathBuf> {
    let out = std::process::Command::new("cargo")
        .current_dir(project)
        .args(["metadata", "--no-deps", "--format-version=1"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = std::str::from_utf8(&out.stdout).ok()?;
    let json: serde_json::Value = serde_json::from_str(text).ok()?;
    json.get("target_directory")?
        .as_str()
        .map(PathBuf::from)
}

/// First binary target name in the project's Cargo.toml, if any.
fn cargo_binary_name(project: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("cargo")
        .current_dir(project)
        .args(["metadata", "--no-deps", "--format-version=1"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = std::str::from_utf8(&out.stdout).ok()?;
    let json: serde_json::Value = serde_json::from_str(text).ok()?;
    let packages = json.get("packages")?.as_array()?;
    for pkg in packages {
        if let Some(targets) = pkg.get("targets").and_then(|v| v.as_array()) {
            for target in targets {
                let kinds = target
                    .get("kind")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>())
                    .unwrap_or_default();
                if kinds.contains(&"bin") {
                    if let Some(name) = target.get("name").and_then(|v| v.as_str()) {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

// ─────────────────────────── run ────────────────────────────

fn run_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut project: Option<PathBuf> = None;
    let mut frames: Option<u64> = None;
    let mut rate_hz: f32 = 60.0;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--frames" => {
                frames = Some(it.next().ok_or_else(|| anyhow!("--frames N"))?.parse()?);
            }
            "--rate" => {
                rate_hz = it.next().ok_or_else(|| anyhow!("--rate Hz"))?.parse()?;
            }
            other if !other.starts_with('-') && project.is_none() => {
                project = Some(PathBuf::from(other));
            }
            other => bail!("unknown argument to `run`: {other}"),
        }
    }
    let project = project.ok_or_else(|| {
        anyhow!("usage: ocarina run <project> [--frames N] [--rate Hz]")
    })?;

    let (mut runtime, report) = ocarina_runtime::Runtime::from_project(&project);
    if !report.errors.is_empty() {
        eprintln!("scene loaded with {} error(s):", report.errors.len());
        for e in &report.errors {
            eprintln!("  - {e}");
        }
    }
    eprintln!(
        "runtime: {} entities, {} script instance(s) attached",
        runtime.entity_count(),
        runtime.script_count()
    );

    if rate_hz <= 0.0 {
        bail!("--rate must be positive");
    }
    let dt = Duration::from_secs_f32(1.0 / rate_hz);
    let target_frames = frames.unwrap_or(1); // default: one tick, then exit

    let started = Instant::now();
    let mut last_log = Instant::now();
    for _ in 0..target_frames {
        let frame_start = Instant::now();
        runtime.tick(dt);
        // Sleep to maintain rate (only when frames > 1, i.e., user
        // asked for a sustained run).
        if target_frames > 1 {
            let elapsed = frame_start.elapsed();
            if let Some(remaining) = dt.checked_sub(elapsed) {
                std::thread::sleep(remaining);
            }
            if last_log.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "frame {} t={:.2}s",
                    runtime.time().frame,
                    runtime.time().elapsed()
                );
                last_log = Instant::now();
            }
        }
    }
    eprintln!(
        "ran {} frame(s) in {:.3}s",
        target_frames,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

// ─────────────────────────── new ────────────────────────────

fn new_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut it = args.into_iter();
    let kind = it
        .next()
        .ok_or_else(|| anyhow!("usage: ocarina new <kind> <name>"))?;
    match kind.as_str() {
        "component" => {
            let name = it
                .next()
                .ok_or_else(|| anyhow!("usage: ocarina new component <Name>"))?;
            scaffold_component(&name)
        }
        other => bail!("unknown kind `{other}` (try `component`)"),
    }
}

fn scaffold_component(name: &str) -> anyhow::Result<()> {
    if !is_valid_pascal(name) {
        bail!(
            "component name must be a single PascalCase identifier (got `{name}`)"
        );
    }
    let snake = to_snake_case(name);
    let target_dir = PathBuf::from("src/components");
    let target = target_dir.join(format!("{snake}.rs"));

    if target.exists() {
        bail!("file already exists: {}", target.display());
    }
    std::fs::create_dir_all(&target_dir).with_context(|| {
        format!("creating {}", target_dir.display())
    })?;

    let body = component_template(name);
    std::fs::write(&target, &body)
        .with_context(|| format!("writing {}", target.display()))?;

    // Best-effort: append a `pub mod {snake};` line to mod.rs if the
    // user already has one. Don't create one if it doesn't exist —
    // every project's module layout is different, and we'd rather not
    // surprise the user with a new file.
    let mod_rs = target_dir.join("mod.rs");
    if mod_rs.exists() {
        let line = format!("pub mod {snake};\n");
        let existing = std::fs::read_to_string(&mod_rs).unwrap_or_default();
        if !existing.contains(line.trim()) {
            let mut updated = existing;
            if !updated.ends_with('\n') {
                updated.push('\n');
            }
            updated.push_str(&line);
            std::fs::write(&mod_rs, updated)?;
        }
    }

    println!("created {}", target.display());
    if !mod_rs.exists() {
        println!(
            "note: add `pub mod {snake};` to your components module by hand \
             (no mod.rs found at {})",
            mod_rs.display()
        );
    }
    Ok(())
}

fn component_template(name: &str) -> String {
    let snake = to_snake_case(name);
    format!(
        r#"//! `{name}` component.
//!
//! Authored as a `[{snake}]` section in scene TOML files.

use serde::{{Deserialize, Serialize}};

use ocarina_core::Component;

/// Describe what `{name}` represents in your game.
#[derive(Component, Debug, Clone, Default, Serialize, Deserialize)]
pub struct {name} {{
    // Add your fields here. The derive picks up `#[ocarina(...)]`
    // attrs on each field and exposes them to the LSP / editor lens.
    //
    // Example:
    //   #[ocarina(default = "1.0")]
    //   #[serde(default)]
    //   pub strength: f32,
}}
"#
    )
}

fn is_valid_pascal(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_lower = false;
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i != 0 && prev_lower {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower = false;
        } else {
            out.push(ch);
            prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_pascal() {
        assert!(is_valid_pascal("Transform"));
        assert!(is_valid_pascal("Cave2"));
        assert!(!is_valid_pascal("transform"));
        assert!(!is_valid_pascal("12Cave"));
        assert!(!is_valid_pascal(""));
    }

    #[test]
    fn snake_case() {
        assert_eq!(to_snake_case("Transform"), "transform");
        assert_eq!(to_snake_case("CharacterController"), "character_controller");
    }
}
