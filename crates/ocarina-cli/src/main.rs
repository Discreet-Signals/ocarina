//! `ocarina` CLI entry point.
//!
//! Subcommands:
//! - `ocarina --version` / `-V`
//! - `ocarina schema [--output <path>] [--compact]`
//! - `ocarina lsp` — start an LSP server bound to stdin/stdout.
//! - `ocarina new component <Name>` — scaffold a new component file.

use std::path::PathBuf;

use anyhow::{anyhow, bail, Context};

// Pull in `ocarina_core` so its `#[derive(Component)]` submissions reach
// the inventory linker. Without this `use`, the schema dump and the LSP
// (when running stand-alone) would see zero registered components.
use ocarina_core as _;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--version" | "-V") | None => {
            println!("ocarina {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("schema") => schema_command(args.collect()),
        Some("lsp") => lsp_command(args.collect()),
        Some("new") => new_command(args.collect()),
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
  ocarina new component <Name>"
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
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("creating tokio runtime")?;
    runtime.block_on(ocarina_lsp::run_stdio());
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
