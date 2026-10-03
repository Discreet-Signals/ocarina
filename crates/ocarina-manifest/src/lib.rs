//! `Ocarina.toml` schema + parser.
//!
//! The manifest is the only file an Ocarina project's developer
//! maintains by hand. The CLI reads it and generates an ephemeral
//! cargo crate in `<project>/.ocarina/host/` that links the engine
//! plus any user-listed crates -- the dev never edits a Cargo.toml,
//! never writes a main.rs, never touches a build.rs.
//!
//! ## Example
//!
//! ```toml
//! [project]
//! name        = "my-game"
//! version     = "0.1.0"
//! description = "An ocarina-engine game."
//! license     = "MIT OR Apache-2.0"
//!
//! [dependencies]
//! # Engine subsystems beyond the always-on baseline (core / runtime
//! // gameplay-api / physics). Plus any external crate the dev wants
//! # available to scripts.
//! glam  = "0.30"
//! serde = { version = "1", features = ["derive"] }
//! ```
//!
//! Dependency syntax mirrors Cargo's `[dependencies]` exactly: a
//! string for plain version, or a table with any of `version` /
//! `path` / `git` / `branch` / `tag` / `rev` / `features` /
//! `default-features` / `optional`. The CLI passes these straight
//! through to the generated `Cargo.toml`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// Filename the parser looks for at the project root.
pub const MANIFEST_FILENAME: &str = "Ocarina.toml";

/// Top-level shape of `Ocarina.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub project: Project,
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
}

/// `[project]` table: the metadata cargo needs to identify the
/// generated host crate, plus the human-facing description / license
/// fields games are normally expected to carry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Project {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_license")]
    pub license: String,
}

fn default_version() -> String {
    "0.1.0".to_string()
}
fn default_license() -> String {
    "MIT OR Apache-2.0".to_string()
}

/// One entry in `[dependencies]`. Mirrors `cargo`'s dependency
/// shape: either `name = "X.Y"` (just a version) or a table with
/// the usual cargo fields.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Dependency {
    /// `name = "1.2.3"`.
    Simple(String),
    /// `name = { version = "...", features = [...], path = "...", ... }`.
    Detailed(DependencyDetail),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct DependencyDetail {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_features: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}

impl Dependency {
    /// Re-emit this dependency in cargo's TOML inline-table syntax,
    /// suitable for splicing into a generated `Cargo.toml`.
    #[must_use]
    pub fn to_cargo_toml(&self) -> String {
        match self {
            Self::Simple(v) => format!("\"{v}\""),
            Self::Detailed(d) => {
                let mut parts = Vec::new();
                if let Some(v) = &d.version {
                    parts.push(format!("version = \"{v}\""));
                }
                if let Some(p) = &d.path {
                    parts.push(format!("path = \"{p}\""));
                }
                if let Some(g) = &d.git {
                    parts.push(format!("git = \"{g}\""));
                }
                if let Some(b) = &d.branch {
                    parts.push(format!("branch = \"{b}\""));
                }
                if let Some(t) = &d.tag {
                    parts.push(format!("tag = \"{t}\""));
                }
                if let Some(r) = &d.rev {
                    parts.push(format!("rev = \"{r}\""));
                }
                if let Some(f) = &d.features {
                    let list = f
                        .iter()
                        .map(|s| format!("\"{s}\""))
                        .collect::<Vec<_>>()
                        .join(", ");
                    parts.push(format!("features = [{list}]"));
                }
                if let Some(df) = d.default_features {
                    parts.push(format!("default-features = {df}"));
                }
                if let Some(opt) = d.optional {
                    parts.push(format!("optional = {opt}"));
                }
                format!("{{ {} }}", parts.join(", "))
            }
        }
    }
}

impl Manifest {
    /// Read `Ocarina.toml` from `project_root` and parse it. Errors
    /// surface the file path so the dev knows which manifest is bad.
    pub fn load(project_root: &Path) -> Result<Self> {
        let path = project_root.join(MANIFEST_FILENAME);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let manifest: Manifest = toml::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        if self.project.name.is_empty() {
            return Err(anyhow!("[project] name must not be empty"));
        }
        if !is_valid_crate_name(&self.project.name) {
            return Err(anyhow!(
                "[project] name `{}` is not a valid crate name (lowercase ASCII letters / digits / `_` / `-`)",
                self.project.name
            ));
        }
        Ok(())
    }
}

fn is_valid_crate_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars().all(|c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_manifest() {
        let toml = r#"
            [project]
            name = "demo-game"
        "#;
        let m: Manifest = toml::from_str(toml).unwrap();
        assert_eq!(m.project.name, "demo-game");
        assert_eq!(m.project.version, "0.1.0");
        assert!(m.dependencies.is_empty());
    }

    #[test]
    fn parses_full_manifest() {
        let toml = r#"
            [project]
            name        = "demo-game"
            version     = "1.2.3"
            description = "A test game."
            license     = "GPL-3.0"

            [dependencies]
            glam        = "0.30"
            serde       = { version = "1", features = ["derive"] }
            local-lib   = { path = "../local-lib" }
        "#;
        let m: Manifest = toml::from_str(toml).unwrap();
        assert_eq!(m.project.name, "demo-game");
        assert_eq!(m.project.version, "1.2.3");
        assert_eq!(m.project.license, "GPL-3.0");
        assert_eq!(m.dependencies.len(), 3);
    }

    #[test]
    fn dependency_round_trip() {
        let d = Dependency::Detailed(DependencyDetail {
            version: Some("1".into()),
            features: Some(vec!["derive".into()]),
            ..Default::default()
        });
        assert!(d.to_cargo_toml().contains("version = \"1\""));
        assert!(d.to_cargo_toml().contains("features = [\"derive\"]"));
    }

    #[test]
    fn rejects_invalid_crate_name() {
        let toml = r#"[project]
name = "Demo Game"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        assert!(m.validate().is_err());
    }
}
