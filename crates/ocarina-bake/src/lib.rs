//! Project source files → runtime asset archive.
//!
//! Bake walks a project directory, identifies which files are
//! *referenced* by some `.toml`, and packages just those into an
//! archive that `ocarina-runtime` can load. Files no `[component]` /
//! `[[script]]` ever points at are silently dropped — that's the
//! "every file in the project is a candidate, only referenced ones
//! ship" rule.
//!
//! The archive layout is intentionally simple for v1:
//!
//! ```text
//! <output>/
//!   manifest.toml      list of referenced assets + scene root
//!   scene/             project's `.toml` tree, copied verbatim
//!   <asset paths>      every referenced file, preserving its
//!                      project-relative path
//! ```
//!
//! Phase 6.5 may swap this for a single tar/zip bundle and bincode-
//! serialised `World`, but the directory form is enough to exercise
//! the bake → runtime contract end-to-end.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use ocarina_schema::{iter_schemas, FieldType};
use serde::{Deserialize, Serialize};

// ─────────────────────────── Harvest ───────────────────────────

/// Set of project-relative paths a bake harvested from the TOML files.
/// Iteration order is sorted (BTreeSet) so manifests are reproducible.
#[derive(Debug, Default, Clone)]
pub struct AssetInventory {
    pub paths: BTreeSet<String>,
}

/// Walk every `.toml` under `project_root` and return the set of
/// asset paths every component instance references.
///
/// Discovery is schema-driven: for each known component (registered
/// via `#[derive(Component)]`), we know which fields are
/// `FieldType::AssetPath` and only collect those. Plus a special-case
/// for the `[[script]]` array-of-tables — its `path` field is always
/// an asset reference.
pub fn harvest_assets(project_root: &Path) -> Result<AssetInventory> {
    // 1. Schema → list of asset-path field names per section.
    let mut asset_fields: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for schema in iter_schemas() {
        let fields: Vec<String> = schema
            .fields
            .iter()
            .filter(|f| f.ty == FieldType::AssetPath)
            .map(|f| f.name.clone())
            .collect();
        asset_fields.insert(schema.name, fields);
    }

    let mut paths: BTreeSet<String> = BTreeSet::new();
    walk_toml_files(project_root, &mut |toml_path| {
        let text = match fs::read_to_string(toml_path) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("bake: read {} failed: {e}", toml_path.display());
                return;
            }
        };
        let table: toml::Table = match toml::from_str(&text) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("bake: parse {} failed: {e}", toml_path.display());
                return;
            }
        };
        collect_paths_from_table(&table, &asset_fields, &mut paths);
    })?;

    Ok(AssetInventory { paths })
}

fn collect_paths_from_table(
    table: &toml::Table,
    asset_fields: &std::collections::HashMap<String, Vec<String>>,
    out: &mut BTreeSet<String>,
) {
    for (section, value) in table {
        // `[[script]]` is the one built-in array-of-tables form.
        // Each element's `path` field is the script's asset path.
        if section == "script" {
            if let toml::Value::Array(arr) = value {
                for elem in arr {
                    if let Some(p) = elem.get("path").and_then(|v| v.as_str()) {
                        if !p.is_empty() {
                            out.insert(p.to_string());
                        }
                    }
                }
            }
            continue;
        }

        let Some(field_names) = asset_fields.get(section) else {
            continue;
        };
        let toml::Value::Table(section_table) = value else {
            continue;
        };
        for fname in field_names {
            if let Some(p) = section_table.get(fname).and_then(|v| v.as_str()) {
                if !p.is_empty() {
                    out.insert(p.to_string());
                }
            }
        }
    }
}

fn walk_toml_files(
    root: &Path,
    visit: &mut dyn FnMut(&Path),
) -> Result<()> {
    let rd = fs::read_dir(root)
        .with_context(|| format!("read_dir {}", root.display()))?;
    for entry in rd.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'))
        {
            continue;
        }
        if path.is_dir() {
            walk_toml_files(&path, visit)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            visit(&path);
        }
    }
    Ok(())
}

// ─────────────────────────── Manifest ───────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version of this manifest. Bumped when the layout
    /// changes; runtime checks for compatibility before loading.
    pub version: u32,
    /// Project-relative paths of every asset packaged in the archive.
    pub assets: Vec<String>,
}

const MANIFEST_VERSION: u32 = 1;

// ─────────────────────────── Bake ───────────────────────────

/// Run the full bake: harvest, copy scene tree, copy referenced
/// assets, write the manifest. The output directory is created if
/// it doesn't exist; existing files are overwritten.
///
/// Asset copies preserve the project-relative path. A reference to
/// `"assets/wood.png"` results in `<output>/assets/wood.png`.
pub fn bake(project_root: &Path, output: &Path) -> Result<BakeReport> {
    let project_root = project_root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", project_root.display()))?;

    let inventory = harvest_assets(&project_root)?;

    fs::create_dir_all(output)
        .with_context(|| format!("create {}", output.display()))?;

    // 1. Copy the scene tree (every .toml) into <output>/scene/.
    let scene_out = output.join("scene");
    fs::create_dir_all(&scene_out)?;
    let mut scene_files = 0usize;
    walk_toml_files(&project_root, &mut |toml_path| {
        if let Ok(rel) = toml_path.strip_prefix(&project_root) {
            let dst = scene_out.join(rel);
            if let Some(parent) = dst.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if fs::copy(toml_path, &dst).is_ok() {
                scene_files += 1;
            } else {
                log::warn!("bake: copy {} failed", toml_path.display());
            }
        }
    })?;

    // 2. Copy every referenced asset, preserving project-relative path.
    let mut copied_assets = 0usize;
    let mut missing: Vec<String> = Vec::new();
    for rel in &inventory.paths {
        let src = project_root.join(rel);
        let dst = output.join(rel);
        if !src.exists() {
            missing.push(rel.clone());
            continue;
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)
            .with_context(|| format!("copy {} → {}", src.display(), dst.display()))?;
        copied_assets += 1;
    }

    // 3. Write the manifest.
    let manifest = Manifest {
        version: MANIFEST_VERSION,
        assets: inventory.paths.iter().cloned().collect(),
    };
    let manifest_text = toml::to_string_pretty(&manifest)?;
    fs::write(output.join("manifest.toml"), manifest_text)?;

    Ok(BakeReport {
        scene_files,
        referenced_assets: inventory.paths.len(),
        copied_assets,
        missing_assets: missing,
    })
}

#[derive(Debug, Default, Clone)]
pub struct BakeReport {
    pub scene_files: usize,
    pub referenced_assets: usize,
    pub copied_assets: usize,
    pub missing_assets: Vec<String>,
}

// ─────────────────────────── Tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Force the inventory linker to retain ocarina-core's components
    /// — schemas need to be registered for the harvest to know which
    /// fields are AssetPath.
    use ocarina_core as _;

    fn temp_project(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ocarina-bake-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn harvest_collects_basic_mesh_and_script_paths() {
        let project = temp_project("harvest");
        let player = project.join("player");
        fs::create_dir_all(&player).unwrap();
        fs::write(
            player.join("player.toml"),
            "[transform]\nposition = [0,0,0]\n\
             [basic_mesh]\nsource = \"assets/player.obj\"\ntexture = \"assets/player.png\"\n\
             [[script]]\npath = \"scripts/movement.rs\"\n\
             [[script]]\npath = \"scripts/health.rs\"\n",
        )
        .unwrap();

        let inv = harvest_assets(&project).unwrap();
        let v: Vec<_> = inv.paths.iter().cloned().collect();
        assert!(v.contains(&"assets/player.obj".to_string()), "{v:?}");
        assert!(v.contains(&"assets/player.png".to_string()), "{v:?}");
        assert!(v.contains(&"scripts/movement.rs".to_string()), "{v:?}");
        assert!(v.contains(&"scripts/health.rs".to_string()), "{v:?}");

        let _ = fs::remove_dir_all(&project);
    }

    #[test]
    fn bake_drops_unreferenced_files() {
        let project = temp_project("drop");
        let cave = project.join("cave");
        fs::create_dir_all(&cave).unwrap();
        fs::write(
            cave.join("cave.toml"),
            "[basic_mesh]\nsource = \"assets/used.obj\"\n",
        )
        .unwrap();
        fs::create_dir_all(project.join("assets")).unwrap();
        fs::write(project.join("assets/used.obj"), b"used\n").unwrap();
        fs::write(project.join("assets/orphan.obj"), b"orphan\n").unwrap();

        let out = temp_project("drop-out");
        let report = bake(&project, &out).unwrap();
        assert_eq!(report.copied_assets, 1);
        assert!(out.join("assets/used.obj").exists());
        assert!(!out.join("assets/orphan.obj").exists());
        assert!(out.join("scene/cave/cave.toml").exists());
        assert!(out.join("manifest.toml").exists());

        let _ = fs::remove_dir_all(&project);
        let _ = fs::remove_dir_all(&out);
    }
}
