//! Folder → entity loader + TOML-file parser.
//!
//! Strategy:
//! 1. `reload_full` walks the project directory, creates one entity per
//!    folder, and loads every `.toml` file found.
//! 2. `reload_file` handles single-file edits: re-parse, replace the
//!    components declared by that file on its entity. We don't know
//!    which specific components came from the file without tracking,
//!    so for v1 we take a simple approach: *files declare components*
//!    but since multiple files in a folder can target the same entity,
//!    a file edit only replaces the components listed in that file.

use std::fs;
use std::path::{Path, PathBuf};

use bevy_ecs::entity::Entity;

use ocarina_schema::{find_loader, toml};

use crate::scene::{Scene, SceneError};

const RESERVED_PREFIX: &str = "_";

#[derive(Debug, Default)]
pub struct LoadReport {
    pub files_parsed: usize,
    pub entities_touched: usize,
    pub errors: Vec<SceneError>,
}

/// Full reload: discards everything under the root entity, walks the
/// directory tree, and re-populates the ECS.
pub fn reload_full(scene: &mut Scene) -> LoadReport {
    // Reset: drop every entity except root, clear indices.
    let to_despawn: Vec<Entity> = scene
        .dir_entity
        .iter()
        .filter_map(|(_, &e)| (e != scene.root).then_some(e))
        .collect();
    for e in to_despawn {
        scene.world.despawn(e);
    }
    scene.dir_entity.clear();
    scene
        .dir_entity
        .insert(scene.root_dir.clone(), scene.root);
    scene.file_entity.clear();
    // Also strip components from the root entity — we re-add them below.
    // Pragmatic approach: despawn + respawn the root so no stale
    // components linger. Any external `Entity` references to the old
    // root are invalidated, but Scene owns the root, so this is fine.
    scene.world.despawn(scene.root);
    scene.root = scene.world.spawn_empty().id();
    scene
        .dir_entity
        .insert(scene.root_dir.clone(), scene.root);

    let mut report = LoadReport::default();
    walk_and_load(scene, &scene.root_dir.clone(), &mut report);
    report
}

/// Re-parse a single file and replace the components it declares on
/// the entity of its parent folder.
///
/// The path is canonicalized to match the absolute form `Scene::empty`
/// stores. Without this, a path supplied in a different form (relative
/// vs absolute, with vs. without symlink resolution) would map to a
/// fresh entity and you'd end up with duplicates that all render at
/// the same place in the viewport.
pub fn reload_file(scene: &mut Scene, path: &Path) -> Result<(), SceneError> {
    if !is_toml(path) {
        return Ok(());
    }
    let path = canonical(path);
    let parent_dir = path.parent().unwrap_or(&scene.root_dir).to_path_buf();
    let entity = scene.entity_for_dir(&parent_dir);
    apply_toml_file(scene, entity, &path)?;
    scene.file_entity.insert(path, entity);
    Ok(())
}

/// Remove the components a file contributed, when the file is deleted.
///
/// Minimal implementation: we don't track which components came from
/// which file, so we just do a full reload of the affected folder
/// by re-parsing the surviving files in it.
pub fn remove_file(scene: &mut Scene, path: &Path) -> LoadReport {
    let path = canonical(path);
    scene.file_entity.remove(&path);
    let parent_dir = path.parent().unwrap_or(&scene.root_dir).to_path_buf();
    let entity = match scene.dir_entity.get(&parent_dir) {
        Some(&e) => e,
        None => return LoadReport::default(),
    };
    let mut report = LoadReport::default();
    reload_folder_components(scene, entity, &parent_dir, &mut report);
    report
}

// ───────────────────────── Internals ─────────────────────────

fn walk_and_load(scene: &mut Scene, dir: &Path, report: &mut LoadReport) {
    let entity = scene.entity_for_dir(dir);
    reload_folder_components(scene, entity, dir, report);

    let rd = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => {
            report.errors.push(SceneError::Io {
                path: dir.to_path_buf(),
                source: e,
            });
            return;
        }
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            // Skip hidden / reserved directories.
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.starts_with('.') {
                    continue;
                }
            }
            walk_and_load(scene, &path, report);
        }
    }
}

fn reload_folder_components(
    scene: &mut Scene,
    entity: Entity,
    dir: &Path,
    report: &mut LoadReport,
) {
    // Strip all components from the entity first (we'll re-add from
    // whatever files are still present).
    // bevy_ecs 0.18 doesn't expose a "clear all components on entity"
    // directly; the cheap correct thing is to despawn + respawn and
    // fix up our dir_entity map.
    let new_entity = if entity == scene.root {
        scene.world.despawn(entity);
        let e = scene.world.spawn_empty().id();
        scene.root = e;
        e
    } else {
        scene.world.despawn(entity);
        scene.world.spawn_empty().id()
    };
    scene.dir_entity.insert(dir.to_path_buf(), new_entity);

    let rd = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    let mut touched = false;
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() || !is_toml(&path) {
            continue;
        }
        if let Err(e) = apply_toml_file(scene, new_entity, &path) {
            report.errors.push(e);
        } else {
            report.files_parsed += 1;
            touched = true;
            scene.file_entity.insert(path, new_entity);
        }
    }
    if touched {
        report.entities_touched += 1;
    }
}

fn apply_toml_file(scene: &mut Scene, entity: Entity, path: &Path) -> Result<(), SceneError> {
    let source = fs::read_to_string(path).map_err(|e| SceneError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let parsed: toml::Value =
        toml::from_str(&source).map_err(|e: toml::de::Error| SceneError::Parse {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
    let table = match parsed {
        toml::Value::Table(t) => t,
        _ => {
            return Err(SceneError::Parse {
                path: path.to_path_buf(),
                message: "top-level TOML value must be a table".into(),
            });
        }
    };

    // `_scene.toml` is a reserved metadata file that targets the root
    // entity (handled by being loaded at the root folder anyway);
    // otherwise no special logic is needed — sections in any file
    // apply to this entity.
    for (section, value) in &table {
        if section.starts_with(RESERVED_PREFIX) {
            continue;
        }
        let loader = find_loader(section).ok_or_else(|| SceneError::UnknownComponent {
            path: path.to_path_buf(),
            section: section.clone(),
        })?;
        let mut entity_mut = scene.world.entity_mut(entity);
        (loader.load)(&mut entity_mut, value).map_err(|message| SceneError::ComponentLoad {
            path: path.to_path_buf(),
            section: section.clone(),
            message,
        })?;
    }

    log::debug!(
        "scene: loaded {} into entity {:?}",
        path.display(),
        entity
    );
    Ok(())
}

fn is_toml(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("toml"))
}

/// Best-effort canonicalization. If the path doesn't exist (e.g.,
/// during a `Removed` event we're processing after the file is gone)
/// we keep the original — `entity_for_dir` and `file_entity` lookups
/// will then key off the same form `dir_entity` was populated with at
/// initial load, since `Scene::empty` canonicalized the root.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

// ───────────────────────── Utility ─────────────────────────

/// Convenience wrapper for the editor: create a scene for `root_dir`
/// and perform the initial full load in one step.
pub fn open(root_dir: impl Into<PathBuf>) -> (Scene, LoadReport) {
    let mut scene = Scene::empty(root_dir);
    let report = reload_full(&mut scene);
    (scene, report)
}
