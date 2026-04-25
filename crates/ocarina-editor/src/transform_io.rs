//! Read / splice the `[transform]` section of an entity's `.toml` file.
//!
//! The gizmo lives in the editor; the bytes on disk are the source of
//! truth. We treat the in-memory ECS Transform as a *cache* — gizmo
//! drags update both the cache (for immediate visual feedback) and the
//! file (for persistence). The watcher then replays the file change
//! through the same loader path everyone else uses.
//!
//! Formatting and surrounding fields are preserved via `toml_edit`.

use std::path::{Path, PathBuf};

use glam::Vec3;

/// Find the `.toml` file inside `entity_dir` whose contents declare
/// a top-level `[transform]` section. Returns the first match. Used to
/// answer "where do I write the new transform?" when the gizmo commits.
#[must_use]
pub fn find_transform_file(entity_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(entity_dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("toml"))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with('_'))
        })
        .collect();
    // Stable order so repeated calls hit the same file.
    candidates.sort();
    for path in candidates {
        if file_has_transform_section(&path) {
            return Some(path);
        }
    }
    None
}

fn file_has_transform_section(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(doc) = text.parse::<toml_edit::DocumentMut>() else {
        return false;
    };
    doc.get("transform").is_some()
}

/// Update `[transform]` in `path` to the given position / rotation /
/// scale. Other fields, comments, and formatting are preserved. If the
/// file or section doesn't exist it's created.
pub fn write_transform(
    path: &Path,
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut doc = text
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| format!("parse {}: {e}", path.display()))?;

    let section = doc
        .entry("transform")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| format!("`transform` in {} is not a table", path.display()))?;

    section.insert("position", toml_edit::value(vec3_array(position)));
    section.insert("rotation", toml_edit::value(vec3_array(rotation)));
    section.insert("scale", toml_edit::value(vec3_array(scale)));

    std::fs::write(path, doc.to_string())
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

fn vec3_array(v: Vec3) -> toml_edit::Array {
    let mut arr = toml_edit::Array::new();
    arr.push(round(v.x) as f64);
    arr.push(round(v.y) as f64);
    arr.push(round(v.z) as f64);
    // Compact "[1.0, 2.0, 3.0]" rather than the default formatter's
    // tendency to break arrays across lines.
    arr.set_trailing("");
    arr.fmt();
    arr
}

/// Trim float noise: round to 4 decimals so 1.0000001 doesn't end up in
/// the file every time the gizmo is bumped one pixel.
fn round(x: f32) -> f32 {
    (x * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_other_sections() {
        let path = std::env::temp_dir().join(format!(
            "ocarina-transform-io-{}.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "[transform]\nposition = [0.0, 0.0, 0.0]\n\n[name]\nvalue = \"keepme\"\n",
        )
        .unwrap();

        write_transform(
            &path,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 90.0, 0.0),
            Vec3::ONE,
        )
        .unwrap();

        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.contains("position = [1.0, 2.0, 3.0]"), "{after}");
        assert!(after.contains("[name]"), "{after}");
        assert!(after.contains("value = \"keepme\""), "{after}");
        let _ = std::fs::remove_file(&path);
    }
}
