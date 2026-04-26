//! Project-relative reference to an asset on disk.
//!
//! Components store `AssetPath` rather than `String` for fields that
//! reference files (meshes, textures, scripts, etc.) so the rest of the
//! engine can recognize them as assets:
//!
//! - **Bake** walks every component instance and harvests every
//!   `AssetPath` value to build the inclusion set; files not referenced
//!   are not packaged.
//! - **LSP** can validate the path exists at edit time.
//! - **Editor** widgets can offer a file-picker for these fields.
//!
//! Paths are always **project-relative** (`"assets/wood.png"`,
//! `"scripts/movement.rs"`) — never absolute. The runtime resolves them
//! against the project root.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A project-relative path to a file. Serialises as a TOML string, so
/// authoring stays `source = "assets/wood.png"` — no special syntax.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetPath(pub String);

impl AssetPath {
    /// Construct from anything string-like.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Borrow as a `&str` for path comparisons / hashing.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Borrow as a `&Path` for filesystem operations.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }

    /// Resolve against a project root, producing an absolute path.
    /// Used by runtime asset loaders and by bake's harvester.
    #[must_use]
    pub fn resolve(&self, project_root: &Path) -> PathBuf {
        project_root.join(&self.0)
    }

    /// True for the empty path — equivalent to "no asset assigned".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for AssetPath {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for AssetPath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_as_toml_string() {
        // Wrap in a struct so toml has a top-level table to start from.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            source: AssetPath,
        }
        let h = Holder {
            source: AssetPath::new("assets/wood.png"),
        };
        let s = toml::to_string(&h).unwrap();
        assert!(s.contains("source = \"assets/wood.png\""), "{s}");
        let back: Holder = toml::from_str(&s).unwrap();
        assert_eq!(back.source.as_str(), "assets/wood.png");
    }

    #[test]
    fn resolves_against_root() {
        let p = AssetPath::new("assets/wood.png");
        let resolved = p.resolve(Path::new("/proj"));
        assert_eq!(resolved, PathBuf::from("/proj/assets/wood.png"));
    }
}
