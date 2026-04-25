//! Open project: wraps a [`Scene`] + [`SceneWatcher`] + a cached
//! filesystem snapshot for the file-browser UI.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ocarina_scene::{LoadReport, Scene, SceneEvent, SceneWatcher};

/// Cached snapshot of the project directory tree. Rebuilt on demand
/// after filesystem events; the file-browser pane reads this.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    /// How many descendant files + dirs this node has (itself excluded).
    pub fn descendant_count(&self) -> usize {
        self.children.iter().map(|c| 1 + c.descendant_count()).sum()
    }
}

pub struct Project {
    pub scene: Scene,
    pub watcher: Option<SceneWatcher>,
    pub tree: TreeNode,
    pub last_report: LoadReport,
    pub last_load_time: Instant,
    pub errors: Vec<String>,
    /// Currently-highlighted node in the file browser (may be a file or
    /// a directory). Drives the selection highlight, and — when it's a
    /// file — Phase 3+ will open it in the Editor pane.
    pub selected: Option<PathBuf>,
    /// Which directories are expanded in the file browser.
    pub expanded: std::collections::HashSet<PathBuf>,
    /// When `selected` changes, set this flag so the UI knows to scroll
    /// the selected row into view on the next frame.
    pub scroll_to_selected: bool,
}

impl Project {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        // Canonicalize at the door so `tree`, `scene`, and `watcher`
        // all observe the same path form. Watcher events on macOS
        // (FSEvents) are always absolute canonical, and `Scene::empty`
        // canonicalizes too — staying consistent here keeps the file
        // browser's `selected` paths matching `Scene::file_entity`
        // keys, which other systems rely on.
        let root = root.into();
        let root = root.canonicalize().unwrap_or(root);
        let (scene, report) = ocarina_scene::loader::open(&root);
        let tree = build_tree(&root);
        let watcher = match SceneWatcher::start(&root) {
            Ok(w) => Some(w),
            Err(e) => {
                log::warn!("scene watcher failed to start: {e}");
                None
            }
        };
        let errors = report
            .errors
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        log::info!(
            "project: loaded {} files, {} entities touched, {} errors",
            report.files_parsed,
            report.entities_touched,
            report.errors.len()
        );
        for e in &report.errors {
            log::warn!("project load error: {e}");
        }
        Self {
            scene,
            watcher,
            tree,
            last_report: report,
            last_load_time: Instant::now(),
            errors,
            selected: None,
            expanded: std::collections::HashSet::new(),
            scroll_to_selected: false,
        }
    }

    /// The currently-selected node, if it's a regular file (not a dir).
    /// Consumers that care specifically about opening files — e.g. the
    /// Editor pane in Phase 3 — read this.
    pub fn selected_file(&self) -> Option<&Path> {
        let sel = self.selected.as_deref()?;
        if sel.is_file() {
            Some(sel)
        } else {
            None
        }
    }

    /// Pull filesystem events from the watcher and apply them to the
    /// scene. Called once per editor frame.
    pub fn pump(&mut self) -> PumpReport {
        let Some(watcher) = &self.watcher else {
            return PumpReport::default();
        };
        let events = watcher.drain();
        if events.is_empty() {
            return PumpReport::default();
        }
        let mut pr = PumpReport::default();
        let mut needs_tree_rebuild = false;
        for ev in events {
            match ev {
                SceneEvent::Changed(path) => {
                    if let Err(e) = ocarina_scene::loader::reload_file(&mut self.scene, &path) {
                        self.errors.push(e.to_string());
                        pr.errors += 1;
                    } else {
                        pr.files_changed += 1;
                    }
                }
                SceneEvent::Removed(path) => {
                    let r = ocarina_scene::loader::remove_file(&mut self.scene, &path);
                    pr.files_removed += 1;
                    pr.errors += r.errors.len();
                    for e in r.errors {
                        self.errors.push(e.to_string());
                    }
                    needs_tree_rebuild = true;
                }
                SceneEvent::Rescan(_) => {
                    pr.full_reloads += 1;
                    let r = ocarina_scene::loader::reload_full(&mut self.scene);
                    pr.errors += r.errors.len();
                    for e in r.errors {
                        self.errors.push(e.to_string());
                    }
                    needs_tree_rebuild = true;
                }
            }
        }
        if needs_tree_rebuild || pr.files_changed > 0 {
            self.tree = build_tree(&self.scene.root_dir);
            self.last_load_time = Instant::now();
        }
        // Cap the error log so a runaway watcher doesn't swell memory.
        if self.errors.len() > 200 {
            self.errors.drain(..self.errors.len() - 200);
        }
        pr
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PumpReport {
    pub files_changed: usize,
    pub files_removed: usize,
    pub full_reloads: usize,
    pub errors: usize,
}

impl PumpReport {
    pub fn any(&self) -> bool {
        self.files_changed + self.files_removed + self.full_reloads + self.errors > 0
    }
}

pub fn build_tree(root: &Path) -> TreeNode {
    let name = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| root.to_str().unwrap_or("<project>"))
        .to_string();
    walk_tree(name, root)
}

fn walk_tree(name: String, path: &Path) -> TreeNode {
    let is_dir = path.is_dir();
    let mut children = Vec::new();
    if is_dir {
        if let Ok(rd) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = rd
                .flatten()
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|s| !s.starts_with('.'))
                })
                .collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let child_path = entry.path();
                let child_name = entry.file_name().to_string_lossy().into_owned();
                children.push(walk_tree(child_name, &child_path));
            }
        }
    }
    TreeNode {
        name,
        path: path.to_path_buf(),
        is_dir,
        children,
    }
}
